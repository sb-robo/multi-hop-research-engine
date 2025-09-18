import os
import datetime
import streamlit as st
from fpdf import FPDF
from google import genai
from dotenv import load_dotenv

from sentence_transformers import CrossEncoder

from utils.utils import get_pdf_text, get_pdf_chunks, get_vector_embeddings
from multi_hop_rag.multi_hop_rag import multi_hop_rag

load_dotenv()

# ALL Env Variable
GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY"))

client = genai.Client(api_key=GOOGLE_API_KEY)


@st.cache_resource
def load_cross_encoder():
    """Load CrossEncoder with proper device handling"""
    try:
        # Force CPU usage to avoid device conflicts
        device = "cpu"
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device)
        return cross_encoder
    except Exception as e:
        st.error(f"Error while loading CrossEncoder: {e}")
        return None


cross_encoder = load_cross_encoder()


def main():
    st.set_page_config(page_title="Q&A chatbot")
    st.markdown(
        """
    <style>
        .block-container {
            max-width: 95% !important;  /* default is ~ 700px */
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("Q&A Chat with Multiple PDF")

    st.sidebar.subheader("Menu:")
    pdf_docs = st.sidebar.file_uploader("Please Upload pdf files: ", accept_multiple_files=True)
    process_button = st.sidebar.button("Process PDF")

    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "history" not in st.session_state:
        st.session_state.history = []

    if pdf_docs and process_button:
        pdf_content = get_pdf_text(pdf_docs)
        chunks = get_pdf_chunks(pdf_content)
        message = get_vector_embeddings(chunks, client)

        if message == "Success":
            st.success("Your PDFs are processed. You May ask your queries now")
        else:
            st.error(f"Error while creating embeddings: {message}")

    # ---- PAGE LAYOUT WITH RIGHT SIDEBAR ----
    col_main, col_right = st.columns([4, 2])

    with col_main:
        query = st.text_input("Please ask you query: ")
        ask_button = st.button("Ask")

        if query and ask_button:
            response = multi_hop_rag(query, client, cross_encoder)

            if isinstance(response, str) and (
                response == "No chunks found" or response == "Invalid Query"
            ):
                st.write("No relevated information found! Please ask a valid query!")
            else:
                responses, similar_chunks = response
                answer = "\n".join([resp.text for resp in responses])
                st.write(answer)
                st.session_state.history.append({"query": query, "answer": answer})

                for chunk in similar_chunks:
                    st.write(f"Source: {chunk.get('source')}, Page: {chunk.get('page')}")

    with col_right:
        if st.session_state.history:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=10)
                pdf.set_auto_page_break(auto=True, margin=15)

                for chat in st.session_state.history:
                    for key, value in chat.items():
                        clean_text = "".join(c for c in str(value) if ord(c) < 128)
                        label = "Q" if "query" in key.lower() else "A"
                        pdf.multi_cell(0, 6, f"{label}: {clean_text}")
                        pdf.ln(3)
                    pdf.ln(5)

                pdf_output = bytes(pdf.output(dest="S"))

                st.download_button(
                    label="Download PDF",
                    data=pdf_output,
                    file_name=f"chat_history{datetime.datetime.now()}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Export failed - try again: {e}")

            for i, chat in enumerate(reversed(st.session_state.history), 1):
                with st.expander(f"Q{i}: {chat['query'][:40]}..."):
                    st.markdown(f"**Q:** {chat['query']}")
                    st.markdown(f"**A:** {chat['answer'][:300]}...")
        else:
            st.info("No chat history yet.")


if __name__ == "__main__":
    main()
