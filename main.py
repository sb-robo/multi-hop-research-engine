import os
from dotenv import load_dotenv
import streamlit as st
from google import genai
from sentence_transformers import CrossEncoder

from utils.utils import get_pdf_text, get_pdf_chunks, get_vector_embeddings
from multi_hop_rag.multi_hop_rag import multi_hop_rag

load_dotenv()

# ALL Env Variable
GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY"))

client = genai.Client(api_key=GOOGLE_API_KEY)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def main():
    st.set_page_config(page_title="Q&A chatbot")
    st.title("Q&A Chat with Multiple PDF")

    st.sidebar.subheader("Menu:")
    pdf_docs = st.sidebar.file_uploader("Please Upload pdf files: ", accept_multiple_files=True)
    process_button = st.sidebar.button("Process PDF")

    if "processed" not in st.session_state:
        st.session_state.processed = False

    if pdf_docs and process_button:
        pdf_content = get_pdf_text(pdf_docs)
        chunks = get_pdf_chunks(pdf_content)
        message = get_vector_embeddings(chunks, client)

        if message == "Success":
            st.success("Your PDFs are processed. You May ask your question now")
        else:
            st.error(f"Error while creating embeddings: {message}")

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
            for response in responses:
                st.write(response.text)

            for chunk in similar_chunks:
                st.write(f"Source: {chunk.get('source')}, Page: {chunk.get('page')}")


if __name__ == "__main__":
    main()
