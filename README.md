# Multi-Hop RAG

Multi-Hop RAG is a Streamlit-based Q&A chatbot that enables multi-hop reasoning over multiple PDF documents using Google Gemini embeddings and cross-encoder reranking.

## Features

-   Upload and process multiple PDF files.
-   Extracts and chunks PDF text for semantic search.
-   Embeds document chunks using Gemini embeddings.
-   Retrieves and reranks relevant chunks for multi-hop question answering.

## Usage

1. **Install dependencies**  
   Ensure Python 3.11+ is installed.  
   Install [uv](https://github.com/astral-sh/uv):

    ```sh
    pip install uv
    ```

    Then install project dependencies:

    ```sh
    uv sync
    ```

2. **Set up environment**  
   Create a `.env` file with your Google API key:

    ```
    GOOGLE_API_KEY=your_api_key_here
    ```

3. **Run the app**

    ```sh
    un run streamlit run main.py
    ```

4. **Interact**
    - Upload PDFs in the sidebar.
    - Process PDFs.
    - Ask questions in the main interface.

## Project Structure

-   `main.py` – Streamlit app entry point.
-   `utils/` – PDF processing and embedding utilities.
-   `multi_hop_rag/` – Multi-hop retrieval and reasoning logic.
-   `vectorindex/` – Stores vector index for document search.

##
