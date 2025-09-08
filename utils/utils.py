import os
import faiss
import pickle as pk
import numpy as np
from google.genai import types
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_pdf_text(pdf_docs: List) -> List[Dict]:
    docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            docs.append({"text": text, "source": pdf.name, "page": page_num + 1})
    return docs


def get_pdf_chunks(docs: List[Dict]) -> List[Dict]:
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for doc in docs:
        start_id = len(all_chunks)
        chunks = text_splitter.split_text(doc["text"])
        all_chunks.extend(
            [
                {
                    "chunk_id": start_id + i,
                    "chunk": chunk.strip(),
                    "source": doc["source"],
                    "page": doc["page"],
                }
                for i, chunk in enumerate(chunks)
            ]
        )

    return all_chunks


def get_vector_embeddings(chunks: List[Dict], client):
    contents = [chunk.get("chunk") for chunk in chunks]
    print(f"Total chunks to process: {len(contents)}")

    batch_size = 60
    all_embeddings = []

    for i in range(0, len(contents), batch_size):
        batch_contents = contents[i : i + batch_size]

        try:
            embed = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch_contents,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )

            batch_embeddings = np.array([embeddings.values for embeddings in embed.embeddings])
            all_embeddings.append(batch_embeddings)

        except Exception as ex:
            return ex

    all_embeddings = np.vstack(all_embeddings)
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings.astype("float32"))
    data = {"index": index, "metadata": chunks}

    if not os.path.exists("./vectorindex"):
        os.mkdir("./vectorindex")

    with open("./vectorindex/vectorindex.pkl", "wb") as f:
        pk.dump(data, f)

    return "Success"
