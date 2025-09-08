from google.genai import types
from typing import List, Dict
import numpy as np
import pickle as pk


def get_similar_chunks(query: str, client) -> List[Dict] | Dict:
    with open("./vectorindex/vectorindex.pkl", "rb") as f:
        data = pk.load(f)

    query_embeddings = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    ).embeddings
    query_embeddings = np.array(query_embeddings[0].values).reshape(1, -1)

    index = data["index"]
    metadata = data["metadata"]

    _, Idxs = index.search(query_embeddings, k=20)

    if len(Idxs) == 0:
        return [
            {"Not found": "No relevant chunks found, Please send a valid and query related to docs"}
        ]

    return [metadata[idx] for idx in Idxs[0]]


def reranker_cross_encoders(query: str, similar_chunks: List[Dict], cross_encoder) -> List[Dict]:
    doc = [(query, chunk["chunk"]) for chunk in similar_chunks]
    rank_scores = cross_encoder.predict(doc)

    for chunk, score in zip(similar_chunks, rank_scores):
        chunk["score"] = float(score)

    similar_chunks = sorted(similar_chunks, key=lambda x: x["score"], reverse=True)
    return similar_chunks[:5]


def generate_structured_reasoning(
    original_query: str, current_query: str, doc_chunks: List[Dict], reasoning_trace: Dict, client
) -> List[Dict]:
    """Generate structured reasoning for multi-hop RAG system."""

    if not reasoning_trace.get("hops"):
        previous_context = "It is the first hop"
    else:
        previous_reasoning = []
        for i, hop_context in enumerate(reasoning_trace.get("hops")):
            previous_reasoning.append(f"reasoning{i + 1}: {hop_context.get('reasoning')}")
            previous_reasoning.append(f"missing_info{i + 1}: {hop_context.get('missing_info')}")

        previous_context = "\n".join(previous_reasoning)

    context = f"""You are an Intelligent AI, You have to analyze Doc chunks, original query and Current sub Query. 
    After analyzing with your reasoning skill identify the missing info's in the doc chunks.
    If the query is completely irrelevant to the docs state irrelevant query
    
    Original Query: {original_query}
    Current Sub Query: {current_query}
    Docs chunk: {[chunk.get("chunk") for chunk in doc_chunks]}
    Previous reasoning trace: {previous_context}

    INSTRUCTIONS:
    1. Carefully analyze how the document chunks relate to both the original query and current sub-query
    2. Consider the previous reasoning steps to avoid redundancy
    3. Identify specific missing information needed to progress toward a complete answer
    4. Provide clear, logical reasoning for your analysis

    Give ur reasoning in below format:
    reasoning: [your reasoning - what the chunks contain and how they related to query]
    missing_info: [Specific missing info related to the query or 'No missing info' if chunks are sufficient]
    irrelevant: ['irrelevant query' if the query is completely irrelevant to the doc chunks else 'relevant query']
    """

    response = client.models.generate_content(model="gemini-2.5-flash", contents=context)

    reasoning = ""
    missing_info = ""
    irrelevant = ""

    lines = response.text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("reasoning:"):
            reasoning = line[len("reasoning:") :].strip()
        if line.lower().startswith("missing_info:"):
            missing_info = line[len("missing_info:") :].strip()
        if line.lower().startswith("irrelevant:"):
            irrelevant = line[len("irrelevant:") :].strip()

    reasoning_data = {
        "reasoning": reasoning,
        "missing_info": missing_info,
        "irrelevant": irrelevant,
    }

    return reasoning_data


def generate_next_query(original_query: str, current_query: str, missing_info: str, client) -> str:
    """Query Generation based on missing_info"""

    context = f"""Generate a query or sub question based on the missing info that will help to find relevants document chunks.
    
    Original Query: {original_query}
    Current Query: {current_query}
    Missing Info: {missing_info}

    Analyze the missing info and Generate a targeted query or sub-question and follow the below instructions

    The Query should
    1. Be different from the current query
    2. Be concise and focused
    3. Address the specific missing information identified in the reasoning step

    Provide only the query:
    """

    query = ""
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=context)
        query = response.text.strip()
    except Exception as e:
        print(f"Encountered error while generating the next query. Error: {e}")
        return "No query generated"

    return query


def remove_duplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    non_duplicate_chunks = list({chunk["chunk_id"]: chunk for chunk in chunks}.values())
    return non_duplicate_chunks


def get_final_structred_response(context: List[Dict], reasoning_trace: Dict, query: str, client):
    reasoning_trace_data = []
    reasoning = ""

    for i, hop in enumerate(reasoning_trace.get("hops", "")):
        reasoning_trace_data.append(f"Query_{i} - {hop.get('current_query', '')}")
        reasoning_trace_data.append(f"Reasoning_{i} - {hop.get('reasoning', '')}")
        if hop.get("reasoning", "").lower() not in ("no missing info", "none", "nothing", ""):
            reasoning_trace_data.append(f"Missing_Info{i} - {hop.get('missing_info', '')}")
        reasoning_trace_data.append("")

    reasoning = "\n".join(reasoning_trace_data)

    template = f"""You are a QA chat bot. Give a comprehensive answer based on the multi-hop reasoning trace and document context

    Original_Question : {query}
    Reasoning : {reasoning}
    Document_Context_with_metadata : {context}
    
    Generated Answer Instructions:
    1. The answer should be clear, concise, comprehensive and focused
    2. Use reasoning trace to analyze reasoning and missing info of each hop
    3. Structure your answer logically
    
    Answer:"""

    response = client.models.generate_content_stream(model="gemini-2.5-flash", contents=template)
    return response


def multi_hop_rag(query: str, client, cross_encoder):
    no_of_hops = 3
    all_retrived_chunks = []
    reasoning_trace = {"original_query": query, "hops": []}
    current_query = query
    reranked_chunks = None
    irrelevant_query_flag = 0

    for hop_no in range(no_of_hops):
        reasoning_step = {"hop_no": hop_no, "current_query": current_query}

        retrieved_chunks = get_similar_chunks(current_query, client)
        if isinstance(reranked_chunks, dict) and "Not found" in retrieved_chunks:
            retrieved_chunks = []
        else:
            reranked_chunks = reranker_cross_encoders(
                current_query, retrieved_chunks, cross_encoder
            )
            all_retrived_chunks.extend(reranked_chunks)

        reasoning_data = generate_structured_reasoning(
            query, current_query, reranked_chunks, reasoning_trace, client
        )

        if reasoning_data:
            reasoning_step["reasoning"] = reasoning_data.get("reasoning", "")
            reasoning_step["missing_info"] = reasoning_data.get("missing_info", "")
            reasoning_step["irrelevant"] = reasoning_data.get("irrelevant", "")

        reasoning_trace.get("hops").append(reasoning_step)

        if reasoning_step.get("irrelevant", "").lower() not in ("relevant query", "relevant"):
            irrelevant_query_flag += 1

            if reasoning_step["missing_info"].lower() in ("no missing info", "none", "nothing"):
                reasoning_step["missing_info"] = "Query is irrelevant"

        if irrelevant_query_flag > 1:
            return "Invalid Query"

        missing_info_flag = ("no missing info", "none", "nothing")
        if (
            hop_no < no_of_hops - 1
            and reasoning_step["missing_info"].lower() not in missing_info_flag
        ):
            current_query = generate_next_query(
                query, current_query, reasoning_step["missing_info"], client
            )
        else:
            break

    if len(all_retrived_chunks) == 0:
        return "No chunks found"

    filtered_chunks = remove_duplicate_chunks(all_retrived_chunks)
    response = get_final_structred_response(filtered_chunks, reasoning_trace, query, client)

    return (response, filtered_chunks)
