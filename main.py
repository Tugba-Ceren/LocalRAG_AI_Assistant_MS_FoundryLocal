import math
from foundry_local_sdk import Configuration, FoundryLocalManager
#Week 2 Exercise 2
import json
import sqlite3
# Knowledge base
documents = [
    "SpotterAI is an autonomous AI-powered security camera designed for local object detection and event reporting.",
    "The SpotterAI SDK supports Python, C++, C#, and JavaScript for local integration and custom alerts.",
    "SpotterAI generates vector embeddings locally on the device to index video metadata and text event logs.",
    "SpotterAI leverages ONNX Runtime and local NPU acceleration to process 4K video feeds at 60 FPS offline.",
    "The SpotterAI model catalog includes pre-trained computer vision and small language models downloadable for on-device execution.",
    "SpotterAI uses retrieval-augmented generation to summarize security events using only locally stored log data.",
    "Vector similarity search in SpotterAI helps security teams query past incidents by semantic intent rather than timestamp.",
    "SpotterAI chat completions generate natural language incident summaries directly from prompt instructions and camera logs.",
]
# 2. SQLite Database Functions

#Week 2 ,Exercise 2
DB_NAME = "spotterai_rag.db"
#Define the function 

def init_db():
    """Create SQLite table to store manual text and vector embeddings."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def ingest_documents(embedding_client):
    """Embed documents using Foundry Local SDK and persist them in SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Prevent duplicate insertions
    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    if cursor.fetchone()[0] > 0:
        print("Knowledge base already ingested in SQLite.")
        conn.close()
        return

    print("Generating embeddings and saving to SQLite...")
    response = embedding_client.generate_embeddings(documents)

    for doc_text, item in zip(documents, response.data):
        # Store floating-point embedding array as a JSON string in SQLite
        cursor.execute(
            "INSERT INTO document_chunks (content, embedding) VALUES (?, ?)",
            (doc_text, json.dumps(item.embedding)),
        )

    conn.commit()
    conn.close()
    print(f"Stored {len(documents)} document chunks in {DB_NAME}.")


def fetch_all_documents():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, embedding FROM document_chunks")
    rows = cursor.fetchall()
    conn.close()

   
    db_ids = [row[0] for row in rows]
    db_docs = [row[1] for row in rows]
    db_embeddings = [json.loads(row[2]) for row in rows]
    return db_ids ,db_docs, db_embeddings

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding, doc_embeddings, top_k=2):
    """Return the indices and scores of the top-k most similar documents."""
    scores = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
def answer_query(query, embedding_client, chat_client, db_ids, db_docs, db_embeddings):
    """Week 4 Core Function: Retrieves context from SQLite and generates an answer using local LLM."""
    # 1. Embed query
    query_response = embedding_client.generate_embeddings([query])
    # Access the vector from the response object
    query_embedding = query_response.data[0].embedding
    # 2. Retrieve top matching chunks 
    
    results = find_relevant(query_embedding, db_embeddings, top_k=2)
    
    # 3. Format context with document source labels for Responsible AI citations
    context_blocks = []
    for index, _ in results:
        doc_id = db_ids[index]
        text = db_docs[index]
        context_blocks.append(f"[Document {doc_id}]: {text}")
    
    context = "\n".join(context_blocks)

    # 4. Construct prompt with safety rules and citation requirements
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and polite technical assistant.\n"
                "Answer the user's question accurately using ONLY the provided context.\n"
                "If the context does not contain enough information to answer, state clearly: "
                "'I do not have enough context to answer that question.'\n"
                "Whenever possible, cite the document numbers used in your answer (e.g., '[Document X]').\n\n"
                f"Context:\n{context}"
            ),
        },
        {"role": "user", "content": query},
    ]

    # 5. Stream LLM output cleanly with fallback checks
    print("\nAnswer: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
        if not getattr(chunk, "choices", None):
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta else None
        if content:
            print(content, end="", flush=True)
    print("\n")


def main():
    init_db()
    # Initialize the SDK
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Load the embedding model
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # Save documents into SQLite
    ingest_documents(embedding_client) 

    # Retrieve stored documents and vector embeddings straight from SQLite
    db_ids, db_docs, db_embeddings = fetch_all_documents()
    print(f"Loaded {len(db_embeddings)} documents directly from SQLite.")
    # Load the chat model
    chat_model = manager.catalog.get_model("qwen2.5-0.5b")
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nModels loaded. Ready for questions.")
    print("\nThe knowledge base contains information about:")
    print("  - SpotterAI  features and architecture")
    print("  - Supported programming languages")
    print("  - Embedding models and vector search")
    print("  - ONNX Runtime inference")
    print("  - The model catalog")
    print("  - RAG and chat completions")
    print("\nExample questions:")
    print('  "What languages does the SpotterAI support?"')
    print('  "How does SpotterAI works?"')
    print('  "What is retrieval-augmented generation?"')
    print('\nType "quit" to exit.\n')

    # Interactive query loop
    while True:
        query = input("Question: ").strip()
        if not query or query.lower() == "quit":
            break
        answer_query(
            query,
            embedding_client,
            chat_client,
            db_ids,
            db_docs,
            db_embeddings,
        )

    
     

    # Clean up
    embedding_model.unload()
    chat_model.unload()
    print("Models unloaded. Done!")


if __name__ == "__main__":
    main()