import time
# Import from main.py 
from main import (
    answer_query,
    fetch_all_documents,
    init_db,
    Configuration,
    FoundryLocalManager,
)
# Defined test cases
TEST_CASES = [
    # 1. Answerable Queries (In-scope)
    {
        "type": "In-Scope",
        "query": "What programming languages does the SpotterAI SDK support?",
        "expected_keywords": ["Python", "C++", "C#", "JavaScript"],
        "should_cite": True,
    },
    {
        "type": "In-Scope",
        "query": "What hardware acceleration does SpotterAI use for 4K video?",
        "expected_keywords": ["ONNX", "NPU"],
        "should_cite": True,
    },
    # 2. Unanswerable Queries (Out-of-scope / Fallback)
    {
        "type": "Out-of-Scope",
        "query": "How much does a SpotterAI security camera cost?",
        "expected_keywords": ["do not have enough context"],
        "should_cite": False,
    },
    {
        "type": "Out-of-Scope",
        "query": "What is the battery life of the device?",
        "expected_keywords": ["do not have enough context"],
        "should_cite": False,
    },
    # 3. Edge Cases
    {
        "type": "Edge Case",
        "query": "Tell me about security.",
        "expected_keywords": ["SpotterAI"],
        "should_cite": True,
    },
]


def run_evaluation_suite():
    init_db()

    # Load SDK models once
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    emb_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    emb_model.load()
    emb_client = emb_model.get_embedding_client()

    chat_model = manager.catalog.get_model("qwen2.5-0.5b")
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    db_ids, db_docs, db_embeddings = fetch_all_documents()

    print("=" * 60)
    print("RUNNING WEEK 5 SYSTEM EVALUATION")
    print("=" * 60)

    results_log = []

    for idx, test in enumerate(TEST_CASES, 1):
        print(f"\n[Test {idx}/{len(TEST_CASES)}] Type: {test['type']}")
        print(f"Query: '{test['query']}'")

        start_time = time.time()

        # Run query through pipeline
        answer_query(
            test["query"],
            emb_client,
            chat_client,
            db_ids,
            db_docs,
            db_embeddings,
        )

        elapsed = time.time() - start_time
        print(f"Latency: {elapsed:.2f} seconds")

    # Clean up
    emb_model.unload()
    chat_model.unload()


if __name__ == "__main__":
    run_evaluation_suite()