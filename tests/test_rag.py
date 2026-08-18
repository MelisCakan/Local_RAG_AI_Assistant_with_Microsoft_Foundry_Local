import json
import sys
import time
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_core import answer_query, initialize_models


def evaluate_test(test, answer):
    category = test["category"]
    answer = answer.strip().lower()

    if category == "answerable":
        return (
        bool(answer)
        and "i don't know based on the provided context." not in answer.lower()
        )

    if category == "unanswerable":
        return "i don't know based on the provided context." in answer

    if category == "edge_case":
        return answer == "please enter a question."

    return False


def main():
    with open("tests/test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    embedding_client, chat_client = initialize_models()

    results = []

    for test in test_cases:
        query = test["query"]

        start_time = time.perf_counter()

        try:
            answer = answer_query(
                query,
                embedding_client,
                chat_client
            )

            elapsed_time = time.perf_counter() - start_time

            passed = evaluate_test(test, answer)

            results.append({
                "id": test["id"],
                "category": test["category"],
                "query": query,
                "answer": answer,
                "time": round(elapsed_time, 2),
                "passed": passed,
            })

        except Exception as e:
            results.append({
                "id": test["id"],
                "category": test["category"],
                "query": query,
                "answer": f"ERROR: {e}",
                "time": round(
                    time.perf_counter() - start_time,
                    2
                ),
                "passed": False,
            })

    print("\n===== TEST RESULTS =====\n")

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"

        print(f"{result['id']} - {status}")
        print(f"Category: {result['category']}")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Response time: {result['time']}s")
        print("-" * 60)

    passed_count = sum(r["passed"] for r in results)

    print("\n===== SUMMARY =====")
    print(f"Passed: {passed_count}/{len(results)}")
    print(f"Failed: {len(results) - passed_count}")


if __name__ == "__main__":
    main()