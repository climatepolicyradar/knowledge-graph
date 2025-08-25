import json
import re
import time
from concurrent.futures import ThreadPoolExecutor


def create_pattern(labels):
    return r"\b(?:" + "|".join(labels) + r")\b"


def main():
    print("Loading benchmark data...")

    # Load the benchmark data created by test_predict_large
    benchmark_file = "../../benchmark_data_CCLW.executive.10014.4470.json"
    with open(benchmark_file, "r") as f:
        benchmark_data = json.load(f)

    print(
        f"Loaded {benchmark_data['total_blocks']} passages from {benchmark_data['document_id']}"
    )

    # Extract texts for batch processing
    texts = [passage["text"] for passage in benchmark_data["passages"]]

    # Load the actual classifier labels from the benchmark data
    case_sensitive_labels = benchmark_data["classifier_labels"]["case_sensitive"]
    case_insensitive_labels = benchmark_data["classifier_labels"]["case_insensitive"]

    # Case-sensitive pattern: matches exactly as provided
    case_sensitive_pattern = re.compile(create_pattern(case_sensitive_labels))

    # Case-insensitive pattern: matches regardless of case
    case_insensitive_pattern = re.compile(
        create_pattern(case_insensitive_labels), re.IGNORECASE
    )

    print(f"Case sensitive labels: {len(case_sensitive_labels)} labels")
    print(f"Case insensitive labels: {len(case_insensitive_labels)} labels")

    def predict(text: str):
        """Predict whether the supplied text contains an instance of the concept."""
        spans = []
        matched_positions = set()

        # Case-sensitive matching
        for match in case_sensitive_pattern.finditer(text):
            start, end = match.span()
            if start != end and not any(start <= p < end for p in matched_positions):
                spans.append(
                    {
                        "text": text[start:end],
                        "start_index": start,
                        "end_index": end,
                    }
                )
                matched_positions.update(range(start, end))

        # Case-insensitive matching
        for match in case_insensitive_pattern.finditer(text):
            start, end = match.span()
            if start != end and not any(start <= p < end for p in matched_positions):
                spans.append(
                    {
                        "text": text[start:end],
                        "start_index": start,
                        "end_index": end,
                    }
                )
                matched_positions.update(range(start, end))

        return spans

    # Single-threaded benchmark
    print("\nRunning single-threaded benchmark...")
    start_time = time.time()

    doc_labels = []
    for text in texts:
        spans = predict(text)
        passage = {"spans": spans, "text": text, "has_matches": len(spans) > 0}
        doc_labels.append(passage)

    end_time = time.time()
    total_time = end_time - start_time

    passages_with_matches = sum(1 for passage in doc_labels if passage["has_matches"])
    total_spans = sum(len(passage["spans"]) for passage in doc_labels)

    print(f"Single-threaded time: {total_time:.4f}s")
    print(f"Passages with matches: {passages_with_matches}")
    print(f"Total spans found: {total_spans}")

    # Multi-threaded benchmark
    print("\nRunning multi-threaded benchmark (2 threads)...")
    start_time_mt = time.time()

    def process_text(text):
        spans = predict(text)
        return {"spans": spans, "text": text, "has_matches": len(spans) > 0}

    doc_labels_mt = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_text, text) for text in texts]
        for future in futures:
            doc_labels_mt.append(future.result())

    end_time_mt = time.time()
    total_time_mt = end_time_mt - start_time_mt

    passages_with_matches_mt = sum(
        1 for passage in doc_labels_mt if passage["has_matches"]
    )
    total_spans_mt = sum(len(passage["spans"]) for passage in doc_labels_mt)

    print(f"Multi-threaded time: {total_time_mt:.4f}s")
    print(f"Passages with matches: {passages_with_matches_mt}")
    print(f"Total spans found: {total_spans_mt}")

    if total_time_mt > 0:
        speedup = total_time / total_time_mt
        improvement_percent = (speedup - 1) * 100
        print(f"Speedup: {speedup:.2f}x ({improvement_percent:.1f}% faster)")

    print(f"Results consistent: {passages_with_matches == passages_with_matches_mt}")


if __name__ == "__main__":
    main()
