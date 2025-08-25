import json
import time

def main():
    print("Loading benchmark data...")
    
    # Load the benchmark data created by test_predict_large
    benchmark_file = "../benchmark_data_CCLW.executive.10014.4470.json"
    with open(benchmark_file, "r") as f:
        benchmark_data = json.load(f)
    
    print(f"Loaded {benchmark_data['total_blocks']} passages from {benchmark_data['document_id']}")

    # Test Rust extension with FFI overhead
    try:
        import pyru_rust
        print("\nRunning Rust extension benchmark (with FFI overhead)...")
        
        # Convert benchmark data to JSON string for Rust
        benchmark_json = json.dumps(benchmark_data)
        
        rust_time, rust_passages, rust_spans, rust_total = pyru_rust.batch_regex_predict(benchmark_json)
        
        print(f"Rust extension time: {rust_time:.4f}s")
        print(f"Passages with matches: {rust_passages}")
        print(f"Total spans found: {rust_spans}")
        print(f"Total passages processed: {rust_total}")
        
    except ImportError as e:
        print(f"\nRust extension not available: {e}")
        print("Run 'uv run maturin develop --release' in rust/ directory first")
        print("Then make sure you're running with 'uv run python main.py'")


if __name__ == "__main__":
    main()