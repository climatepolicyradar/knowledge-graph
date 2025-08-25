use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::time::Instant;

#[derive(Debug)]
struct Span {
    text: String,
    start_index: usize,
    end_index: usize,
}

#[derive(Debug)]
struct Passage {
    spans: Vec<Span>,
    has_matches: bool,
}

fn position_overlaps(matched_positions: &HashSet<usize>, start: usize, end: usize) -> bool {
    (start..end).any(|pos| matched_positions.contains(&pos))
}

fn predict(
    text: &str,
    case_sensitive_pattern: &Regex,
    case_insensitive_pattern: &Regex,
) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut matched_positions = HashSet::new();

    // Case-sensitive matching
    for mat in case_sensitive_pattern.find_iter(text) {
        let (start, end) = (mat.start(), mat.end());
        if start != end && !position_overlaps(&matched_positions, start, end) {
            spans.push(Span {
                text: text[start..end].to_string(),
                start_index: start,
                end_index: end,
            });
            matched_positions.extend(start..end);
        }
    }

    // Case-insensitive matching
    for mat in case_insensitive_pattern.find_iter(text) {
        let (start, end) = (mat.start(), mat.end());
        if start != end && !position_overlaps(&matched_positions, start, end) {
            spans.push(Span {
                text: text[start..end].to_string(),
                start_index: start,
                end_index: end,
            });
            matched_positions.extend(start..end);
        }
    }

    spans
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading benchmark data for Rust test...");

    // Load the benchmark data
    let benchmark_file = "../../benchmark_data_CCLW.executive.10014.4470.json";
    let benchmark_data_str = fs::read_to_string(benchmark_file)?;
    let benchmark_data: Value = serde_json::from_str(&benchmark_data_str)?;

    let passages = benchmark_data["passages"].as_array().unwrap();
    let classifier_labels = &benchmark_data["classifier_labels"];

    println!(
        "Loaded {} passages from {}",
        benchmark_data["total_blocks"],
        benchmark_data["document_id"].as_str().unwrap()
    );

    // Extract texts
    let texts: Vec<&str> = passages
        .iter()
        .map(|p| p["text"].as_str().unwrap())
        .collect();

    // Build regex patterns from labels
    let case_sensitive_labels: Vec<&str> = classifier_labels["case_sensitive"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    let case_insensitive_labels: Vec<&str> = classifier_labels["case_insensitive"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    // Create patterns
    let cs_pattern_str = format!(r"\b(?:{})\b", case_sensitive_labels.join("|"));
    let ci_pattern_str = format!(r"(?i)\b(?:{})\b", case_insensitive_labels.join("|"));

    let case_sensitive_pattern = Regex::new(&cs_pattern_str)?;
    let case_insensitive_pattern = Regex::new(&ci_pattern_str)?;

    println!(
        "Case sensitive labels: {} labels",
        case_sensitive_labels.len()
    );
    println!(
        "Case insensitive labels: {} labels",
        case_insensitive_labels.len()
    );

    // Single-threaded benchmark
    println!("\nRunning Rust single-threaded benchmark...");
    let start_time = Instant::now();

    let mut doc_labels = Vec::new();
    for text in &texts {
        let spans = predict(text, &case_sensitive_pattern, &case_insensitive_pattern);
        doc_labels.push(Passage {
            has_matches: !spans.is_empty(),
            spans,
        });
    }

    let total_time = start_time.elapsed().as_secs_f64();

    // Calculate metrics
    let passages_with_matches = doc_labels.iter().filter(|p| p.has_matches).count();
    let total_spans: usize = doc_labels.iter().map(|p| p.spans.len()).sum();

    println!("Rust single-threaded time: {:.4}s", total_time);
    println!("Passages with matches: {}", passages_with_matches);
    println!("Total spans found: {}", total_spans);
    println!("Total passages processed: {}", doc_labels.len());

    Ok(())
}
