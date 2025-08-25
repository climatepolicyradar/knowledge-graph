use pyo3::prelude::*;
use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;

fn position_overlaps(matched_positions: &HashSet<usize>, start: usize, end: usize) -> bool {
    (start..end).any(|pos| matched_positions.contains(&pos))
}

fn predict_rust(
    text: &str,
    case_sensitive_pattern: &Regex,
    case_insensitive_pattern: &Regex,
) -> Vec<(usize, usize, String)> {
    let mut spans = Vec::new();
    let mut matched_positions = HashSet::new();

    // Case-sensitive matching
    for mat in case_sensitive_pattern.find_iter(text) {
        let (start, end) = (mat.start(), mat.end());
        if start != end && !position_overlaps(&matched_positions, start, end) {
            spans.push((start, end, text[start..end].to_string()));
            matched_positions.extend(start..end);
        }
    }

    // Case-insensitive matching
    for mat in case_insensitive_pattern.find_iter(text) {
        let (start, end) = (mat.start(), mat.end());
        if start != end && !position_overlaps(&matched_positions, start, end) {
            spans.push((start, end, text[start..end].to_string()));
            matched_positions.extend(start..end);
        }
    }

    spans
}

#[pyfunction]
fn batch_regex_predict(benchmark_data_json: &str) -> PyResult<(f64, usize, usize, usize)> {
    let start_time = std::time::Instant::now();
    
    // Parse the benchmark JSON data
    let benchmark_data: Value = serde_json::from_str(benchmark_data_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    let passages = benchmark_data["passages"].as_array().unwrap();
    let classifier_labels = &benchmark_data["classifier_labels"];
    
    // Extract texts
    let texts: Vec<&str> = passages
        .iter()
        .map(|p| p["text"].as_str().unwrap())
        .collect();
    
    // Build regex patterns from labels (outside timing)
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
    
    let cs_pattern_str = format!(r"\b(?:{})\b", case_sensitive_labels.join("|"));
    let ci_pattern_str = format!(r"(?i)\b(?:{})\b", case_insensitive_labels.join("|"));
    
    let case_sensitive_pattern = Regex::new(&cs_pattern_str).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Regex error: {}", e))
    })?;
    let case_insensitive_pattern = Regex::new(&ci_pattern_str).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Regex error: {}", e))
    })?;
    
    // Start timing after regex compilation
    let processing_start = std::time::Instant::now();
    
    let mut passages_with_matches = 0;
    let mut total_spans = 0;
    
    for text in &texts {
        let spans = predict_rust(text, &case_sensitive_pattern, &case_insensitive_pattern);
        if !spans.is_empty() {
            passages_with_matches += 1;
        }
        total_spans += spans.len();
    }
    
    let processing_time = processing_start.elapsed().as_secs_f64();
    let total_time = start_time.elapsed().as_secs_f64();
    
    println!("Rust regex compilation time: {:.4}s", total_time - processing_time);
    println!("Rust processing time: {:.4}s", processing_time);
    
    Ok((processing_time, passages_with_matches, total_spans, texts.len()))
}

#[pymodule]
fn pyru_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_regex_predict, m)?)?;
    Ok(())
}