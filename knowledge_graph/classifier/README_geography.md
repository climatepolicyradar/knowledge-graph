# Geography Classifier

This document describes the geography classifier architecture, implementation details, and experimental comparison between two approaches: GliNER + Text Search and spaCy Entity Linker.

## Problem Statement

- **200 geography classes** to classify (countries, cities, regions)
- **20 million texts** to process
- **Zero-shot** classification (no training data, working from Wikidata labels)
- **Span-level extraction** with proper entity boundaries
- **Contextual disambiguation** needed (e.g., "Georgia" → country vs. US state)
- **Accuracy critical**, GPU available
- Must integrate with existing `Classifier` API

## Architecture Overview

Both implementations follow a two-stage pipeline pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Backend (Singleton)                      │
│                                                                  │
│  ┌──────────────────────┐    ┌────────────────────────────────┐ │
│  │  NER Model           │    │  Entity Linking                │ │
│  │  (extracts spans)    │───▶│  (maps to Wikidata IDs)        │ │
│  └──────────────────────┘    └────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Disambiguation (for ambiguous entities like "Georgia")   │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GeographyClassifier                            │
│                (Implements Classifier ABC)                       │
│                                                                  │
│  - One instance per geography Concept                           │
│  - Filters backend results to its specific WikibaseID           │
│  - Maintains API compatibility with existing classifiers        │
└─────────────────────────────────────────────────────────────────┘
```

## Two Approaches

### Approach A: GliNER + Text Search (Recommended)

**Location:** `knowledge_graph/classifier/geography.py`

**Classes:**
- `LabelIndex` - Maps text labels to Wikidata IDs
- `GeographyClassifierBackend` - Singleton managing GliNER model
- `GeographyClassifier` - Classifier facade for individual concepts
- `GeographyBatchPredictor` - Efficient bulk prediction

**How it works:**
1. **NER Stage:** GliNER extracts geographic entity spans using type prompts ("country", "city", "region", etc.)
2. **Entity Linking:** Look up extracted text in a pre-built label index from Wikidata labels
3. **Disambiguation:** For ambiguous labels (e.g., "Georgia"), use embedding similarity between context and entity descriptions

**Pros:**
- Full control over entity types via GliNER prompts
- Custom label index built from your exact 200 geographies
- GPU-accelerated NER
- Explicit disambiguation with tunable context window

**Cons:**
- Requires two models (GliNER + SentenceTransformer for disambiguation)
- Label index must be maintained

### Approach B: spaCy Entity Linker

**Location:** `knowledge_graph/classifier/geography.py`

**Classes:**
- `SpacyGeographyBackend` - Singleton managing spaCy pipeline
- `SpacyGeographyClassifier` - Classifier facade for individual concepts
- `SpacyGeographyBatchPredictor` - Bulk prediction

**How it works:**
1. **NER Stage:** spaCy's built-in NER identifies entities
2. **Entity Linking:** `spacy-entity-linker` directly links entities to Wikidata
3. **Filtering:** Keep only entities matching our 200 geography QIDs

**Pros:**
- Direct Wikidata linking without custom label index
- Mature spaCy ecosystem
- Entity linker handles disambiguation internally
- Single pipeline

**Cons:**
- Less control over which entity types are recognized
- May not recognize all 200 geographies (depends on spaCy NER training)
- CPU-based entity linker (no GPU acceleration)
- Requires downloading Wikidata knowledge base (~1GB)

## Usage

### GliNER Approach

```python
from knowledge_graph.classifier.geography import (
    initialize_geography_classifiers,
    GeographyClassifier,
    GeographyBatchPredictor,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID

# Define your geography concepts
brazil_concept = Concept(
    wikibase_id=WikibaseID("Q155"),
    preferred_label="Brazil",
    alternative_labels=["Brasil", "Federative Republic of Brazil"],
    description="Country in South America",
)

# Initialize the shared backend with ALL geography concepts
initialize_geography_classifiers([brazil_concept, ...])

# Create individual classifiers
brazil_classifier = GeographyClassifier(brazil_concept)
spans = brazil_classifier.predict("Brazilian climate policies were announced.")

# Or use batch predictor for all geographies at once (more efficient)
predictor = GeographyBatchPredictor()
predictor.initialize(all_geography_concepts)
results = predictor.predict_all(texts)  # Returns dict[WikibaseID, list[list[Span]]]
```

### spaCy Approach

```python
from knowledge_graph.classifier.geography import (
    initialize_spacy_geography_classifiers,
    SpacyGeographyClassifier,
    SpacyGeographyBatchPredictor,
)

# Initialize the shared backend
initialize_spacy_geography_classifiers(all_geography_concepts)

# Create individual classifiers
brazil_classifier = SpacyGeographyClassifier(brazil_concept)
spans = brazil_classifier.predict("Brazilian climate policies were announced.")

# Or use batch predictor
predictor = SpacyGeographyBatchPredictor()
predictor.initialize(all_geography_concepts)
results = predictor.predict_all(texts)
```

## Installation

Both approaches require the `transformers` optional dependency group:

```bash
# CPU version
uv sync --extra transformers

# GPU version (CUDA)
uv sync --extra transformers_gpu
```

For spaCy entity linker, you also need to download the spaCy model and knowledge base:

```bash
uv run python -m spacy download en_core_web_md
uv run python -m spacy_entity_linker "download_knowledge_base"
```

## Evaluation & Comparison

### Generating Evaluation Dataset

Use the provided script to create a labelled evaluation dataset:

```bash
# Generate with built-in sample texts
uv run python scripts/generate_geography_eval_set.py generate --use-samples

# Generate with custom input file (one text per line)
uv run python scripts/generate_geography_eval_set.py generate --input data/sample_texts.txt

# Validate an existing dataset
uv run python scripts/generate_geography_eval_set.py validate data/geography_eval_set.json
```

The script uses Claude (via pydantic-ai) to annotate texts with geography mentions, including:
- Span boundaries (start/end indices)
- Wikidata IDs
- Confidence levels

### Running the Comparison

```python
import time
from knowledge_graph.classifier.geography import (
    GeographyBatchPredictor,
    SpacyGeographyBatchPredictor,
)

# Load your evaluation texts
texts = [...]  # Your test texts
concepts = [...]  # Your geography concepts

# GliNER approach
gliner_predictor = GeographyBatchPredictor()
gliner_predictor.initialize(concepts)

start = time.time()
gliner_results = gliner_predictor.predict_all(texts, threshold=0.5)
gliner_time = time.time() - start

# spaCy approach
spacy_predictor = SpacyGeographyBatchPredictor()
spacy_predictor.initialize(concepts)

start = time.time()
spacy_results = spacy_predictor.predict_all(texts, threshold=0.0)
spacy_time = time.time() - start

print(f"GliNER: {len(texts)/gliner_time:.1f} texts/sec")
print(f"spaCy:  {len(texts)/spacy_time:.1f} texts/sec")
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | % of predicted spans that are correct |
| **Recall** | % of actual geography mentions found |
| **Linking accuracy** | % of found spans linked to correct QID |
| **Speed** | Texts processed per second |
| **Memory** | Peak GPU/CPU memory usage |

### Decision Criteria

| Criteria | GliNER wins if... | spaCy wins if... |
|----------|-------------------|------------------|
| Accuracy | F1 within 5% of spaCy | F1 > 5% better |
| Speed | >2x faster | Comparable speed |
| Control | Need custom label handling | Default Wikidata works |
| Maintenance | Want minimal dependencies | Trust spaCy ecosystem |

## Known Ambiguous Geographies

These labels require disambiguation:

| Label | Candidates | Disambiguation Signal |
|-------|-----------|----------------------|
| Georgia | Q230 (country), Q1428 (US state) | Tbilisi/Caucasus vs Atlanta |
| Guinea | Q1006 (Guinea), Q790 (Haiti area) | Conakry vs Port-au-Prince |
| Congo | Q971 (DRC), Q1025 (Republic) | Kinshasa vs Brazzaville |
| Niger | Q1032 (country), Q3542 (river) | Niamey vs waterway context |

## Configuration

### GliNER Backend

```python
GeographyClassifierBackend.get_instance(
    gliner_model="urchade/gliner_multi-v2.1",  # HuggingFace model ID
    embedding_model="all-MiniLM-L6-v2",        # For disambiguation
    device="cuda",                              # or "cpu", "mps"
)
```

### spaCy Backend

```python
SpacyGeographyBackend.get_instance(
    spacy_model="en_core_web_md",  # or "en_core_web_lg" for better accuracy
)
```

### Classifier Parameters

```python
GeographyClassifier(
    concept=brazil_concept,
    prediction_threshold=0.5,  # NER confidence threshold
    context_window=100,        # Characters of context for disambiguation
)

SpacyGeographyClassifier(
    concept=brazil_concept,
    prediction_threshold=0.0,  # Entity linker confidence threshold
)
```

## Running Tests

```bash
# Run unit tests (no GPU/models required)
uv run pytest tests/test_geography.py -v -k "not transformers"

# Run all tests including model-dependent tests
uv run --extra transformers pytest tests/test_geography.py -v

# Run only GliNER tests
uv run --extra transformers pytest tests/test_geography.py -v -k "Geography and not Spacy"

# Run only spaCy tests
uv run --extra transformers pytest tests/test_geography.py -v -k "Spacy"
```

## File Structure

```
knowledge_graph/classifier/
├── geography.py              # Main implementation (both approaches)
├── README_geography.md       # This file
└── __init__.py              # Exports

scripts/
└── generate_geography_eval_set.py  # Evaluation dataset generation

tests/
└── test_geography.py         # Unit and integration tests
```

## Implementation Notes

### Singleton Pattern

Both backends use a singleton pattern to share the loaded models across all classifier instances. This is critical for efficiency when you have 200 geography classifiers - each shares the same NER model.

```python
# Reset singleton for testing
GeographyClassifierBackend.reset_instance()
SpacyGeographyBackend.reset_instance()
```

### Lazy Loading

Models are loaded lazily on first use, not at import time. This keeps import fast and allows you to configure the backend before loading:

```python
backend = GeographyClassifierBackend.get_instance(device="cuda")
# Model not loaded yet
backend.initialize_index(concepts)
# Model loads here on first predict_all() call
```

### Batch Prediction

For maximum efficiency when classifying all 200 geographies:

```python
# EFFICIENT: Single NER pass, returns all geographies
predictor = GeographyBatchPredictor()
predictor.initialize(concepts)
all_results = predictor.predict_all(texts)

# INEFFICIENT: 200 separate NER passes
for concept in concepts:
    classifier = GeographyClassifier(concept)
    results = classifier.predict(texts)  # Each call runs NER again
```

### Unknown Entity Handling

When the NER model extracts a geography not in our 200:
- **Action:** Ignore for classification, but log for review
- **Purpose:** Identify potential taxonomy expansion candidates

Check logs for messages like:
```
INFO: Found 5 mentions of 3 unknown geography entities: ['narnia', 'westeros', 'mordor']...
```

## Troubleshooting

### "Backend not initialized" error

Call `initialize_geography_classifiers()` or `predictor.initialize()` before prediction:

```python
initialize_geography_classifiers(concepts)  # Do this first!
classifier = GeographyClassifier(brazil_concept)
classifier.predict(text)  # Now this works
```

### spaCy entity linker not working

Make sure you downloaded the knowledge base:

```bash
uv run python -m spacy_entity_linker "download_knowledge_base"
```

### CUDA out of memory

Reduce batch size or use CPU:

```python
backend = GeographyClassifierBackend.get_instance(device="cpu")
```

### Poor disambiguation accuracy

Try increasing the context window:

```python
classifier = GeographyClassifier(concept, context_window=200)
```

Or use a larger embedding model:

```python
backend = GeographyClassifierBackend.get_instance(
    embedding_model="all-mpnet-base-v2"
)
```
