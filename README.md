# GraphGen: Knowledge Graph Generation Package

This python package provides a highly flexible pipeline for generating Knowledge Graphs from raw text data. It is designed to be modular, decoupling the lexical graph construction from semantic entity extraction.

## 📦 Installation

This package uses `pyproject.toml` configuration.

```bash
pip install -e .
```

## 🚀 Usage

You can run the pipeline directly via the command line interface:

```bash
graphgen
```

### Configuration

GraphGen uses a dual configuration system:
1.  **`config.yaml`**: For Application Logic, Schema Definition, and Defaults.
2.  **`.env`**: For Secrets (API Keys) and Infrastructure Overrides (Hosts, Ports).

#### config.yaml Example
```yaml
# Infrastructure
infra:
  graph_db_type: "neo4j" # options: "falkordb", "neo4j"
  neo4j_host: "neo4j"
  neo4j_port: 7687
  clean_start: true # options: true, false (wipe DB before run)

# Extraction Settings
extraction:
  file_pattern: "*.txt" # IMPORTANT: Match your input files
  backend: "spacy"  # options: "gliner", "spacy", "llm"
  gliner_model: "urchade/gliner_medium-v2.1"  # GLiNER model for entity extraction
  device: "cuda"  # options: "auto", "cuda", "cpu"
  use_onnx: false # Set to true to leverage onnxruntime-gpu
  gliner_labels:  # Entity types when using GLiNER backend
    - "Person"
    - "Organization"
    - "Location"
    - "Event"

# Schema Definition (Dynamic)
schema:
  nodes:
    Doc:
      label: "Document"
      source_type: "document"
      attributes: ["filename"]
    Chunk:
      label: "Chunk"
      source_type: "chunk"
  edges:
    - source_label: "Doc"
      target_label: "Chunk"
      relation_type: "HAS_CHUNK"
      is_hierarchical: true
```

## 🏗️ Architecture & Modules

The package is organized into the following core components:

### Core (`graphgen.types`)
- **`PipelineContext`**: The central "bus" object that holds the state (NetworkX graph, stats, errors) and is passed between all pipeline steps.
- **`ChunkExtractionTask`**: A Pydantic model representing a single unit of text to be processed for entity extraction.
- **`SegmentData`**: Data model representing a document segment.

### Configuration (`graphgen.config`)
- **`settings.py`**: Pydantic models (InfrastructureSettings, LLMSettings, **KGESettings**, **AnalysisSettings**, **TestModeSettings**) for validated configuration.
- **`loader.py`**: Logic to load and merge YAML configuration.
- **`schema.py`**: Pydantic models (`GraphSchema`, `NodeSchema`) for defining the target graph structure dynamically.

### Pipeline Components (`graphgen.pipeline`)
- **`lexical_graph_building.builder`**: Scans input files and constructs the initial graph structure based on the Schema.
- **`entity_relation.extraction`**: Orchestrates entity extraction from text chunks using configured backends (Spacy, GLiNER, LLM).
- **`entity_relation.extractors`**: Contains specific extractor implementations.
- **`graph_cleaning.resolution`**: Handles entity resolution and merging.
- **`embeddings.kge`**: **[NEW]** PyKeen Knowledge Graph Embedding training module. Trains KGE models (e.g., DistMult) on entity-relation triples and computes edge weights for improved community detection.
- **`summarization.core`**: **[NEW]** Hierarchical summarization module. Generates titles and summaries for topics and subtopics using LLMs. Optimized for label robustness, handling variations like `Chunk` and `NamedEntity`.
- **`analysis.topic_separation`**: **[NEW]** Statistical analysis of topic/community embedding separation. Implements silhouette scores, MANOVA, and pairwise tests to verify semantic clustering.

### Utilities (`graphgen.utils`)
- **`graphdb.uploader`**: Handles bulk uploading of the NetworkX graph to FalkorDB (RedisGraph) and hybrid vector storage in Postgres.
- **`graphdb.neo4j_adapter`**: Adapter for uploading graphs to Neo4j, including vector index management.
- **`ontology_parser`**: `OntologyLabelExtractor` for parsing RDF/OWL ontologies and extracting class labels.
- **`labels`**: `resolve_gliner_labels()` for resolving GLiNER labels from manual config, ontology, or merged sources.
- **`parsers.custom`**: `RegexParser` for flexible text segmentation based on configurations.
- **`parsers.life`**: Support for parsing LifeLog CSVs.

## ⚙️ Advanced Configuration

### Test Mode

Enable test mode to limit the number of documents processed, useful for quick testing and development:

```yaml
test_mode:
  enabled: true  # Toggle test mode
  max_documents: 5  # Process only this many documents (0 = no limit)
```

When enabled with `max_documents > 0`, the pipeline will only process the specified number of documents from the input directory. Set `max_documents: 0` or `enabled: false` to process all documents normally.

### Entity Extraction & Ontology Integration

GraphGen supports a robust entity extraction pipeline that combines rapid NER models (GLiNER/Spacy) with high-precision LLM extraction, all governed by your domain ontology.

#### Unified Label Configuration
Manual labels and ontology-derived classes are consolidated into a single master set of `entity_labels`.

```yaml
extraction:
  backend: "gliner" # Recommended for large volumes
  entity_labels: ["Person", "Organization"] # Base types
  
  ontology:
    enabled: true
    ontology_dir: "/app/input/ontology/cdm-4.13.2"
    top_level_only: true # Extract only high-level categories
    min_subclasses: 1    # Filter for broad organizational nodes
```

#### 🛡️ NER Gatekeeper Logic (Precision Mode)
The pipeline uses a "gatekeeper" architecture to ensure high precision and strictly valid graphs:
1.  **Discovery**: GLiNER/Spacy scans a text chunk for all classes defined in the ontology.
2.  **Schema Filtering**: The extraction schema for that specific chunk is dynamically restricted to *only* the classes discovered in step 1.
3.  **Strict LLM Extraction**: The LLM acts as the final extractor, operating in `strict_mode=True`. It identifies relationships and instances but is strictly constrained to the ontology types verified by the NER gatekeeper.
4.  **Classification**: Entities are automatically tagged with their validated `ontology_class`, enabling seamless integration with existing RDF/OWL knowledge bases.

### Knowledge Graph Embeddings (PyKeen)

Enable KGE training to use embedding-based edge weights for community detection:

```yaml
kge:
  enabled: true  # Toggle KGE training (computationally expensive)
  model: "DistMult"  # PyKeen model type
  embedding_dim: 64
  learning_rate: 0.01
  num_epochs: 50
  early_stopping_patience: 5
```

When enabled, the pipeline:
1. Trains a KGE model on all entity-relation triples
2. Computes cosine similarity between connected entities
3. Uses similarities as edge weights for Leiden community detection

### Community Detection (Optimized)

The pipeline uses the Leiden algorithm for community detection, optimized for consistency and quality:

```yaml
community:
  resolutions: [0.5, 0.75, 1.0, 1.25, 1.5] # Multiple resolutions to try
  n_iterations: 10 # Runs per resolution to find stable partition
  min_community_size: 3 # Merges tiny communities into neighbors
  seed: 42 # For reproducible partitions
```

## Tasks
- [x] Research and Planning
- [x] Create verification script
- [x] Update `Dockerfile` to CUDA-enabled base image
- [x] Update `docker-compose.yaml` with GPU reservations
- [x] Verify GPU availability in container
- [x] Update README.md

When running, the detector:
1. Performs a grid search over the specified `resolutions`.
2. For each resolution, runs the Leiden algorithm `n_iterations` times.
3. Selects the partition with the highest modularity.
4. Optionally merges small communities based on `min_community_size`.
5. Recalculates modularity after any post-processing.

### Statistical Analysis

The pipeline can run statistical tests to verify that communities form distinct semantic clusters:

```yaml
analysis:
  topic_separation_test: true  # Enable statistical tests
  output_file: "topic_separation_stats.json"
  hierarchy_levels:
    - "COMMUNITY"
    - "SUBCOMMUNITY"
```

Output includes:
- **Silhouette Score**: Measures cluster quality (-1 to 1, higher is better)
- **MANOVA Approximation**: Tests if community centroids differ significantly
- **Pairwise Comparisons**: Bonferroni-corrected t-tests between community pairs
- **PCA Variance**: Explained variance ratios for embedding space

## 🛠️ Developer Notes

### Adding a New Parser
Inherit from `graphgen.utils.parsers.base.BaseDocumentParser` and implement the `parse` method. Then register it or use it in the pipeline configuration.

### Customizing the Pipeline
The `KnowledgePipeline` class in `graphgen.orchestrator.py` defines the sequence of steps. You can subclass it or modify the `run` method to inject new steps (e.g. specialized topic modeling).

### Pipeline Steps (in order)
1. **Lexical Graph Building**: Document parsing and chunking
2. **Entity Extraction**: NLP-based entity and relation extraction
3. **Semantic Enrichment**: RAG embeddings and entity resolution
4. **KGE Training** (optional): PyKeen embedding generation
5. **Community Detection**: Leiden algorithm with optional KGE weights
6. **Topic Analysis** (optional): Statistical separation tests
7. **Pruning**: Graph cleanup
8. **Upload**: Database persistence
9. **Artifacts**: GraphML and report generation

---

## 🚀 GPU Setup & Troubleshooting

If you are using the GLiNER backend and want to leverage GPU acceleration, follow these steps:

### 1. Prerequisites
- Ensure **NVIDIA Drivers** are installed on your host machine.
- Ensure the **NVIDIA Container Toolkit** is installed and configured for Docker.

### 2. Update Environment
After updating the `Dockerfile` and `docker-compose.yaml`, you must rebuild your containers:

```bash
docker compose down
docker compose up -d --build
```

### 3. Verify GPU Availability
Run the included verification script inside the container:

```bash
docker compose exec dev python3 tests/verify_gpu.py
```

This script checks:
1. If PyTorch can detect the GPU.
2. If GLiNER can successfully load onto the CUDA device.

If you see "ERROR: CUDA is not available", double-check your host's NVIDIA driver installation and ensure Docker is allowed to access the GPU.
