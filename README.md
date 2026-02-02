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

# Extraction Settings
extraction:
  file_pattern: "*.txt" # IMPORTANT: Match your input files
  backend: "spacy"

# Schema Definition (Dynamic)
schema:
  nodes:
    Doc:
      label: "Document"
      source_type: "document"
      attributes: ["filename"]
    Chunk:
      label: "TextChunk"
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
- **`settings.py`**: Pydantic models (InfrastructureSettings, LLMSettings) for validated configuration.
- **`loader.py`**: Logic to load and merge YAML configuration.
- **`schema.py`**: Pydantic models (`GraphSchema`, `NodeSchema`) for defining the target graph structure dynamically.

### Pipeline Components (`graphgen.pipeline`)
- **`lexical_graph_building.builder`**: Scans input files and constructs the initial graph structure based on the Schema.
- **`entity_relation.extraction`**: Orchestrates entity extraction from text chunks using configured backends (Spacy, GLiNER, LLM).
- **`entity_relation.extractors`**: Contains specific extractor implementations.
- **`graph_cleaning.resolution`**: Handles entity resolution and merging.

### Utilities (`graphgen.utils`)
- **`graphdb.uploader`**: Handles bulk uploading of the NetworkX graph to FalkorDB (RedisGraph) and hybrid vector storage in Postgres.
- **`graphdb.neo4j_adapter`**: **[NEW]** Adapter for uploading graphs to Neo4j, including vector index management.
- **`parsers.custom`**: `RegexParser` for flexible text segmentation based on configurations.
- **`parsers.life`**: Support for parsing LifeLog CSVs.

## 🛠️ Developer Notes

### Adding a New Parser
Inherit from `graphgen.utils.parsers.base.BaseDocumentParser` and implement the `parse` method. Then register it or use it in the pipeline configuration.

### Customizing the Pipeline
The `KnowledgePipeline` class in `graphgen.orchestrator.py` defines the sequence of steps. You can subclass it or modify the `run` method to inject new steps (e.g. specialized topic modeling).
