# GraphGen: Knowledge Graph Generation Pipeline

GraphGen transforms raw documents into a structured knowledge graph with optional analytics and Neo4j upload.

## Key Architectural Features

- **Dependency Injection**: The `KnowledgePipeline` orchestrator accepts uploader and extractor dependencies via its constructor.
- **Config-driven**: Loads `config.yaml` + `.env` via `pydantic-settings` for consistent overrides.
- **Modular Pipeline**: Each major step lives under `graphgen/pipeline/` for testable, isolated logic.
- **Iterative Mode**: Optional batch-based experimentation for incremental graph evolution.

## Module Structure

```
graphgen/
├── main.py                # Entrypoint
├── orchestrator.py        # KnowledgePipeline (core orchestrator)
├── analytics/             # Reports, metrics, visualizations
├── config/                # Settings, schema, loaders
├── pipeline/
│   ├── lexical_graph_building/  # Segment/chunk graph construction
│   ├── entity_relation/         # Entity/relationship extraction
│   ├── embeddings/              # KGE/RAG embeddings
│   ├── graph_cleaning/          # Pruning + resolution
│   ├── community/               # Leiden detection + subcommunities
│   ├── summarization/           # LLM summaries
│   ├── analysis/                # Topic separation statistics
│   └── visualization/           # Plots and reports
├── utils/
│   ├── graphdb/                 # Neo4j adapter
│   └── parsers/                 # Input parsers
└── tools/                   # CLI utilities
```

## Running the Pipeline

### 1. Configuration
Set environment variables in `.env` at the repository root (examples):

```env
NEO4J_HOST=neo4j
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=sk-...
INPUT_DIR=/app/input
OUTPUT_DIR=/app/output
```

Adjust pipeline behavior in `config.yaml` as needed (chunk sizes, thresholds, feature flags).

### 2. Execution

```bash
python3 -m graphgen.main
```

## Pipeline Stages

Standard pipeline (`graphgen.main` -> `KnowledgePipeline`):

1. **Lexical Graph Construction**: Build a heterogeneous document/segment/chunk scaffold.
2. **Entity Extraction**: Run NER-guided constrained entity-relation extraction on chunks.
3. **Semantic Enrichment**: Generate embeddings and attempt duplicate resolution.
4. **Community Detection**: Cluster the entity-relation subgraph into topics/subtopics.
5. **Summarization**: Generate LLM summaries for communities.
6. **Topic Analysis** (optional): Statistical separation tests on embeddings.
7. **Pruning**: Cleanup and simplify low-value nodes/edges.
8. **Upload**: Persist to Neo4j as the final runtime step.
9. **Artifacts**: Save GraphML and analysis outputs.

Iterative experimental pipeline (`IterativeOrchestrator`):
- Repeats extraction over cumulative batches.
- Uses real sentence-transformer embeddings for entity resolution.
- Can optionally apply Node2Vec-based edge weighting before Leiden.
- Uploads only the final cumulative graph.

## Operational Notes

- Logging defaults to INFO and stdout. Enable debug logs by setting `debug: true` in `config.yaml`.
- See `/app/README.md` for full logging and troubleshooting guidance.