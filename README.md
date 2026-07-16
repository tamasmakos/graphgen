# From Discourse to Structure

**A Knowledge Graph-Based Approach to Topic Modeling in Political Debate**

This is the research repository for a master's thesis investigating whether political
topics can be recovered by building an LLM-extracted knowledge graph from parliamentary
speeches and detecting communities within it — an explicitly *structural* alternative to
probabilistic topic models such as LDA.

The method is applied to the European Parliament's *This is Europe* debate series
(13 EU leaders, 2022–2024) and validated against the independent expert analysis of the
European Parliamentary Research Service (EPRS).

## Findings (full-corpus run)

- **Scale-free graph** (7,975 nodes; Clauset MLE α ≈ 2.38, KS = 0.023) — the extraction
  preserves genuine discourse structure.
- **node2vec edge reweighting** lifts Leiden modularity **0.702 → 0.812** (+15.7%),
  confirmed by a permutation test (*p* < 0.001) and a 20-seed stability analysis (*d* = 58.9).
- **EPRS alignment**: all six recurring expert themes recovered (100% coverage).
- **LDA baseline** on the same corpus is beaten on coverage, alignment, and stability.

## Pipeline

Text → knowledge graph → topics, in seven stages (`graphgen/`, orchestrated by `orchestrator.py`):

1. **Chunking** — speeches split into 512-char chunks (64 overlap) → ~2,085 chunks.
2. **Ontology matching** — each chunk is embedded and matched to the best-fitting ontology
   (EU CDM family + related vocabularies), which supplies its permitted entity types.
3. **Extraction** — a typed DSPy signature prompts an LLM (Llama-3.3-70B via OpenRouter) to emit
   Pydantic-validated `(source, relation, target)` triplets; entity *types* are ontology-constrained,
   relations emerge freely.
4. **Resolution & pruning** — entities merged in three passes (acronym, containment, semantic
   cosine > 0.95; 5,321 → 4,841); low-confidence edges and isolated nodes dropped.
5. **Community detection** — Leiden (resolution 0.5, seed 42) on the graph, with **node2vec**
   structural embeddings reweighting edges beforehand; applied hierarchically → 99 topics, 683 subtopics.
6. **Summarisation** — each community summarised bottom-up by the LLM into a titled, evidence-grounded report.
7. **Analytics** — scale-free fit, modularity uplift + significance tests, silhouette, centrality,
   and EPRS ground-truth alignment (all written to `output/thesis_outputs/`).

## Layout

| Path | Contents |
|------|----------|
| `thesis/` | Thesis manuscript (`main_final.tex` / `.pdf`), figures, EPRS ground truth |
| `graphgen/` | The pipeline: extraction → resolution → community detection → summarisation → analytics |
| `input/` | Corpus (translated speeches) and ontologies |
| `output/thesis_outputs/` | Canonical run artifacts — graph, plots, community/topic CSVs, ground-truth alignment |

## Reproducing the pipeline

```bash
pip install -e .
graphgen            # runs the pipeline per config.yaml
```

Configuration is split between `config.yaml` (pipeline logic, schema, defaults) and
`.env` (API keys, infrastructure). Extraction uses an LLM via OpenRouter; see `config.yaml`
for models and parameters.
