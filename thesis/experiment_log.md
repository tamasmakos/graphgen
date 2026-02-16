# Experiment Log & Decision Record

This document tracks key architectural decisions, pivots, and experimental hypotheses in the graph-based topic modeling project.

## 2026-02-16: Pivot from KGE-weighted Leiden to Node2Vec Sanity Check

### Decision
We are temporarily suspending the "KGE-weighted Leiden" approach and adopting a "Node2Vec sanity check".

### Rationale ("Why")
Analysis of `output/iterative_experiment_results.csv` reveals a concerning trend:
-   **KGE-weighted Modularity** (approx 0.67-0.69 in iterations 8-10) is consistently **lower** than the **Baseline Modularity** (approx 0.73-0.75 in iterations 8-9, 0.69 in it 10).
-   This suggests the current Knowledge Graph Embeddings (likely TransE/RotatE based) are introducing noise rather than reinforcing the community structure, or that the weighting scheme is distorting valid structural communities.
-   We need to determine if *any* embedding-based weighting can help, or if the graph's native topology is the strongest signal.

### Methodology ("What")
1.  **Drop KGE-weighted Leiden**: Suspend the use of semantic KGEs for community detection weighting.
2.  **Adopt Node2Vec Sanity Check**: Generate Node2Vec embeddings, which purely capture structural equivalence and neighborhood connectivity.
3.  **Test Impact**: Use Node2Vec similarity to weight the Leiden algorithm.

### Hypothesis & Interpretation
We will compare Node2Vec-weighted modularity against the previous KGE-weighted results and the unweighted baseline.

-   **Hypothesis 1: Node2Vec > KGE**
    -   *Observation*: Modularity increases significantly compared to the KGE runs.
    -   *Implication*: The KGE model was "wrong" (mismatched to the task or poor quality). Structural embeddings (Node2Vec) are better suited for this graph topology.
-   **Hypothesis 2: Node2Vec ≈ Baseline**
    -   *Observation*: Modularity is flat or very similar to the unweighted baseline.
    -   *Implication*: The graph is already strongly clustered. "Ground truth" communities are structural. Embeddings (whether semantic or structural) provide little marginal gain over the raw topology.

### Context
-   **Data Source**: `output/iterative_experiment_results.csv` (Iterations 7-10 show KGE modularity lagging baseline).
-   **Next Steps**: Execute Node2Vec pipeline and analyze modularity delta.

## 2026-02-16: Node2Vec Sanity Check Results

### Experiment Overview
-   **Experiment**: Node2Vec Sanity Check
-   **Date**: 2026-02-16
-   **Method**: Trained Node2Vec on Iteration 10 graph. Used cosine similarity of embeddings to weight existing edges. Ran Leiden algorithm for community detection.

### Results
-   **Baseline (Unweighted) Modularity**: 0.7184
-   **Node2Vec Weighted Modularity**: 0.7735
-   **Delta**: +0.0551

### Conclusion & Decision
The experiment yielded a significant INCREASE in modularity (+0.0551). This confirms **Hypothesis 1**: the previously observed drop in modularity with KGEs was due to the specific embedding model (likely DistMult) being ill-suited or noisy for community detection on this topology, rather than an inherent limitation of embedding-weighted clustering.

Node2Vec embeddings successfully captured and reinforced the underlying structural communities. However, given the complexity trade-off, the decision to proceed with the **Simplified Pipeline (Unweighted Leiden)** stands for the core thesis argument, with Node2Vec reserved as a potential enhancement for future work if structural enrichment becomes critical. The semantic KGE weighting approach is formally deprecated.
