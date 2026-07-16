"""Metrics for graph quality and embedding analysis."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_ENTITY_NODE_TYPES = {"ENTITY_CONCEPT", "ENTITY", "PLACE", "NAMEDENTITY"}


def _log_log_r_squared(degrees_arr: np.ndarray, xmin: int) -> float:
    """Log-log binned-PDF regression R² (kept for the visualisation only).

    NOTE: This R² is *not* a valid goodness-of-fit statistic for a power
    law (Clauset, Shalizi & Newman, 2009); it is retained purely so the
    right-hand panel of the scale-free figure can show the OLS reference
    line.  The formal fit decision is made by :func:`test_scale_free` via
    the MLE + likelihood-ratio-test machinery of the ``powerlaw`` package.
    """
    max_deg = int(degrees_arr.max())
    bins = np.unique(
        np.round(np.logspace(0, np.log10(max(max_deg, 2)), 20)).astype(int)
    )
    counts, edges = np.histogram(degrees_arr, bins=bins)
    widths = np.diff(edges)
    bin_centres = (edges[:-1] + edges[1:]) / 2.0
    pdf = counts / (float(len(degrees_arr)) * widths)

    mask = (pdf > 0) & (bin_centres >= xmin)
    if mask.sum() < 3:
        return 0.0
    log_k = np.log10(bin_centres[mask])
    log_p = np.log10(pdf[mask])
    slope, intercept = np.polyfit(log_k, log_p, 1)
    log_p_pred = intercept + slope * log_k
    ss_res = np.sum((log_p - log_p_pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def test_scale_free(graph: nx.Graph) -> Dict[str, Any]:
    """Test whether *graph* has a scale-free (power-law) degree distribution.

    Uses the statistically principled method of Clauset, Shalizi & Newman
    (2009) as implemented in the ``powerlaw`` package:

    1. The power-law exponent ``alpha`` and lower bound ``xmin`` are
       estimated by maximum likelihood (not log-log regression, which is
       biased).
    2. Goodness-of-fit is summarised by the Kolmogorov-Smirnov distance
       ``ks_distance`` between the empirical and fitted CDFs above ``xmin``.
    3. The power law is compared against plausible alternative heavy-tailed
       distributions (lognormal, exponential, truncated power law) via
       normalised log-likelihood-ratio tests.  Each test yields a signed
       ratio ``R`` (positive ⇒ power law favoured) and a significance
       ``p``.

    A distribution is reported as scale-free only when the power law is
    *not* decisively beaten by an alternative — in particular it must not
    lose significantly to the exponential (the key "is the tail actually
    heavy?" comparison).

    If the ``powerlaw`` package is unavailable the function falls back to
    the legacy Hill-estimator + log-log-R² heuristic.

    Returns:
        Dict with keys ``degree_sequence``, ``alpha``, ``xmin``,
        ``sigma``, ``ks_distance``, ``log_log_r_squared``, ``ks_p_value``,
        ``distribution_comparison``, ``is_scale_free``, ``method``,
        ``n_nodes``.  A too-small graph yields a ``skipped`` key instead.
    """
    g_undirected = graph.to_undirected() if graph.is_directed() else graph
    degrees = [d for _, d in g_undirected.degree() if d > 0]
    n_nodes = graph.number_of_nodes()

    if len(degrees) < 10:
        logger.warning(
            "test_scale_free: too few non-zero-degree nodes (%d); skipping.", len(degrees)
        )
        return {
            "degree_sequence": degrees,
            "n_nodes": n_nodes,
            "skipped": f"Insufficient non-zero-degree nodes (found {len(degrees)}, need >= 10)",
        }

    degrees_arr = np.array(degrees, dtype=float)

    try:
        import warnings

        import powerlaw  # type: ignore[import-not-found]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = powerlaw.Fit(degrees_arr, discrete=True, verbose=False)

            # powerlaw returns NaN when it cannot find a valid fit (e.g. a
            # near-constant degree sequence); fall back to the legacy method.
            if fit.xmin is None or not np.isfinite(fit.xmin) or not np.isfinite(fit.alpha):
                logger.warning(
                    "test_scale_free: powerlaw fit is degenerate (xmin=%s, alpha=%s); "
                    "falling back to legacy heuristic.",
                    fit.xmin,
                    fit.alpha,
                )
                return _test_scale_free_legacy(degrees, degrees_arr, n_nodes)

            alpha = float(fit.alpha)
            xmin = int(fit.xmin)
            sigma = float(fit.sigma)
            ks_distance = float(fit.power_law.D)

            # Likelihood-ratio tests vs alternative heavy-tailed laws.
            comparison: Dict[str, Dict[str, float]] = {}
            for alt in ("lognormal", "exponential", "truncated_power_law"):
                try:
                    R, p = fit.distribution_compare(
                        "power_law", alt, normalized_ratio=True
                    )
                    comparison[alt] = {
                        "loglik_ratio_R": float(R),
                        "p_value": float(p),
                        "power_law_favored": bool(R > 0),
                    }
                except Exception:
                    logger.debug("LRT power_law vs %s failed.", alt, exc_info=True)

        r_squared = _log_log_r_squared(degrees_arr, xmin)

        # Decision: the power law must not lose *significantly* to the
        # exponential (heavy-tail test) or to the lognormal.
        def _loses_to(alt: str) -> bool:
            c = comparison.get(alt)
            return bool(c and not c["power_law_favored"] and c["p_value"] < 0.05)

        is_scale_free = not (_loses_to("exponential") or _loses_to("lognormal"))

        return {
            "degree_sequence": [int(d) for d in degrees],
            "alpha": alpha,
            "xmin": xmin,
            "sigma": sigma,
            "ks_distance": ks_distance,
            "log_log_r_squared": r_squared,
            "ks_p_value": None,  # formal GOF p is reported via the LRTs below
            "distribution_comparison": comparison,
            "is_scale_free": bool(is_scale_free),
            "method": "clauset_mle_powerlaw",
            "n_nodes": int(n_nodes),
        }

    except ImportError:
        logger.warning(
            "test_scale_free: 'powerlaw' package unavailable; falling back to "
            "the legacy Hill-estimator + log-log-R² heuristic."
        )
        return _test_scale_free_legacy(degrees, degrees_arr, n_nodes)


def _test_scale_free_legacy(
    degrees: List[int], degrees_arr: np.ndarray, n_nodes: int
) -> Dict[str, Any]:
    """Legacy Hill-estimator + log-log-R² scale-free heuristic (fallback)."""
    from scipy.stats import ks_1samp

    xmin = max(1, int(np.percentile(degrees_arr, 10)))
    tail = degrees_arr[degrees_arr >= xmin]
    if tail.size < 5:
        xmin = 1
        tail = degrees_arr

    log_ratios = np.log(tail / xmin)
    alpha = float(1.0 + tail.size / np.sum(log_ratios)) if np.sum(log_ratios) > 0 else 2.0
    r_squared = _log_log_r_squared(degrees_arr, xmin)

    def _power_law_cdf(k: np.ndarray) -> np.ndarray:
        return 1.0 - (k / xmin) ** (-(alpha - 1.0))

    ks_p_value: Optional[float] = None
    try:
        _, ks_p = ks_1samp(np.sort(tail), _power_law_cdf)
        ks_p_value = float(ks_p)
    except Exception:
        logger.debug("test_scale_free: KS test failed.", exc_info=True)

    is_scale_free = bool(r_squared > 0.8 and (ks_p_value is None or ks_p_value > 0.05))

    return {
        "degree_sequence": [int(d) for d in degrees],
        "alpha": alpha,
        "xmin": int(xmin),
        "log_log_r_squared": r_squared,
        "ks_p_value": ks_p_value,
        "is_scale_free": is_scale_free,
        "method": "legacy_hill_loglog",
        "n_nodes": int(n_nodes),
    }


def run_node2vec_permutation_test(
    graph: nx.Graph,
    observed_modularity: float,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Permutation test for whether Node2Vec edge-weight structure matters.

    For each permutation the edge weights on entity–entity edges are
    randomly shuffled while the community partition (read from node
    attributes) is held fixed.  Leiden is *not* re-run; instead the
    modularity of the fixed partition under shuffled weights is computed
    directly, giving a null distribution of "random-weight" modularities.

    Args:
        graph: Entity-relation graph with Node2Vec ``weight`` attributes
               on edges and ``community_id`` attributes on entity nodes.
        observed_modularity: The modularity achieved with the true weights.
        n_permutations: Number of random weight shuffles.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys ``null_distribution``, ``p_value``,
        ``n_permutations``, ``null_95th``, ``null_99th``.
    """
    rng = np.random.default_rng(seed)

    # --- extract entity nodes that have a community assignment ---
    entity_nodes: set = set()
    partition: Dict[str, int] = {}
    for node, data in graph.nodes(data=True):
        if str(data.get("node_type", "")).upper() in _ENTITY_NODE_TYPES:
            comm_id = data.get("community_id")
            if comm_id is not None:
                partition[node] = int(comm_id)
                entity_nodes.add(node)

    if len(partition) < 2:
        logger.warning(
            "run_node2vec_permutation_test: fewer than 2 entity nodes with "
            "community assignments; returning empty result."
        )
        return {
            "null_distribution": [],
            "p_value": None,
            "n_permutations": n_permutations,
            "null_95th": None,
            "null_99th": None,
        }

    # --- collect entity-entity edges and their weights ---
    edge_endpoints: List[tuple] = []
    edge_weights: List[float] = []
    for u, v, data in graph.edges(data=True):
        if u in entity_nodes and v in entity_nodes:
            edge_endpoints.append((u, v))
            edge_weights.append(float(data.get("weight", 1.0)))

    if not edge_endpoints:
        logger.warning("run_node2vec_permutation_test: no entity-entity edges found.")
        return {
            "null_distribution": [],
            "p_value": None,
            "n_permutations": n_permutations,
            "null_95th": None,
            "null_99th": None,
        }

    # --- build community partition as list-of-sets for nx.modularity ---
    comm_sets: Dict[int, set] = defaultdict(set)
    for node, comm_id in partition.items():
        comm_sets[comm_id].add(node)
    communities_list = [s for s in comm_sets.values() if s]

    weights_arr = np.array(edge_weights)
    null_distribution: List[float] = []

    base_graph: nx.Graph = nx.Graph()
    base_graph.add_nodes_from(entity_nodes)

    for _ in range(n_permutations):
        shuffled = rng.permutation(weights_arr)
        temp_graph = base_graph.copy()
        for (u, v), w in zip(edge_endpoints, shuffled):
            temp_graph.add_edge(u, v, weight=float(w))
        try:
            q = nx.community.modularity(temp_graph, communities_list, weight="weight")
            null_distribution.append(float(q))
        except Exception:
            logger.debug("run_node2vec_permutation_test: modularity call failed.", exc_info=True)

    if not null_distribution:
        return {
            "null_distribution": [],
            "p_value": None,
            "n_permutations": n_permutations,
            "null_95th": None,
            "null_99th": None,
        }

    null_arr = np.array(null_distribution)
    n = len(null_distribution)
    # Add-one smoothing avoids reporting an impossible p = 0 when the
    # observed statistic exceeds every permutation (North et al., 2002).
    exceedances = int(np.sum(null_arr >= observed_modularity))
    p_value = (exceedances + 1) / (n + 1)
    p_value_display = f"< {1.0 / (n + 1):.4g}" if exceedances == 0 else f"{p_value:.4g}"

    return {
        "null_distribution": null_distribution,
        "p_value": float(p_value),
        "p_value_display": p_value_display,
        "n_permutations": n,
        "null_95th": float(np.percentile(null_arr, 95)),
        "null_99th": float(np.percentile(null_arr, 99)),
    }


def compare_modularity_distributions(
    baseline_mods: List[float],
    weighted_mods: List[float],
) -> Dict[str, Any]:
    """Test whether Node2Vec weighting shifts the modularity distribution.

    Given two samples of modularity values — one from repeated Leiden runs
    on the unweighted baseline graph, one from repeated runs on the
    Node2Vec-weighted graph — this reports descriptive statistics, a
    one-sided Mann-Whitney U test (H₁: weighted > baseline), and Cohen's
    *d* effect size.  Using the *distributions* rather than single point
    estimates lets the thesis state that the uplift exceeds seed-to-seed
    variance, not merely that one weighted run beat one baseline run.
    """
    from scipy.stats import mannwhitneyu

    result: Dict[str, Any] = {
        "n_runs_baseline": len(baseline_mods),
        "n_runs_weighted": len(weighted_mods),
    }
    if len(baseline_mods) < 2 or len(weighted_mods) < 2:
        result["skipped"] = "Need >= 2 runs per condition"
        return result

    base = np.array(baseline_mods, dtype=float)
    weig = np.array(weighted_mods, dtype=float)
    base_std = float(np.std(base, ddof=1))
    weig_std = float(np.std(weig, ddof=1))
    mean_delta = float(np.mean(weig) - np.mean(base))

    result.update(
        {
            "baseline_mean": float(np.mean(base)),
            "baseline_std": base_std,
            "weighted_mean": float(np.mean(weig)),
            "weighted_std": weig_std,
            "mean_delta": mean_delta,
        }
    )

    pooled_var = (np.var(base, ddof=1) + np.var(weig, ddof=1)) / 2.0
    _EPS = 1e-12

    if pooled_var <= _EPS:
        # Leiden converged to an identical partition for every seed in each
        # condition: the partitions are deterministic and the uplift is
        # exact, not an artefact of seed noise.  A variance-based effect
        # size / rank test is undefined (and would explode) here, so we
        # report the deterministic verdict instead.
        result["deterministic"] = True
        result["cohens_d"] = None
        result["mann_whitney_u"] = None
        result["p_value"] = None
        result["significant"] = bool(mean_delta > 0)
        result["note"] = (
            "Both conditions are deterministic across seeds (zero variance); "
            "the modularity uplift is exact and not attributable to seed noise."
        )
        logger.info(
            "compare_modularity_distributions: deterministic partitions "
            "(baseline=%.4f, weighted=%.4f, exact delta=%+.4f).",
            float(np.mean(base)), float(np.mean(weig)), mean_delta,
        )
        return result

    result["deterministic"] = False
    try:
        u_stat, p_val = mannwhitneyu(weig, base, alternative="greater")
        result["mann_whitney_u"] = float(u_stat)
        result["p_value"] = float(p_val)
        result["significant"] = bool(p_val < 0.05)
    except Exception:
        logger.debug("compare_modularity_distributions: Mann-Whitney failed.", exc_info=True)

    # Cohen's d with pooled standard deviation.
    result["cohens_d"] = float(mean_delta / np.sqrt(pooled_var))

    return result


def analyze_similarity_distribution(
    similarity_matrix: np.ndarray,
    labels: List[str],
) -> Dict[str, Any]:
    """Compute descriptive statistics and a Shapiro-Wilk normality test.

    Extracts all unique pairwise similarities from the upper triangle of
    *similarity_matrix* (diagonal excluded) and returns summary statistics
    alongside a Shapiro-Wilk normality test.

    Args:
        similarity_matrix: Square float array of pairwise cosine
                           similarities, as produced by
                           ``sklearn.metrics.pairwise.cosine_similarity``.
        labels: Sorted list of node identifiers corresponding to matrix
                rows/columns.  Used only to record ``n`` in the output.

    Returns:
        Dict with keys ``pairwise_values``, ``mean``, ``std``, ``min``,
        ``max``, ``n_pairs``, ``normality_test``.
    """
    from scipy.stats import shapiro

    n = len(labels)
    pairwise: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(float(similarity_matrix[i, j]))

    if not pairwise:
        return {
            "pairwise_values": [],
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "n_pairs": 0,
            "normality_test": {},
        }

    arr = np.array(pairwise)
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    normality_test: Dict[str, Any] = {}
    if len(pairwise) >= 3:
        try:
            # Shapiro-Wilk is limited to 5000 samples
            sample = arr[:5000]
            stat, p_value = shapiro(sample)
            normality_test = {
                "shapiro_wilk_statistic": float(stat),
                "shapiro_wilk_p_value": float(p_value),
                "is_normal": bool(p_value > 0.05),
                "n_tested": int(len(sample)),
            }
        except Exception:
            logger.debug("analyze_similarity_distribution: Shapiro-Wilk failed.", exc_info=True)

    return {
        "pairwise_values": pairwise,
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_pairs": int(len(pairwise)),
        "normality_test": normality_test,
    }

def calculate_modularity(graph: nx.Graph, communities: Dict[str, int]) -> float:
    """Calculate modularity of the graph given community assignments."""
    try:
        # Create community sets for nodes that exist in both graph and communities
        community_sets = defaultdict(set)
        common_nodes = set()
        
        for node, comm_id in communities.items():
            if node in graph:
                community_sets[comm_id].add(node)
                common_nodes.add(node)
        
        # Filter out empty communities
        valid_communities = [nodes for nodes in community_sets.values() if nodes]
        
        if not valid_communities:
            logger.info(
                "Skipping modularity: no valid communities (nodes=%d, communities=%d).",
                graph.number_of_nodes(),
                len(community_sets),
            )
            return 0.0
            
        # Create subgraph of only the nodes involved in communities
        # Modularity requires partition to cover all nodes in the graph
        subgraph = graph.subgraph(common_nodes)
        if len(subgraph) == 0 or subgraph.number_of_edges() == 0:
             return 0.0

        return nx.community.modularity(subgraph, valid_communities)
    except ZeroDivisionError:
        logger.warning("Modularity calculation failed due to ZeroDivisionError (likely no edges).")
        return 0.0
    except Exception:
        logger.exception("Failed to calculate modularity.")
        return 0.0

def calculate_topic_overlap(topic_embeddings: Dict[str, np.ndarray]) -> float:
    """Calculate average cosine similarity between all pairs of topic embeddings."""
    try:
        if len(topic_embeddings) < 2:
            logger.info(
                "Skipping topic overlap: need at least 2 embeddings (found=%d).",
                len(topic_embeddings),
            )
            return 0.0
        
        embeddings = list(topic_embeddings.values())
        if not embeddings:
            return 0.0
            
        # Stack embeddings
        matrix = np.vstack(embeddings)
        
        # Calculate pairwise cosine similarity
        sim_matrix = cosine_similarity(matrix)
        
        # Get upper triangle excluding diagonal
        upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
        
        if len(upper_tri) == 0:
            logger.info("Skipping topic overlap: no pairwise similarities computed.")
            return 0.0
            
        return float(np.mean(upper_tri))
    except Exception:
        logger.exception("Failed to calculate topic overlap.")
        return 0.0

def analyze_modularity_vs_overlap(
    results_history: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Correlate modularity with topic overlap across iterations.
    Expects list of dicts with keys 'modularity' and 'topic_overlap'.
    """
    if len(results_history) < 3:
        logger.info(
            "Skipping correlation analysis: need >= 3 data points (found=%d).",
            len(results_history),
        )
        return {}
        
    stats = {}
    try:
        mods = [r.get('modularity', 0) for r in results_history]
        overlaps = [r.get('topic_overlap', 0) for r in results_history]
        
        # Pearson correlation
        p_corr, p_val = pearsonr(mods, overlaps)
        stats['pearson_correlation'] = p_corr
        stats['pearson_p_value'] = p_val
        
        # Spearman correlation (robust to outliers)
        s_corr, s_val = spearmanr(mods, overlaps)
        stats['spearman_correlation'] = s_corr
        stats['spearman_p_value'] = s_val
        
    except Exception:
        logger.exception("Correlation analysis failed.")
        
    return stats

def calculate_node2vec_significance(
    results_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform a paired t-test to determine if Node2Vec significantly improved modularity.
    Expects list of dicts with keys 'modularity' and 'modularity_baseline'.
    """
    stats_res = {}
    try:
        modularity = [r.get('modularity', 0.0) for r in results_history]
        baseline = [r.get('modularity_baseline', 0.0) for r in results_history]
        
        if len(modularity) < 2:
            logger.info("Skipping Node2Vec significance test: need at least 2 iterations.")
            return {}

        # Paired t-test
        t_stat, p_val = ttest_rel(modularity, baseline)
        
        stats_res['t_statistic'] = float(t_stat)
        stats_res['p_value'] = float(p_val)
        stats_res['significant'] = bool(p_val < 0.05)
        
        logger.info(
            f"Node2Vec Significance Test (n={len(modularity)}): t={t_stat:.4f}, p={p_val:.4f} "
            f"({'Significant' if stats_res['significant'] else 'Not Significant'})"
        )
        
    except Exception:
        logger.exception("Node2Vec significance analysis failed.")
        
    return stats_res
