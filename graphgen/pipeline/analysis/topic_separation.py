"""
Statistical Analysis of Topic/Community Embedding Separation.

Tests whether community embeddings show clear separation, supporting the
hypothesis that topics form distinct semantic clusters (as per GraphRAG approach).

Tests performed:
1. Silhouette Score - measures cluster quality
2. MANOVA - multivariate analysis of variance across groups  
3. Pairwise comparisons - compare specific community pairs
4. PCA analysis - dimensionality reduction for visualization data

Reference: https://graphrag.com/reference/knowledge-graph/lexical-graph-extracted-entities-community-summaries/
"""

import json
import logging
import numpy as np
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Statistical analysis imports
try:
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some statistical tests will be limited.")

try:
    from scipy import stats
    from scipy.stats import f_oneway, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some statistical tests will be limited.")


@dataclass
class LevelAnalysisResult:
    """Results for a single hierarchy level (COMMUNITY or SUBCOMMUNITY)."""
    level: str
    n_samples: int
    n_groups: int
    silhouette_score: Optional[float]
    silhouette_per_group: Optional[Dict[str, float]]
    anova_f_statistic: Optional[float]
    anova_p_value: Optional[float]
    multivariate_anova_on_pcs: Optional[Dict[str, float]]
    interpretation: str


@dataclass
class PairwiseComparison:
    """Result of a pairwise comparison between two groups."""
    group1: str
    group2: str
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool


@dataclass
class TopicSeparationReport:
    """Complete statistical analysis report."""
    timestamp: str
    config: Dict[str, Any]
    entity_level: Optional[LevelAnalysisResult]  # New field
    community_level: Optional[LevelAnalysisResult]
    subcommunity_level: Optional[LevelAnalysisResult]
    pairwise_comparisons: List[PairwiseComparison]
    pca_explained_variance: List[float]
    global_separation: Optional[float]  # Average pairwise cosine distance
    global_overlap: Optional[float] # Average pairwise cosine similarity
    overall_interpretation: str


def _align_embeddings_and_labels(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    common_ids = list(set(embeddings.keys()) & set(labels.keys()))
    X = np.array([embeddings[id_] for id_ in common_ids])
    y = np.array([labels[id_] for id_ in common_ids])
    return common_ids, X, y


def _compute_pca_scores(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    if not SKLEARN_AVAILABLE:
        return np.empty((0, 0)), np.array([])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    actual_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if actual_components < 1:
        return np.empty((0, 0)), np.array([])
    pca = PCA(n_components=actual_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_


def _anova_diagnostics(groups: Iterable[np.ndarray]) -> Dict[str, Any]:
    results: Dict[str, Any] = {"group_sizes": [int(len(g)) for g in groups]}
    if not SCIPY_AVAILABLE:
        results["skipped"] = "scipy_unavailable"
        return results

    group_list = [g for g in groups if len(g) > 0]
    if len(group_list) < 2:
        results["skipped"] = "insufficient_groups"
        return results

    shapiro_results = []
    for g in group_list:
        if len(g) < 3:
            shapiro_results.append({"n": int(len(g)), "statistic": None, "p_value": None})
            continue
        stat, p_val = stats.shapiro(g)
        shapiro_results.append({"n": int(len(g)), "statistic": float(stat), "p_value": float(p_val)})
    results["shapiro"] = shapiro_results

    try:
        lev_stat, lev_p = stats.levene(*group_list)
        results["levene"] = {"statistic": float(lev_stat), "p_value": float(lev_p)}
    except Exception as e:
        results["levene"] = {"error": str(e)}

    return results


def compute_global_separation(embeddings: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Compute average pairwise cosine distance and similarity between all embeddings.
    
    Returns:
        Tuple[float, float]: (average_distance, average_similarity)
    """
    if len(embeddings) < 2:
        return 0.0, 0.0
    
    try:
        X = np.array(list(embeddings.values()))
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms
        
        # Cosine Similarity Matrix
        sim_matrix = np.dot(X_norm, X_norm.T)
        
        # Cosine Distance = 1 - Similarity
        dist_matrix = 1.0 - sim_matrix
        
        # Average of upper triangle (excluding diagonal)
        n = X.shape[0]
        tri_indices = np.triu_indices(n, k=1)
        if len(tri_indices[0]) > 0:
            avg_dist = np.mean(dist_matrix[tri_indices])
            avg_sim = np.mean(sim_matrix[tri_indices])
            return float(avg_dist), float(avg_sim)
        return 0.0, 0.0
    except Exception as e:
        logger.error(f"Global separation calculation failed: {e}")
        return 0.0, 0.0


def extract_topic_embeddings(
    graph: nx.DiGraph,
    levels: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract embeddings for topic nodes (communities/subcommunities).
    
    For each community node, tries to find an embedding in this order:
    1. Node's own 'embedding' attribute (from summary embedding)
    2. Node's 'kge_embedding' attribute
    3. Mean of member entity embeddings
    
    Args:
        graph: NetworkX DiGraph containing the knowledge graph
        levels: Which hierarchy levels to extract
        
    Returns:
        Dictionary: {level: {node_id: embedding}}
    """
    if levels is None:
        levels = ["COMMUNITY", "SUBCOMMUNITY"]

    result: Dict[str, Dict[str, np.ndarray]] = {}
    
    for level in levels:
        level_embeddings: Dict[str, np.ndarray] = {}
        # Mapping level to node_type
        target_node_types = []
        if level == "COMMUNITY":
            target_node_types = ["COMMUNITY", "TOPIC"]
        elif level == "SUBCOMMUNITY":
            target_node_types = ["SUBCOMMUNITY", "SUBTOPIC"]
        else:
            target_node_types = [level]
            
        for node_id, node_data in graph.nodes(data=True):
            if str(node_data.get('node_type', '')).upper() not in target_node_types:
                continue
            
            embedding = None
            
            # Try 1: Direct embedding (from summary)
            if 'embedding' in node_data:
                emb = node_data['embedding']
                if isinstance(emb, list):
                    embedding = np.array(emb)
                elif isinstance(emb, np.ndarray):
                    embedding = emb
            
            # Try 2: KGE embedding
            if embedding is None and 'kge_embedding' in node_data:
                emb = node_data['kge_embedding']
                if isinstance(emb, list):
                    embedding = np.array(emb)
                elif isinstance(emb, np.ndarray):
                    embedding = emb
            
            # Try 3: Mean of member entity embeddings
            if embedding is None:
                member_embeddings = []
                for predecessor in graph.predecessors(node_id):
                    pred_data = graph.nodes.get(predecessor, {})
                    if str(pred_data.get('node_type', '')).upper() in ['ENTITY_CONCEPT', 'ENTITY', 'NAMEDENTITY', 'PLACE']:
                        for key in ['embedding', 'kge_embedding']:
                            if key in pred_data:
                                emb = pred_data[key]
                                if isinstance(emb, list):
                                    member_embeddings.append(np.array(emb))
                                elif isinstance(emb, np.ndarray):
                                    member_embeddings.append(emb)
                                break
                
                if member_embeddings:
                    embedding = np.mean(member_embeddings, axis=0)
            
            if embedding is not None:
                level_embeddings[node_id] = embedding
        
        result[level] = level_embeddings
        logger.info(f"Extracted {len(level_embeddings)} embeddings for {level} level")
    
    return result


def extract_entity_embeddings_and_labels(graph: nx.DiGraph) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Extract embeddings and community labels for ENTITY_CONCEPT nodes.
    Used for validating the clustering quality (Silhouette) at the entity level.
    """
    embeddings = {}
    labels = {}
    
    for node_id, node_data in graph.nodes(data=True):
        if str(node_data.get('node_type', '')).upper() not in ['ENTITY_CONCEPT', 'ENTITY', 'PLACE', 'NAMEDENTITY']:
            continue
            
        # Get embedding
        embedding = None
        if 'embedding' in node_data:
            emb = node_data['embedding']
            if isinstance(emb, list):
                embedding = np.array(emb)
            elif isinstance(emb, np.ndarray):
                embedding = emb
        
        # Get label (community ID)
        # 1. Try attribute
        label = node_data.get('community_id')
        
        # 2. Try edges to TOPIC nodes
        if label is None:
            for succ in graph.successors(node_id):
                edge_data = graph.get_edge_data(node_id, succ)
                # Check for IN_TOPIC or IN_COMMUNITY edge
                if edge_data.get('label') in ['IN_TOPIC', 'IN_COMMUNITY']:
                    succ_data = graph.nodes[succ]
                    # Check if successor is a TOPIC/COMMUNITY node
                    if succ_data.get('node_type') in ['TOPIC', 'COMMUNITY']:
                        # Try to get ID from node name TOPIC_X
                        try:
                            parts = str(succ).split('_')
                            if len(parts) > 1 and parts[1].isdigit():
                                label = int(parts[1])
                                break
                        except (IndexError, ValueError) as exc:
                            logger.debug("Failed to parse community label from %s: %s", succ, exc)
        
        if embedding is not None and label is not None:
            embeddings[node_id] = embedding
            labels[node_id] = int(label)
            
    return embeddings, labels


def run_silhouette_analysis(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int]
) -> Tuple[Optional[float], Optional[Dict[int, float]]]:
    """
    Run silhouette analysis on embeddings.
    
    Silhouette score measures how similar an object is to its own cluster
    compared to other clusters. Score ranges from -1 to 1, where:
    - 1: Well-clustered
    - 0: Overlapping clusters
    - -1: Misclassified
    
    Args:
        embeddings: Dictionary mapping node IDs to embedding vectors
        labels: Dictionary mapping node IDs to cluster labels
        
    Returns:
        Tuple of (overall_score, per_cluster_scores)
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available for silhouette analysis")
        return None, None

    # Align embeddings and labels
    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < 3:
        logger.warning(f"Insufficient samples for silhouette analysis: {len(common_ids)}")
        return None, None

    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])

    # Need at least 2 clusters
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        logger.warning("Need at least 2 clusters for silhouette analysis")
        return None, None

    if len(unique_labels) >= len(X):
        logger.info(f"Skipping silhouette analysis: too many clusters ({len(unique_labels)}) relative to samples ({len(X)})")
        return None, None

    try:
        # Use cosine metric for consistency with the embedding space
        # (KGE and RAG embeddings are compared via cosine similarity throughout the pipeline)
        overall = silhouette_score(X, y, metric='cosine')

        # Per-sample scores for per-cluster analysis
        sample_scores = silhouette_samples(X, y, metric='cosine')

        per_cluster = {}
        for label in unique_labels:
            mask = y == label
            if np.sum(mask) > 0:
                per_cluster[int(label)] = float(np.mean(sample_scores[mask]))

        return overall, per_cluster

    except Exception as e:
        logger.error(f"Silhouette analysis failed: {e}")
        return None, None


def run_silhouette_analysis_with_samples(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int]
) -> Tuple[Optional[float], Optional[Dict[int, float]], Optional[List[Dict[str, Any]]]]:
    """Run silhouette analysis and return per-sample scores."""
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available for silhouette analysis")
        return None, None, None

    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < 3:
        logger.warning(f"Insufficient samples for silhouette analysis: {len(common_ids)}")
        return None, None, None

    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        logger.warning("Need at least 2 clusters for silhouette analysis")
        return None, None, None

    if len(unique_labels) >= len(X):
        logger.info(f"Skipping silhouette analysis: too many clusters ({len(unique_labels)}) relative to samples ({len(X)})")
        return None, None, None

    try:
        overall = silhouette_score(X, y, metric='cosine')
        sample_scores = silhouette_samples(X, y, metric='cosine')

        per_cluster = {}
        for label in unique_labels:
            mask = y == label
            if np.sum(mask) > 0:
                per_cluster[int(label)] = float(np.mean(sample_scores[mask]))

        samples = [
            {
                "id": ids[idx],
                "label": int(y[idx]),
                "score": float(sample_scores[idx]),
            }
            for idx in range(len(ids))
        ]

        return overall, per_cluster, samples

    except Exception as e:
        logger.error(f"Silhouette analysis failed: {e}")
        return None, None, None


def run_anova_analysis(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run ANOVA-like analysis on embeddings.
    
    Since embeddings are multivariate, we reduce to first principal component
    and run one-way ANOVA to test if group means differ significantly.
    
    Args:
        embeddings: Dictionary mapping node IDs to embedding vectors
        labels: Dictionary mapping node IDs to cluster labels
        
    Returns:
        Tuple of (F-statistic, p-value)
    """
    if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.warning("scipy/sklearn not available for ANOVA")
        return None, None
    
    # Align embeddings and labels
    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < 3:
        return None, None
    
    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return None, None
    
    try:
        # Standardize before PCA to avoid scale-dominance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reduce to first PC for univariate ANOVA
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X_scaled).flatten()
        
        # Group by label
        groups = [X_pca[y == label] for label in unique_labels]
        
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return None, None
        
        # Run one-way ANOVA
        # Check if we have at least one group with > 1 sample to estimate variance
        if all(len(g) < 2 for g in groups):
             return None, None

        f_stat, p_value = f_oneway(*groups)
        
        return float(f_stat), float(p_value)
        
    except Exception as e:
        logger.error(f"ANOVA analysis failed: {e}")
        return None, None


def run_multivariate_anova_on_pcs(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int],
    n_components: int = 5,
    return_details: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Multiple univariate ANOVAs on principal components with Bonferroni correction.

    NOTE: This is NOT a true MANOVA (Pillai's trace, Wilks' Lambda, etc.).
    It reduces the embedding space to the top N principal components
    (after standardization), runs separate one-way ANOVAs on each PC,
    and combines p-values using Bonferroni correction.

    Limitations:
    - Ignores covariance between dependent variables (mitigated by PC orthogonality)
    - Bonferroni on minimum p-value is conservative
    - PCA maximises total variance, not group separation (LDA would be preferable)

    Args:
        embeddings: Dictionary mapping node IDs to embedding vectors
        labels: Dictionary mapping node IDs to cluster labels
        n_components: Number of PCs to analyze

    Returns:
        Dictionary with test statistics, or None if insufficient data
    """
    if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
        return None

    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < 10:
        return None

    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return None

    try:
        # Standardize before PCA to avoid scale-dominance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine actual number of components
        actual_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0] - 1)

        pca = PCA(n_components=actual_components)
        X_pca = pca.fit_transform(X_scaled)

        # Run ANOVA on each component
        f_stats = []
        p_values = []
        eta_squared_values = []
        per_component = []

        for i in range(actual_components):
            groups = [X_pca[y == label, i] for label in unique_labels]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue

            # Check if we have at least one group with > 1 sample
            if all(len(g) < 2 for g in groups):
                continue

            f_stat, p_val = f_oneway(*groups)
            f_stats.append(f_stat)
            p_values.append(p_val)

            # Compute eta-squared (effect size) for practical significance
            all_vals = X_pca[:, i]
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            eta_squared_values.append(eta_sq)
            per_component.append({
                "component": int(i + 1),
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "eta_squared": float(eta_sq),
            })

        if not p_values:
            return None

        # Combine using minimum p-value with Bonferroni correction
        min_p = min(p_values) * len(p_values)  # Bonferroni

        result: Dict[str, Any] = {
            "method": "multiple_univariate_anovas_on_pcs",
            "mean_f_statistic": float(np.mean(f_stats)),
            "min_p_value_corrected": float(min(min_p, 1.0)),
            "mean_eta_squared": float(np.mean(eta_squared_values)),
            "n_components_tested": len(p_values),
            "explained_variance_ratio": float(sum(pca.explained_variance_ratio_)),
            "note": "Not a true MANOVA; see docstring for limitations"
        }
        if return_details:
            result["per_component"] = per_component
            result["explained_variance_ratio_by_component"] = [
                float(v) for v in pca.explained_variance_ratio_
            ]
        return result

    except Exception as e:
        logger.error(f"Multivariate ANOVA on PCs failed: {e}")
        return None


def run_pairwise_comparisons(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int],
    alpha: float = 0.05,
    max_pairs: int = 20
) -> List[PairwiseComparison]:
    """
    Run pairwise t-tests between groups.
    
    Uses Welch's t-test on first principal component with Bonferroni correction.
    
    Args:
        embeddings: Dictionary mapping node IDs to embedding vectors
        labels: Dictionary mapping node IDs to cluster labels
        alpha: Significance level
        max_pairs: Maximum number of pairs to test
        
    Returns:
        List of PairwiseComparison results
    """
    if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
        return []
    
    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < 4:
        return []
    
    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])
    
    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce to first PC
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled).flatten()
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return []
    
    results = []
    n_comparisons = 0
    
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1:]:
            if n_comparisons >= max_pairs:
                break
            
            group1 = X_pca[y == label1]
            group2 = X_pca[y == label2]
            
            if len(group1) < 2 or len(group2) < 2:
                continue
            
            try:
                t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
                
                # Hedges' g effect size (corrects Cohen's d for small-sample bias)
                n1, n2 = len(group1), len(group2)
                pooled_std = np.sqrt(
                    ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) /
                    (n1 + n2 - 2)
                )
                if pooled_std > 0:
                    cohens_d = abs(np.mean(group1) - np.mean(group2)) / pooled_std
                    # Apply Hedges' correction factor for small samples
                    correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
                    cohens_d *= correction
                else:
                    cohens_d = 0.0
                
                # Apply Bonferroni correction
                n_total_comparisons = len(unique_labels) * (len(unique_labels) - 1) // 2
                corrected_alpha = alpha / max(1, n_total_comparisons)
                
                results.append(PairwiseComparison(
                    group1=f"group_{label1}",
                    group2=f"group_{label2}",
                    t_statistic=float(t_stat),
                    p_value=float(p_val),
                    effect_size=float(cohens_d),
                    significant=p_val < corrected_alpha
                ))
                
                n_comparisons += 1
                
            except Exception as e:
                logger.warning(f"Pairwise comparison failed for {label1} vs {label2}: {e}")
    
    return results


def compute_pca_variance(
    embeddings: Dict[str, np.ndarray],
    n_components: int = 10
) -> List[float]:
    """Compute PCA explained variance ratios."""
    if not SKLEARN_AVAILABLE:
        return []
    
    if len(embeddings) < 3:
        return []
    
    X = np.array(list(embeddings.values()))
    actual_components = min(n_components, X.shape[1], X.shape[0] - 1)
    
    if actual_components < 1:
        return []
    
    try:
        pca = PCA(n_components=actual_components)
        pca.fit(X)
        return [float(v) for v in pca.explained_variance_ratio_]
    except Exception as e:
        logger.error(f"PCA failed: {e}")
        return []


def interpret_separation(
    silhouette: Optional[float],
    p_value: Optional[float]
) -> str:
    """Generate human-readable interpretation of separation quality."""
    parts = []
    
    if silhouette is not None:
        if silhouette > 0.5:
            parts.append(f"Strong cluster separation (silhouette={silhouette:.3f})")
        elif silhouette > 0.25:
            parts.append(f"Moderate cluster separation (silhouette={silhouette:.3f})")
        elif silhouette > 0:
            parts.append(f"Weak cluster separation (silhouette={silhouette:.3f})")
        else:
            parts.append(f"Poor cluster separation (silhouette={silhouette:.3f})")
    
    if p_value is not None:
        if p_value < 0.001:
            parts.append(f"Highly significant difference (p<0.001)")
        elif p_value < 0.01:
            parts.append(f"Significant difference (p<0.01)")
        elif p_value < 0.05:
            parts.append(f"Marginally significant difference (p<0.05)")
        else:
            parts.append(f"No significant difference (p={p_value:.4f})")
    
    if not parts:
        return "Insufficient data for interpretation"
    
    return "; ".join(parts)


def analyze_level(
    embeddings: Dict[str, np.ndarray],
    level: str
) -> Optional[LevelAnalysisResult]:
    """
    Run complete analysis for a single hierarchy level.
    
    The labels for clustering are derived from the community/subcommunity IDs
    embedded in the node names (e.g., COMMUNITY_0, SUBCOMMUNITY_1_2).
    """
    if len(embeddings) < 3:
        logger.warning(f"Insufficient embeddings for {level} analysis: {len(embeddings)}")
        return None
    
    # Extract labels from node names
    labels: Dict[str, int] = {}
    for node_id in embeddings.keys():
        if (level == "COMMUNITY" or level == "TOPIC") and (node_id.startswith("COMMUNITY_") or node_id.startswith("TOPIC_")):
            try:
                label = int(node_id.split("_")[1])
                labels[node_id] = label
            except (IndexError, ValueError):
                continue
        elif (level == "SUBCOMMUNITY" or level == "SUBTOPIC") and (node_id.startswith("SUBCOMMUNITY_") or node_id.startswith("SUBTOPIC_")):
            try:
                # Use parent community as label for subcommunity analysis
                label = int(node_id.split("_")[1])
                labels[node_id] = label
            except (IndexError, ValueError):
                continue
    
    if len(labels) < 3:
        logger.warning(f"Insufficient labeled samples for {level}: {len(labels)}")
        return None
    
    # Run analyses
    silhouette, silhouette_per_group = run_silhouette_analysis(embeddings, labels)
    f_stat, p_value = run_anova_analysis(embeddings, labels)
    manova = run_multivariate_anova_on_pcs(embeddings, labels)
    
    interpretation = interpret_separation(silhouette, p_value)
    
    return LevelAnalysisResult(
        level=level,
        n_samples=len(labels),
        n_groups=len(set(labels.values())),
        silhouette_score=silhouette,
        silhouette_per_group={str(k): v for k, v in (silhouette_per_group or {}).items()},
        anova_f_statistic=f_stat,
        anova_p_value=p_value,
        multivariate_anova_on_pcs=manova,
        interpretation=interpretation
    )


def generate_topic_separation_report(
    graph: nx.DiGraph,
    output_path: str,
    config: Any
) -> Dict[str, Any]:
    """
    Generate complete topic separation statistical analysis report.
    
    This is the main entry point for the statistical testing pipeline step.
    
    Args:
        graph: NetworkX DiGraph containing the knowledge graph with communities
        output_path: Path to save the JSON report
        config: AnalysisSettings object with configuration
        
    Returns:
        Dictionary containing the report data
    """
    logger.info("Generating topic separation statistical report...")
    
    # Extract embeddings for all levels
    levels = config.hierarchy_levels if hasattr(config, 'hierarchy_levels') else ["COMMUNITY", "SUBCOMMUNITY"]
    all_embeddings = extract_topic_embeddings(graph, levels)
    
    # Analyze each level
    community_result = None
    subcommunity_result = None
    entity_result = None
    
    # 1. Entity Level (Clustering Validity)
    ent_embeddings, ent_labels = extract_entity_embeddings_and_labels(graph)
    entity_groups = len(set(ent_labels.values()))
    if len(ent_embeddings) > 5 and entity_groups > 1:
        logger.info(
            "Entity-level analysis: samples=%d, groups=%d.",
            len(ent_embeddings),
            entity_groups,
        )
        silhouette, silhouette_per_group = run_silhouette_analysis(ent_embeddings, ent_labels)
        manova = run_multivariate_anova_on_pcs(ent_embeddings, ent_labels)
        interpretation = interpret_separation(silhouette, None)
        
        entity_result = LevelAnalysisResult(
            level="ENTITY",
            n_samples=len(ent_embeddings),
            n_groups=len(set(ent_labels.values())),
            silhouette_score=silhouette,
            silhouette_per_group={str(k): v for k, v in (silhouette_per_group or {}).items()},
            anova_f_statistic=None,
            anova_p_value=None,
            multivariate_anova_on_pcs=manova,
            interpretation=interpretation
        )
        logger.info("Entity clustering silhouette: %s", silhouette)
    else:
        logger.info(
            "Skipping entity-level analysis: need >5 samples and >=2 groups (samples=%d, groups=%d).",
            len(ent_embeddings),
            entity_groups,
        )

    if "COMMUNITY" in all_embeddings and all_embeddings["COMMUNITY"]:
        community_result = analyze_level(all_embeddings["COMMUNITY"], "COMMUNITY")
    else:
        logger.info("Skipping community-level analysis: no community embeddings.")
    
    if "SUBCOMMUNITY" in all_embeddings and all_embeddings["SUBCOMMUNITY"]:
        subcommunity_result = analyze_level(all_embeddings["SUBCOMMUNITY"], "SUBCOMMUNITY")
    else:
        logger.info("Skipping subcommunity-level analysis: no subcommunity embeddings.")
    
    # Pairwise comparisons for communities
    pairwise = []
    if community_result and "COMMUNITY" in all_embeddings:
        comm_embeddings = all_embeddings["COMMUNITY"]
        labels = {}
        for node_id in comm_embeddings.keys():
            if node_id.startswith("COMMUNITY_"):
                try:
                    labels[node_id] = int(node_id.split("_")[1])
                except (IndexError, ValueError) as exc:
                    logger.debug("Failed to parse community label from %s: %s", node_id, exc)
                    continue
        pairwise = run_pairwise_comparisons(comm_embeddings, labels)
        logger.info("Pairwise community comparisons completed: %d", len(pairwise))
    
    # PCA variance
    pca_variance = []
    combined_embeddings = {}
    for level_embs in all_embeddings.values():
        combined_embeddings.update(level_embs)
    

    
    global_separation = 0.0
    global_overlap = 0.0
    if combined_embeddings:
        pca_variance = compute_pca_variance(combined_embeddings)
        global_separation, global_overlap = compute_global_separation(combined_embeddings)
    else:
        logger.info("Skipping global separation/overlap: no embeddings available.")
    
    # Overall interpretation
    interpretations = []
    if community_result:
        interpretations.append(f"Communities: {community_result.interpretation}")
    if subcommunity_result:
        interpretations.append(f"Subcommunities: {subcommunity_result.interpretation}")
    overall = "; ".join(interpretations) if interpretations else "Insufficient data for analysis"
    
    # Build report
    report = TopicSeparationReport(
        timestamp=datetime.now().isoformat(),
        config={"hierarchy_levels": levels},
        entity_level=entity_result,
        community_level=community_result,
        subcommunity_level=subcommunity_result,
        pairwise_comparisons=pairwise,
        pca_explained_variance=pca_variance,
        global_separation=global_separation,
        global_overlap=global_overlap,
        overall_interpretation=overall
    )
    
    # NOTE: user tip - for high-dimensional embeddings (e.g. OpenAI 1536d), 
    # consider using UMAP for dimensionality reduction before statistical tests 
    # if PCA shows instability or "curse of dimensionality" issues.
    
    # Convert to dictionary for JSON serialization
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    report_dict = to_dict(report)

    # Optional diagnostics and input exports
    outputs_subdir = getattr(config, "outputs_subdir", "thesis_outputs")
    base_dir = Path(output_path).parent
    thesis_dir = base_dir / outputs_subdir
    prefix = Path(output_path).stem

    def _labels_from_ids(node_ids: Iterable[str], level: str) -> Dict[str, int]:
        labels: Dict[str, int] = {}
        for node_id in node_ids:
            if level == "COMMUNITY" and (node_id.startswith("COMMUNITY_") or node_id.startswith("TOPIC_")):
                try:
                    labels[node_id] = int(node_id.split("_")[1])
                except (IndexError, ValueError):
                    continue
            elif level == "SUBCOMMUNITY" and (node_id.startswith("SUBCOMMUNITY_") or node_id.startswith("SUBTOPIC_")):
                try:
                    labels[node_id] = int(node_id.split("_")[1])
                except (IndexError, ValueError):
                    continue
        return labels

    level_data: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, int]]] = {}
    if ent_embeddings and ent_labels:
        level_data["ENTITY"] = (ent_embeddings, ent_labels)
    if "COMMUNITY" in all_embeddings and all_embeddings["COMMUNITY"]:
        comm_labels = _labels_from_ids(all_embeddings["COMMUNITY"].keys(), "COMMUNITY")
        level_data["COMMUNITY"] = (all_embeddings["COMMUNITY"], comm_labels)
    if "SUBCOMMUNITY" in all_embeddings and all_embeddings["SUBCOMMUNITY"]:
        sub_labels = _labels_from_ids(all_embeddings["SUBCOMMUNITY"].keys(), "SUBCOMMUNITY")
        level_data["SUBCOMMUNITY"] = (all_embeddings["SUBCOMMUNITY"], sub_labels)

    if getattr(config, "save_topic_separation_inputs", False):
        inputs_dir = thesis_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        inputs_payload: Dict[str, Any] = {}
        for level, (embeddings, labels) in level_data.items():
            if not embeddings:
                continue
            ids = [id_ for id_ in embeddings.keys() if id_ in labels]
            if not ids:
                continue
            X = np.array([embeddings[id_] for id_ in ids])
            y = np.array([labels[id_] for id_ in ids])
            inputs_payload[f"{level.lower()}_ids"] = np.array(ids)
            inputs_payload[f"{level.lower()}_embeddings"] = X
            inputs_payload[f"{level.lower()}_labels"] = y
            try:
                X_pca, evr = _compute_pca_scores(X, n_components=5)
                inputs_payload[f"{level.lower()}_pc_scores"] = X_pca
                inputs_payload[f"{level.lower()}_pc_explained_variance"] = evr
                if X_pca.size > 0:
                    inputs_payload[f"{level.lower()}_pc1_scores"] = X_pca[:, 0]
            except Exception:
                logger.exception("Failed to compute PCA scores for %s inputs.", level)
        inputs_path = inputs_dir / f"{prefix}_topic_separation_inputs.npz"
        np.savez_compressed(inputs_path, **inputs_payload)

    if getattr(config, "save_silhouette_samples", False):
        diag_dir = thesis_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        for level, (embeddings, labels) in level_data.items():
            overall, per_cluster, samples = run_silhouette_analysis_with_samples(embeddings, labels)
            if samples is None:
                continue
            payload = {
                "level": level,
                "overall_score": overall,
                "per_cluster": per_cluster,
                "samples": samples,
            }
            out_path = diag_dir / f"{prefix}_{level.lower()}_silhouette_samples.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

    if getattr(config, "save_anova_diagnostics", False):
        diag_dir = thesis_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        for level, (embeddings, labels) in level_data.items():
            if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
                continue
            common_ids = set(embeddings.keys()) & set(labels.keys())
            if len(common_ids) < 3:
                continue
            ids = list(common_ids)
            X = np.array([embeddings[id_] for id_ in ids])
            y = np.array([labels[id_] for id_ in ids])
            X_pca, _ = _compute_pca_scores(X, n_components=1)
            if X_pca.size == 0:
                continue
            pc1 = X_pca[:, 0]
            unique_labels = np.unique(y)
            groups = [pc1[y == label] for label in unique_labels]
            diagnostics = _anova_diagnostics(groups)
            diagnostics["level"] = level
            diagnostics["labels"] = [int(l) for l in unique_labels]
            out_path = diag_dir / f"{prefix}_{level.lower()}_anova_diagnostics.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    if getattr(config, "save_manova_details", False):
        diag_dir = thesis_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        for level, (embeddings, labels) in level_data.items():
            manova_details = run_multivariate_anova_on_pcs(
                embeddings,
                labels,
                return_details=True,
            )
            if not manova_details:
                continue
            out_path = diag_dir / f"{prefix}_{level.lower()}_manova_pc_details.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manova_details, f, indent=2, ensure_ascii=False)
    
    # Save to file
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Topic separation report saved to: {output_path}")
    except Exception:
        logger.exception("Failed to save report.")
    
    return report_dict
