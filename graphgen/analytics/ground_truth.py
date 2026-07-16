"""Quantitative validation of detected topics against an expert ground truth.

The 'This is Europe' debates were independently analysed by the European
Parliamentary Research Service (EPRS; Drachenberg & Bącal, 2024), which
identified six recurring themes plus a set of dominant and peripheral
subjects.  This module turns the previously *qualitative* topic→theme
mapping (thesis §4.2) into reproducible metrics:

* **Best-match alignment** — each detected community summary is embedded
  and matched to its nearest EPRS theme by cosine similarity.
* **Coverage** — the fraction of EPRS themes recovered by at least one
  community (does the model find every expert theme?).
* **Mean alignment** — the average best-match similarity across
  communities (how cleanly do communities land on expert themes?).

The theme set is user-editable: pass a JSON file (a list of
``{"id", "name", "description"}`` objects) to override the defaults below,
e.g. after refining the descriptions from the EPRS briefing PDF.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


# Faithful to EPRS (2024) 'This is Europe' analysis as summarised in the
# thesis (§2.1, §4.2).  Descriptions are written to carry the vocabulary
# each theme uses so that sentence-embedding similarity is meaningful.
DEFAULT_EPRS_THEMES: List[Dict[str, str]] = [
    {
        "id": "value_of_membership",
        "name": "The value of EU membership",
        "description": (
            "The benefits and value of belonging to the European Union: the "
            "single market, solidarity between member states, shared prosperity, "
            "and what membership delivers to citizens and nations."
        ),
    },
    {
        "id": "defending_values",
        "name": "Defending EU values",
        "description": (
            "Defending fundamental European values: democracy, the rule of law, "
            "human rights, fundamental rights, judicial independence, anti-corruption, "
            "and the fight against authoritarianism and democratic backsliding."
        ),
    },
    {
        "id": "main_challenges",
        "name": "The main challenges facing the EU",
        "description": (
            "The principal challenges confronting Europe: Russia's war of aggression "
            "against Ukraine, the energy crisis and energy security, inflation and the "
            "cost of living, migration, climate change, and geopolitical threats."
        ),
    },
    {
        "id": "delivering_for_citizens",
        "name": "Delivering for EU citizens",
        "description": (
            "Delivering concrete results for citizens: economic stability, jobs, social "
            "policy, food security, healthcare, recovery from the pandemic, and protecting "
            "people from the effects of crises."
        ),
    },
    {
        "id": "next_steps_integration",
        "name": "Next steps in EU integration",
        "description": (
            "The future of European integration: enlargement to new candidate countries, "
            "treaty and institutional reform, strategic autonomy, deeper defence and "
            "security cooperation, and the Conference on the Future of Europe."
        ),
    },
    {
        "id": "importance_of_unity",
        "name": "The importance of EU unity",
        "description": (
            "The importance of European unity, cohesion, and acting together: a united "
            "response to crises, common sanctions, solidarity, and speaking with one voice "
            "on the world stage."
        ),
    },
]


def load_ground_truth_themes(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load EPRS themes from a JSON file, or return the documented defaults.

    The JSON file must contain a list of objects with ``id``, ``name`` and
    ``description`` keys.
    """
    if not path:
        return DEFAULT_EPRS_THEMES
    try:
        with open(path, "r", encoding="utf-8") as fh:
            themes = json.load(fh)
        if isinstance(themes, dict) and "themes" in themes:
            themes = themes["themes"]
        cleaned = [
            {"id": str(t.get("id", i)), "name": str(t.get("name", "")),
             "description": str(t.get("description", t.get("name", "")))}
            for i, t in enumerate(themes)
        ]
        logger.info("Loaded %d ground-truth themes from %s", len(cleaned), path)
        return cleaned
    except Exception:
        logger.exception("Failed to load ground-truth themes from %s; using defaults.", path)
        return DEFAULT_EPRS_THEMES


def _community_text(data: Dict[str, Any]) -> str:
    """Build the best available text description for a community node."""
    parts: List[str] = []
    for key in ("name", "title", "summary"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts)


def align_communities_to_ground_truth(
    graph: nx.DiGraph,
    themes: Optional[List[Dict[str, str]]] = None,
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Align detected community summaries to the EPRS ground-truth themes.

    For every COMMUNITY/TOPIC node carrying a title/summary, embed its text
    and its nearest EPRS theme, and report the full similarity matrix, the
    per-community best match, and corpus-level coverage / alignment metrics.

    Returns:
        Dict with ``metrics`` (coverage, mean/median best-match similarity,
        matched/total themes), ``mapping`` (per-community best theme), a
        ``similarity_matrix`` (community × theme), and the axis labels.  A
        ``skipped`` key is present when there are no summarised communities
        or the embedding model is unavailable.
    """
    from graphgen.pipeline.embeddings.rag import get_embedding_model

    themes = themes or DEFAULT_EPRS_THEMES

    community_ids: List[str] = []
    community_names: List[str] = []
    community_texts: List[str] = []
    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type", "")).upper() not in ("COMMUNITY", "TOPIC"):
            continue
        text = _community_text(data)
        if not text:
            continue
        community_ids.append(node_id)
        community_names.append(str(data.get("name", node_id)))
        community_texts.append(text)

    if len(community_ids) < 1:
        logger.warning("Ground-truth alignment skipped: no summarised community nodes.")
        return {"skipped": "No summarised community/topic nodes found"}

    model = get_embedding_model(embed_model_name)
    if model is None:
        return {"skipped": "Embedding model unavailable"}

    theme_texts = [f"{t['name']}. {t['description']}" for t in themes]
    theme_labels = [t["name"] for t in themes]

    comm_emb = np.asarray(model.encode(community_texts, normalize_embeddings=True))
    theme_emb = np.asarray(model.encode(theme_texts, normalize_embeddings=True))

    # Cosine similarity (embeddings already normalised) → communities × themes
    sim = comm_emb @ theme_emb.T

    best_theme_idx = sim.argmax(axis=1)
    best_sim = sim.max(axis=1)

    mapping = []
    for i, cid in enumerate(community_ids):
        mapping.append(
            {
                "community_id": cid,
                "community_name": community_names[i],
                "best_theme_id": themes[best_theme_idx[i]]["id"],
                "best_theme_name": theme_labels[best_theme_idx[i]],
                "similarity": float(best_sim[i]),
            }
        )

    matched_theme_ids = {themes[idx]["id"] for idx in best_theme_idx}
    coverage = len(matched_theme_ids) / len(themes)

    metrics = {
        "n_communities": len(community_ids),
        "n_themes": len(themes),
        "themes_matched": len(matched_theme_ids),
        "coverage": float(coverage),
        "mean_best_match_similarity": float(np.mean(best_sim)),
        "median_best_match_similarity": float(np.median(best_sim)),
        "min_best_match_similarity": float(np.min(best_sim)),
    }

    logger.info(
        "Ground-truth alignment: %d/%d EPRS themes covered (%.0f%%), "
        "mean best-match similarity=%.3f over %d communities.",
        metrics["themes_matched"],
        metrics["n_themes"],
        coverage * 100,
        metrics["mean_best_match_similarity"],
        metrics["n_communities"],
    )

    return {
        "metrics": metrics,
        "mapping": mapping,
        "similarity_matrix": sim.tolist(),
        "community_labels": community_names,
        "theme_labels": theme_labels,
    }


def export_ground_truth_alignment(
    alignment: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """Write the alignment mapping to CSV and the full report to JSON.

    Returns a dict of the written file paths (empty if alignment was skipped).
    """
    if not alignment or alignment.get("skipped"):
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    mapping = alignment.get("mapping", [])
    if mapping:
        csv_path = out / "ground_truth_alignment.csv"
        fieldnames = ["community_id", "community_name", "best_theme_id",
                      "best_theme_name", "similarity"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in mapping:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        written["csv"] = str(csv_path)

    json_path = out / "ground_truth_alignment.json"
    json_report = {k: v for k, v in alignment.items() if k != "similarity_matrix"}
    json_report["metrics"] = alignment.get("metrics", {})
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_report, fh, indent=2, ensure_ascii=False)
    written["json"] = str(json_path)

    logger.info("Ground-truth alignment written to %s", output_dir)
    return written
