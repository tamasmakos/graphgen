import os
import json
import logging
import networkx as nx
from graphgen.config.schema import GraphSchema, get_default_schema

logger = logging.getLogger(__name__)

def save_graph_schema(graph: nx.DiGraph, output_dir: str, schema_config=None):
    """
    Save the graph schema/metagraph to JSON.
    """
    if schema_config:
        try:
            schema = GraphSchema(**schema_config)
        except Exception as exc:
            logger.warning("Failed to parse explicit schema config, falling back to default schema: %s", exc)
            schema = get_default_schema()
    else:
        schema = get_default_schema()
    schema_path = os.path.join(output_dir, "graph_schema.json")
    try:
        with open(schema_path, 'w') as f:
            f.write(schema.model_dump_json(indent=2))
        logger.info(f"Schema saved to {schema_path}")
    except Exception as e:
        logger.error(f"Failed to save schema: {e}")
