"""
Ontology Label Extractor.

Parses RDF/OWL ontology files to extract class labels for use as
GLiNER entity extraction labels. This enables domain-specific entity
recognition based on ontology class definitions.

Usage:
    extractor = OntologyLabelExtractor("/app/input/ontology/cdm-4.13.2")
    labels = extractor.extract_labels()
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

try:
    from rdflib import Graph, URIRef, Literal
    from rdflib.namespace import OWL, RDF, RDFS
except ImportError:
    raise ImportError("rdflib is required. Install with: pip install rdflib")

logger = logging.getLogger(__name__)


class OntologyLabelExtractor:
    """
    Extracts class labels from RDF/OWL ontology files.
    
    This class parses RDF files from a specified directory and extracts
    rdfs:label values from OWL Classes and RDFS Classes. The extracted
    labels can be used for entity extraction systems like GLiNER.
    
    Attributes:
        ontology_dir: Path to directory containing RDF files.
        namespace_filter: Optional namespace prefix to filter classes.
        graph: RDFLib graph containing loaded ontology triples.
    """
    
    def __init__(
        self, 
        ontology_dir: str,
        namespace_filter: Optional[str] = None
    ):
        """
        Initialize the extractor.
        
        Args:
            ontology_dir: Path to directory containing RDF/OWL files.
            namespace_filter: Optional namespace URI prefix to filter classes.
                             Only classes starting with this prefix will be included.
        """
        self.ontology_dir = Path(ontology_dir)
        self.namespace_filter = namespace_filter
        self.graph = Graph()
        self._loaded = False
        
    def _load_rdf_files(self) -> int:
        """
        Load all RDF files from the ontology directory.
        
        Returns:
            Number of files successfully loaded.
        """
        if self._loaded:
            return 0
            
        if not self.ontology_dir.exists():
            logger.warning(f"Ontology directory not found: {self.ontology_dir}")
            return 0
        
        rdf_files = list(self.ontology_dir.glob("*.rdf"))
        if not rdf_files:
            logger.warning(f"No RDF files found in {self.ontology_dir}")
            return 0
        
        loaded_count = 0
        for rdf_file in rdf_files:
            try:
                self.graph.parse(rdf_file, format="xml")
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Could not parse {rdf_file.name}: {e}")
        
        self._loaded = True
        logger.info(f"Loaded {loaded_count} RDF files with {len(self.graph)} triples")
        return loaded_count
    
    def _get_label(self, uri: URIRef) -> Optional[str]:
        """
        Get the rdfs:label for a URI, preferring English labels.
        
        Args:
            uri: The URI to get the label for.
            
        Returns:
            The label string, or None if no label found.
        """
        labels = list(self.graph.objects(uri, RDFS.label))
        if not labels:
            return None
            
        # Prefer English labels
        for label in labels:
            if isinstance(label, Literal):
                if label.language == 'en' or not label.language:
                    return str(label)
        
        # Fallback to first available label
        return str(labels[0])
    
    def _extract_local_name(self, uri: str) -> str:
        """
        Extract the local name from a URI.
        
        Args:
            uri: Full URI string.
            
        Returns:
            The local name (fragment or last path segment).
        """
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]
    
    def extract_labels(
        self, 
        include_local_names: bool = True,
        top_level_only: bool = True,
        min_subclasses: int = 0
    ) -> List[str]:
        """
        Extract class labels from the loaded ontology with hierarchical filtering.
        
        Args:
            include_local_names: If True, use URI local names as fallback.
            top_level_only: If True, only include classes with no named parents.
            min_subclasses: only include classes with at least this many subclasses.
        
        Returns:
            List of unique class labels.
        """
        self._load_rdf_files()
        
        labels: Set[str] = set()
        
        # Get all OWL and RDFS Classes
        all_classes = set(self.graph.subjects(RDF.type, OWL.Class)).union(
                      set(self.graph.subjects(RDF.type, RDFS.Class)))
        
        logger.info(f"Analyzing {len(all_classes)} classes in ontology")
        
        # Technical classes to exclude
        exclude_classes = {
            OWL.Thing, OWL.Nothing, OWL.NamedIndividual,
            RDFS.Resource, RDFS.Class, OWL.Class
        }
        
        for cls in all_classes:
            # Skip blank nodes (complex OWL expressions)
            from rdflib import BNode
            if isinstance(cls, BNode) or cls in exclude_classes:
                continue
                
            cls_uri = str(cls)
            
            # Apply namespace filter if specified
            if self.namespace_filter and not cls_uri.startswith(self.namespace_filter):
                continue
            
            # Hierarchical filtering
            parents = list(self.graph.objects(cls, RDFS.subClassOf))
            # Filter parents: ignore blank nodes and technical classes
            named_parents = [p for p in parents if isinstance(p, URIRef) and p not in exclude_classes]
            
            if top_level_only and named_parents:
                continue
                
            if min_subclasses > 0:
                children = list(self.graph.subjects(RDFS.subClassOf, cls))
                if len(children) < min_subclasses:
                    continue
            
            # Try to get rdfs:label
            label = self._get_label(cls)
            
            if label:
                label = label.strip()
                if label and len(label) > 1:
                    labels.add(label)
            elif include_local_names:
                local_name = self._extract_local_name(cls_uri)
                if local_name and len(local_name) > 1:
                    readable = self._format_local_name(local_name)
                    # Filter out purely numerical or technical-looking local names
                    if any(c.isalpha() for c in readable):
                        labels.add(readable)
        
        result = sorted(list(labels))
        logger.info(f"Extracted {len(result)} unique labels from ontology (top_level={top_level_only}, min_children={min_subclasses})")
        return result
    
    def _format_local_name(self, name: str) -> str:
        """
        Format a local name into a readable label.
        
        Converts camelCase, snake_case, and hyphen-case to space-separated words.
        """
        import re
        
        # Replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Insert space before uppercase letters (camelCase)
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        
        # Title case and collapse multiple spaces
        name = ' '.join(name.split()).title()
        
        return name
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the loaded ontology.
        
        Returns:
            Dictionary with ontology statistics.
        """
        self._load_rdf_files()
        
        owl_classes = set(self.graph.subjects(RDF.type, OWL.Class))
        rdfs_classes = set(self.graph.subjects(RDF.type, RDFS.Class))
        
        return {
            "total_triples": len(self.graph),
            "owl_classes": len(owl_classes),
            "rdfs_classes": len(rdfs_classes),
            "total_classes": len(owl_classes.union(rdfs_classes)),
            "ontology_dir": str(self.ontology_dir),
            "namespace_filter": self.namespace_filter
        }


def _label_for(graph: "Graph", uri: "URIRef") -> Optional[str]:
    """Return the preferred (English) rdfs:label for *uri*, or None."""
    labels = list(graph.objects(uri, RDFS.label))
    if not labels:
        return None
    for label in labels:
        if isinstance(label, Literal) and (label.language == "en" or not label.language):
            return str(label)
    return str(labels[0])


def _format_local_name(name: str) -> str:
    """Turn a URI local name (camelCase / snake_case / hyphen) into words."""
    import re

    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return " ".join(name.split()).title()


def extract_ontology_file_metadata(
    rdf_file: str,
    top_level_only: bool = True,
    min_subclasses: int = 0,
    include_local_names: bool = True,
    max_labels: int = 60,
) -> Optional[dict]:
    """Parse a single RDF/OWL file into matcher-ready metadata.

    Extracts the ontology-level title/description (from ``dcterms`` /
    ``dc`` / ``rdfs:comment``) and the set of class labels contained in
    that one file.  Files with no named classes return ``None`` (there is
    nothing to extract against them).

    Args:
        rdf_file: Path to a single ``.rdf`` file.
        top_level_only: Only keep classes with no named superclass.
        min_subclasses: Only keep classes with at least this many children.
        include_local_names: Fall back to a formatted URI local name when a
            class has no ``rdfs:label``.
        max_labels: Cap on the number of labels returned (shortest first, so
            broad top-level types are preferred over deep technical ones).

    Returns:
        Dict with ``name``, ``title``, ``description``, ``labels`` and
        ``n_classes`` — or ``None`` if the file cannot be parsed or holds
        no classes.
    """
    from rdflib import BNode
    from rdflib.namespace import DCTERMS

    dc_title = URIRef("http://purl.org/dc/elements/1.1/title")
    dc_description = URIRef("http://purl.org/dc/elements/1.1/description")

    path = Path(rdf_file)
    graph = Graph()
    try:
        graph.parse(path, format="xml")
    except Exception as e:
        logger.warning("Could not parse ontology file %s: %s", path.name, e)
        return None

    exclude_classes = {
        OWL.Thing, OWL.Nothing, OWL.NamedIndividual,
        RDFS.Resource, RDFS.Class, OWL.Class,
    }

    all_classes = set(graph.subjects(RDF.type, OWL.Class)) | set(
        graph.subjects(RDF.type, RDFS.Class)
    )

    labels: Set[str] = set()
    for cls in all_classes:
        if isinstance(cls, BNode) or cls in exclude_classes:
            continue
        parents = list(graph.objects(cls, RDFS.subClassOf))
        named_parents = [p for p in parents if isinstance(p, URIRef) and p not in exclude_classes]
        if top_level_only and named_parents:
            continue
        if min_subclasses > 0:
            children = list(graph.subjects(RDFS.subClassOf, cls))
            if len(children) < min_subclasses:
                continue
        label = _label_for(graph, cls)
        if label:
            label = label.strip()
            if len(label) > 1:
                labels.add(label)
        elif include_local_names:
            local = str(cls).split("#")[-1].split("/")[-1]
            readable = _format_local_name(local)
            if len(readable) > 1 and any(c.isalpha() for c in readable):
                labels.add(readable)

    if not labels:
        return None

    # Ontology-level title & description.
    onts = list(graph.subjects(RDF.type, OWL.Ontology))
    title = ""
    description = ""
    if onts:
        o = onts[0]
        for pred in (DCTERMS.title, dc_title, RDFS.label):
            vals = list(graph.objects(o, pred))
            if vals:
                title = str(vals[0]).strip()
                break
        for pred in (DCTERMS.description, dc_description, RDFS.comment):
            vals = list(graph.objects(o, pred))
            if vals:
                description = str(vals[0]).strip()
                break

    sorted_labels = sorted(labels, key=lambda s: (len(s), s))[:max_labels]

    return {
        "name": title or path.stem,
        "title": title,
        "description": description,
        "labels": sorted_labels,
        "n_classes": len(all_classes),
    }


def extract_ontology_labels(
    ontology_dir: str,
    namespace_filter: Optional[str] = None,
    include_local_names: bool = True
) -> List[str]:
    """
    Convenience function to extract labels from an ontology directory.
    
    Args:
        ontology_dir: Path to directory containing RDF/OWL files.
        namespace_filter: Optional namespace prefix to filter classes.
        include_local_names: Use URI local names as fallback when no label.
        
    Returns:
        List of unique class labels.
    """
    extractor = OntologyLabelExtractor(ontology_dir, namespace_filter)
    return extractor.extract_labels(include_local_names)
