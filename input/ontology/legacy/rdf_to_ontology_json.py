#!/usr/bin/env python3
"""
EU Common Data Model (CDM) Ontology Parser

This script processes all RDF files in the CDM ontology directory and generates
a comprehensive JSON representation of the ontology structure, including:
- OWL Classes with hierarchies
- Object Properties and Datatype Properties
- RDFS labels, comments, and metadata
- Subclass relationships and property domains/ranges
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional

try:
    from rdflib import Graph, Namespace, URIRef, Literal, BNode
    from rdflib.namespace import OWL, RDF, RDFS, SKOS, DCTERMS
    from pyvis.network import Network
    import networkx as nx
except ImportError as e:
    print(f"Error: Required libraries missing. Install with: pip install rdflib pyvis networkx")
    print(f"Missing: {e}")
    sys.exit(1)

# Define CDM namespace
CDM = Namespace("http://publications.europa.eu/ontology/cdm#")

class CDMOntologyParser:
    """Parser for CDM RDF ontology files"""
    
    def __init__(self, ontology_dir: str = "/workspaces/kg/input/ontology/cdm-4.13.2"):
        self.ontology_dir = Path(ontology_dir)
        self.graph = Graph()
        self.ontology = {
            "metadata": {
                "name": "EU Common Data Model (CDM)",
                "version": "4.13.2",
                "description": "EU Publications Office Common Data Model ontology",
                "namespace": str(CDM),
                "source": "https://op.europa.eu/en/web/eu-vocabularies/dataset/-/resource?uri=http://publications.europa.eu/resource/dataset/cdm",
                "graph_type": "ontology_graph",
                "generation_timestamp": None,
                "total_files_processed": 0
            },
            "ontology_components": {},
            "namespaces": {},
            "classes": {},
            "object_properties": {},
            "datatype_properties": {},
            "annotation_properties": {},
            "individuals": {},
            "class_hierarchy": {},
            "statistics": {}
        }
        self.class_hierarchy = defaultdict(list)
        
    def load_rdf_files(self):
        """Load all RDF files from the ontology directory"""
        print(f"Loading RDF files from {self.ontology_dir}")
        
        if not self.ontology_dir.exists():
            raise FileNotFoundError(f"Ontology directory not found: {self.ontology_dir}")
        
        rdf_files = list(self.ontology_dir.glob("*.rdf"))
        if not rdf_files:
            raise FileNotFoundError(f"No RDF files found in {self.ontology_dir}")
        
        print(f"Found {len(rdf_files)} RDF files")
        
        for rdf_file in rdf_files:
            try:
                print(f"  Loading {rdf_file.name}...")
                self.graph.parse(rdf_file, format="xml")
            except Exception as e:
                print(f"  Warning: Could not parse {rdf_file.name}: {e}")
        
        print(f"Loaded {len(self.graph)} triples total")
        
    def extract_namespaces(self):
        """Extract namespace declarations"""
        print("Extracting namespaces...")
        
        for prefix, namespace in self.graph.namespaces():
            if prefix:  # Skip empty prefixes
                self.ontology["namespaces"][prefix] = str(namespace)
    
    def extract_ontology_metadata(self):
        """Extract metadata from owl:Ontology declarations in each RDF file"""
        print("Extracting ontology metadata from component files...")
        
        rdf_files = list(self.ontology_dir.glob("*.rdf"))
        
        for rdf_file in rdf_files:
            try:
                # Create a temporary graph for this file
                temp_graph = Graph()
                temp_graph.parse(rdf_file, format="xml")
                
                # Find owl:Ontology declarations
                ontology_uris = list(temp_graph.subjects(RDF.type, OWL.Ontology))
                
                for ont_uri in ontology_uris:
                    if isinstance(ont_uri, URIRef):
                        component_data = {
                            "source_file": rdf_file.name,
                            "ontology_uri": str(ont_uri),
                            "ontology_type": self.classify_ontology_type(rdf_file.name, temp_graph, ont_uri),
                            "title": self.get_label(ont_uri, temp_graph) or self.get_dc_title(ont_uri, temp_graph),
                            "description": self.get_comment(ont_uri, temp_graph) or self.get_dc_description(ont_uri, temp_graph),
                            "version": self.get_version_info(ont_uri, temp_graph),
                            "imports": [str(imp) for imp in temp_graph.objects(ont_uri, OWL.imports)],
                            "class_count": len(list(temp_graph.subjects(RDF.type, OWL.Class))),
                            "property_count": len(list(temp_graph.subjects(RDF.type, OWL.ObjectProperty))) + 
                                            len(list(temp_graph.subjects(RDF.type, OWL.DatatypeProperty)))
                        }
                        
                        # Use file name as component key if no URI found
                        component_key = ont_uri if ont_uri != URIRef("") else rdf_file.stem
                        self.ontology["ontology_components"][str(component_key)] = component_data
                
                # If no owl:Ontology found, create a component entry based on file analysis
                if not ontology_uris:
                    component_data = {
                        "source_file": rdf_file.name,
                        "ontology_uri": f"file://{rdf_file.name}",
                        "ontology_type": self.classify_ontology_type(rdf_file.name, temp_graph),
                        "title": rdf_file.stem.replace('_', ' ').title(),
                        "description": f"Component ontology from {rdf_file.name}",
                        "version": None,
                        "imports": [],
                        "class_count": len(list(temp_graph.subjects(RDF.type, OWL.Class))),
                        "property_count": len(list(temp_graph.subjects(RDF.type, OWL.ObjectProperty))) + 
                                        len(list(temp_graph.subjects(RDF.type, OWL.DatatypeProperty)))
                    }
                    self.ontology["ontology_components"][rdf_file.stem] = component_data
                    
            except Exception as e:
                print(f"  Warning: Could not extract metadata from {rdf_file.name}: {e}")
        
        print(f"Extracted metadata from {len(self.ontology['ontology_components'])} ontology components")
    
    def classify_ontology_type(self, filename: str, graph: Graph, ont_uri: URIRef = None) -> str:
        """Classify the type of ontology based on filename and content analysis"""
        filename_lower = filename.lower()
        
        # Filename-based classification
        if 'cdm' in filename_lower and 'plus' in filename_lower:
            return "cdm_extension"
        elif 'cdm' in filename_lower and any(x in filename_lower for x in ['annotation', 'datatype', 'cataloguing', 'indexation', 'marc21']):
            return "cdm_component"
        elif 'cdm' == filename_lower.replace('.rdf', ''):
            return "cdm_core"
        elif filename_lower.startswith(('skos', 'dcat', 'foaf', 'org', 'vcard')):
            return "external_standard"
        elif filename_lower in ['owl.rdf', 'rdfs.rdf']:
            return "w3c_standard"
        elif 'euvoc' in filename_lower:
            return "eu_vocabulary"
        elif any(x in filename_lower for x in ['import', 'policy']):
            return "configuration"
        
        # Content-based classification
        class_count = len(list(graph.subjects(RDF.type, OWL.Class)))
        prop_count = len(list(graph.subjects(RDF.type, OWL.ObjectProperty)))
        
        if class_count > 100:
            return "core_ontology"
        elif prop_count > class_count * 2:
            return "property_ontology"
        elif class_count > 10:
            return "domain_ontology"
        else:
            return "utility_ontology"
    
    def get_dc_title(self, uri: URIRef, graph: Graph) -> Optional[str]:
        """Get Dublin Core title"""
        dc_title = list(graph.objects(uri, DCTERMS.title))
        if dc_title:
            return str(dc_title[0])
        return None
    
    def get_dc_description(self, uri: URIRef, graph: Graph) -> Optional[str]:
        """Get Dublin Core description"""
        dc_desc = list(graph.objects(uri, DCTERMS.description))
        if dc_desc:
            return str(dc_desc[0])
        return None
    
    def get_version_info(self, uri: URIRef, graph: Graph) -> Optional[str]:
        """Get version information"""
        version = list(graph.objects(uri, OWL.versionInfo))
        if version:
            return str(version[0])
        return None
    
    def get_label(self, uri: URIRef, graph: Graph = None) -> Optional[str]:
        """Get rdfs:label for a URI from specific graph or main graph"""
        target_graph = graph if graph is not None else self.graph
        labels = list(target_graph.objects(uri, RDFS.label))
        if labels:
            # Prefer English labels if available
            for label in labels:
                if isinstance(label, Literal):
                    if label.language == 'en' or not label.language:
                        return str(label)
            return str(labels[0])
        return None
    
    def get_comment(self, uri: URIRef, graph: Graph = None) -> Optional[str]:
        """Get rdfs:comment for a URI from specific graph or main graph"""
        target_graph = graph if graph is not None else self.graph
        comments = list(target_graph.objects(uri, RDFS.comment))
        if comments:
            # Prefer English comments if available
            for comment in comments:
                if isinstance(comment, Literal):
                    if comment.language == 'en' or not comment.language:
                        return str(comment)
            return str(comments[0])
        return None
    
    def get_literals(self, uri: URIRef, predicate: URIRef) -> List[str]:
        """Get all literal values for a predicate"""
        return [str(obj) for obj in self.graph.objects(uri, predicate) 
                if isinstance(obj, Literal)]
    
    def extract_classes(self):
        """Extract OWL classes and RDFS classes"""
        print("Extracting classes...")
        
        # Get all OWL classes
        owl_classes = set(self.graph.subjects(RDF.type, OWL.Class))
        # Get all RDFS classes
        rdfs_classes = set(self.graph.subjects(RDF.type, RDFS.Class))
        
        all_classes = owl_classes.union(rdfs_classes)
        
        for cls in all_classes:
            if isinstance(cls, URIRef):
                class_data = {
                    "uri": str(cls),
                    "type": "owl:Class" if cls in owl_classes else "rdfs:Class",
                    "label": self.get_label(cls),
                    "comment": self.get_comment(cls),
                    "subclass_of": [],
                    "superclass_of": [],
                    "equivalent_classes": [],
                    "disjoint_with": []
                }
                
                # Get subclass relationships
                for superclass in self.graph.objects(cls, RDFS.subClassOf):
                    if isinstance(superclass, URIRef):
                        class_data["subclass_of"].append(str(superclass))
                        self.class_hierarchy[str(superclass)].append(str(cls))
                
                # Get equivalent classes
                for equiv in self.graph.objects(cls, OWL.equivalentClass):
                    if isinstance(equiv, URIRef):
                        class_data["equivalent_classes"].append(str(equiv))
                
                # Get disjoint classes
                for disjoint in self.graph.objects(cls, OWL.disjointWith):
                    if isinstance(disjoint, URIRef):
                        class_data["disjoint_with"].append(str(disjoint))
                
                self.ontology["classes"][str(cls)] = class_data
        
        # Build reverse hierarchy (superclass_of relationships)
        for class_uri, subclasses in self.class_hierarchy.items():
            if class_uri in self.ontology["classes"]:
                self.ontology["classes"][class_uri]["superclass_of"] = subclasses
        
        # Ensure all classes have proper list initialization (not None)
        for class_uri, class_data in self.ontology["classes"].items():
            for list_field in ["subclass_of", "superclass_of", "equivalent_classes", "disjoint_with"]:
                if class_data[list_field] is None:
                    class_data[list_field] = []
        
        print(f"Extracted {len(self.ontology['classes'])} classes")
    
    def extract_properties(self):
        """Extract object properties, datatype properties, and annotation properties"""
        print("Extracting properties...")
        
        # Object Properties
        for prop in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            if isinstance(prop, URIRef):
                prop_data = {
                    "uri": str(prop),
                    "label": self.get_label(prop),
                    "comment": self.get_comment(prop),
                    "domain": [str(d) for d in self.graph.objects(prop, RDFS.domain) if isinstance(d, URIRef)],
                    "range": [str(r) for r in self.graph.objects(prop, RDFS.range) if isinstance(r, URIRef)],
                    "subproperty_of": [str(s) for s in self.graph.objects(prop, RDFS.subPropertyOf) if isinstance(s, URIRef)],
                    "superproperty_of": [],
                    "inverse_of": [str(i) for i in self.graph.objects(prop, OWL.inverseOf) if isinstance(i, URIRef)],
                    "functional": (prop, RDF.type, OWL.FunctionalProperty) in self.graph,
                    "inverse_functional": (prop, RDF.type, OWL.InverseFunctionalProperty) in self.graph,
                    "transitive": (prop, RDF.type, OWL.TransitiveProperty) in self.graph,
                    "symmetric": (prop, RDF.type, OWL.SymmetricProperty) in self.graph
                }
                self.ontology["object_properties"][str(prop)] = prop_data
        
        # Datatype Properties
        for prop in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            if isinstance(prop, URIRef):
                prop_data = {
                    "uri": str(prop),
                    "label": self.get_label(prop),
                    "comment": self.get_comment(prop),
                    "domain": [str(d) for d in self.graph.objects(prop, RDFS.domain) if isinstance(d, URIRef)],
                    "range": [str(r) for r in self.graph.objects(prop, RDFS.range)],
                    "subproperty_of": [str(s) for s in self.graph.objects(prop, RDFS.subPropertyOf) if isinstance(s, URIRef)],
                    "superproperty_of": [],
                    "functional": (prop, RDF.type, OWL.FunctionalProperty) in self.graph
                }
                self.ontology["datatype_properties"][str(prop)] = prop_data
        
        # Annotation Properties
        for prop in self.graph.subjects(RDF.type, OWL.AnnotationProperty):
            if isinstance(prop, URIRef):
                prop_data = {
                    "uri": str(prop),
                    "label": self.get_label(prop),
                    "comment": self.get_comment(prop),
                    "subproperty_of": [str(s) for s in self.graph.objects(prop, RDFS.subPropertyOf) if isinstance(s, URIRef)]
                }
                self.ontology["annotation_properties"][str(prop)] = prop_data
        
        # Build superproperty relationships
        for prop_dict in [self.ontology["object_properties"], self.ontology["datatype_properties"]]:
            for prop_uri, prop_data in prop_dict.items():
                for subprop_uri in prop_data["subproperty_of"]:
                    if subprop_uri in prop_dict:
                        prop_dict[subprop_uri]["superproperty_of"].append(prop_uri)
        
        print(f"Extracted {len(self.ontology['object_properties'])} object properties")
        print(f"Extracted {len(self.ontology['datatype_properties'])} datatype properties")
        print(f"Extracted {len(self.ontology['annotation_properties'])} annotation properties")
    
    def extract_individuals(self):
        """Extract named individuals"""
        print("Extracting individuals...")
        
        for individual in self.graph.subjects(RDF.type, OWL.NamedIndividual):
            if isinstance(individual, URIRef):
                # Get types
                types = [str(t) for t in self.graph.objects(individual, RDF.type) 
                        if isinstance(t, URIRef) and t != OWL.NamedIndividual]
                
                individual_data = {
                    "uri": str(individual),
                    "label": self.get_label(individual),
                    "comment": self.get_comment(individual),
                    "types": types
                }
                self.ontology["individuals"][str(individual)] = individual_data
        
        print(f"Extracted {len(self.ontology['individuals'])} individuals")
    
    def build_class_hierarchy_tree(self):
        """Build a hierarchical tree structure of classes"""
        print("Building class hierarchy tree...")
        
        # Find root classes (classes with no superclasses)
        root_classes = []
        for class_uri, class_data in self.ontology["classes"].items():
            if not class_data["subclass_of"]:
                root_classes.append(class_uri)
        
        def build_subtree(class_uri):
            """Recursively build hierarchy subtree"""
            subtree = {
                "uri": class_uri,
                "label": self.ontology["classes"].get(class_uri, {}).get("label"),
                "children": []
            }
            
            # Add children
            if class_uri in self.ontology["classes"]:
                for child_uri in self.ontology["classes"][class_uri]["superclass_of"]:
                    subtree["children"].append(build_subtree(child_uri))
            
            return subtree
        
        hierarchy_tree = []
        for root_class in root_classes:
            hierarchy_tree.append(build_subtree(root_class))
        
        self.ontology["class_hierarchy"] = hierarchy_tree
        
        print(f"Built hierarchy tree with {len(root_classes)} root classes")
    
    def calculate_statistics(self):
        """Calculate ontology statistics"""
        print("Calculating statistics...")
        
        stats = {
            "total_classes": len(self.ontology["classes"]),
            "total_object_properties": len(self.ontology["object_properties"]),
            "total_datatype_properties": len(self.ontology["datatype_properties"]),
            "total_annotation_properties": len(self.ontology["annotation_properties"]),
            "total_individuals": len(self.ontology["individuals"]),
            "total_triples": len(self.graph),
            "namespaces_count": len(self.ontology["namespaces"]),
            "ontology_components_count": len(self.ontology["ontology_components"])
        }
        
        # CDM-specific statistics
        cdm_classes = [uri for uri in self.ontology["classes"] if uri.startswith(str(CDM))]
        cdm_properties = [uri for uri in self.ontology["object_properties"] if uri.startswith(str(CDM))]
        cdm_properties.extend([uri for uri in self.ontology["datatype_properties"] if uri.startswith(str(CDM))])
        
        stats["cdm_classes"] = len(cdm_classes)
        stats["cdm_properties"] = len(cdm_properties)
        
        # FRBR core classes
        frbr_core = ["work", "expression", "manifestation", "item"]
        frbr_classes = []
        for core in frbr_core:
            frbr_uri = str(CDM) + core
            if frbr_uri in self.ontology["classes"]:
                frbr_classes.append(frbr_uri)
        
        stats["frbr_core_classes"] = frbr_classes
        
        # Ontology component type breakdown
        component_types = {}
        for comp_data in self.ontology["ontology_components"].values():
            ont_type = comp_data["ontology_type"]
            if ont_type not in component_types:
                component_types[ont_type] = {
                    "count": 0,
                    "total_classes": 0,
                    "total_properties": 0,
                    "files": []
                }
            component_types[ont_type]["count"] += 1
            component_types[ont_type]["total_classes"] += comp_data["class_count"]
            component_types[ont_type]["total_properties"] += comp_data["property_count"]
            component_types[ont_type]["files"].append(comp_data["source_file"])
        
        stats["component_types"] = component_types
        
        self.ontology["statistics"] = stats
        
        print(f"Statistics calculated: {stats['total_classes']} classes, {stats['total_object_properties']} object properties")
        print(f"Ontology components: {stats['ontology_components_count']} files with {len(component_types)} different types")
    
    def get_node_color(self, uri: str) -> str:
        """Get color for a node based on its namespace and type"""
        if uri.startswith(str(CDM)):
            # FRBR core classes get special colors
            if uri.endswith('#work'):
                return '#FF6B6B'  # Red for Work
            elif uri.endswith('#expression'):
                return '#4ECDC4'  # Teal for Expression
            elif uri.endswith('#manifestation'):
                return '#45B7D1'  # Blue for Manifestation
            elif uri.endswith('#item'):
                return '#96CEB4'  # Green for Item
            elif 'agent' in uri.lower():
                return '#FFEAA7'  # Yellow for Agent classes
            elif 'dossier' in uri.lower():
                return '#DDA0DD'  # Plum for Dossier classes
            elif 'event' in uri.lower():
                return '#F39C12'  # Orange for Event classes
            else:
                return '#74B9FF'  # Light blue for other CDM classes
        elif 'skos' in uri.lower():
            return '#E17055'  # Coral for SKOS
        elif 'dcat' in uri.lower():
            return '#A29BFE'  # Purple for DCAT
        elif 'foaf' in uri.lower():
            return '#FD79A8'  # Pink for FOAF
        elif 'org' in uri.lower():
            return '#FDCB6E'  # Yellow for ORG
        elif 'vcard' in uri.lower():
            return '#6C5CE7'  # Violet for VCard
        else:
            return '#B2DFDB'  # Light gray for others
    
    def get_node_size(self, uri: str, class_data: Dict) -> int:
        """Calculate node size based on number of subclasses and importance"""
        base_size = 20
        
        # FRBR core classes are larger
        if uri in [str(CDM) + core for core in ['work', 'expression', 'manifestation', 'item']]:
            base_size = 50
        
        # Size based on number of subclasses
        superclass_of = class_data.get('superclass_of', []) or []
        subclass_count = len(superclass_of)
        size_boost = min(subclass_count * 3, 30)  # Cap at 30
        
        return base_size + size_boost
    
    def create_ontology_visualization(self, output_file: str = "cdm_ontology_graph.html", 
                                    include_properties: bool = True):
        """Create an interactive visualization of the ontology using Pyvis"""
        total_classes = len(self.ontology["classes"])
        print(f"Creating ontology visualization with all {total_classes} nodes...")
        
        # Initialize Pyvis network
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#222222", 
            font_color="white",
            directed=True
        )
        
        # Configure physics for better layout with more nodes
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"enabled": true, "iterations": 200},
            "barnesHut": {
              "gravitationalConstant": -12000,
              "centralGravity": 0.1,
              "springLength": 120,
              "springConstant": 0.02,
              "damping": 0.15
            }
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          },
          "layout": {
            "hierarchical": {
              "enabled": false
            }
          }
        }
        """)
        
        # Use ALL classes - no limitation
        selected_classes = list(self.ontology["classes"].keys())
        
        # FRBR core classes for special styling
        frbr_core = [str(CDM) + core for core in ['work', 'expression', 'manifestation', 'item']]
        
        print(f"Visualizing all {len(selected_classes)} classes")
        
        # Add nodes
        for i, uri in enumerate(selected_classes):
            if i % 100 == 0:
                print(f"Processing node {i+1}/{len(selected_classes)}: {uri}")
                
            try:
                class_data = self.ontology["classes"][uri]
                
                # Create node label
                label = class_data.get('label') or uri.split('#')[-1].split('/')[-1]
                if len(label) > 30:
                    label = label[:27] + "..."
                
                # Create hover info
                comment = class_data.get('comment', 'No description available')
                if len(comment) > 200:
                    comment = comment[:197] + "..."
                
                hover_info = f"""
                <b>{label}</b><br/>
                URI: {uri}<br/>
                Type: {class_data.get('type', 'Unknown')}<br/>
                Subclasses: {len(class_data.get('superclass_of', []) or [])}<br/>
                Superclasses: {len(class_data.get('subclass_of', []) or [])}<br/>
                <br/>
                <i>{comment}</i>
                """
                
                # Determine node properties
                color = self.get_node_color(uri)
                size = self.get_node_size(uri, class_data)
                
                # Special styling for FRBR core
                if uri in frbr_core:
                    net.add_node(
                        uri, 
                        label=f"[FRBR] {label}",
                        title=hover_info,
                        color=color,
                        size=size,
                        borderWidth=3,
                        borderWidthSelected=5,
                        font={'size': 16, 'face': 'arial black'}
                    )
                else:
                    net.add_node(
                        uri, 
                        label=label,
                        title=hover_info,
                        color=color,
                        size=size
                    )
            except Exception as e:
                print(f"Error processing node {uri}: {e}")
                continue
        
        # Add subclass relationships
        edge_count = 0
        for class_uri in selected_classes:
            class_data = self.ontology["classes"][class_uri]
            
            # Add subclass edges
            subclass_of = class_data.get('subclass_of', []) or []
            for superclass_uri in subclass_of:
                if superclass_uri in selected_classes and class_uri in selected_classes:
                    try:
                        net.add_edge(
                            class_uri, 
                            superclass_uri, 
                            label="subClassOf",
                            color={'color': '#848484', 'highlight': '#848484'},
                            width=1,
                            arrows={'to': {'enabled': True, 'scaleFactor': 0.8}}
                        )
                        edge_count += 1
                    except Exception as e:
                        print(f"Error adding edge from {class_uri} to {superclass_uri}: {e}")
        
        # Add some key object properties if requested
        property_edges_added = 0
        if include_properties and edge_count < 1000:  # Don't overwhelm the graph
            max_property_edges = min(100, 1000 - edge_count)
            
            # Focus on CDM-specific properties
            cdm_properties = [(uri, data) for uri, data in self.ontology["object_properties"].items() 
                            if uri.startswith(str(CDM)) and data.get('label')]
            
            for prop_uri, prop_data in cdm_properties[:50]:  # Limit to top 50 properties
                if property_edges_added >= max_property_edges:
                    break
                    
                domains = prop_data.get('domain', []) or []
                ranges = prop_data.get('range', []) or []
                
                for domain in domains:
                    for range_cls in ranges:
                        if domain in selected_classes and range_cls in selected_classes:
                            try:
                                label = prop_data.get('label', prop_uri.split('#')[-1])
                                if len(label) > 20:
                                    label = label[:17] + "..."
                                    
                                net.add_edge(
                                    domain,
                                    range_cls,
                                    label=label,
                                    color={'color': '#FFA726', 'highlight': '#FF9800'},
                                    width=2,
                                    dashes=True,
                                    arrows={'to': {'enabled': True, 'scaleFactor': 1.0}}
                                )
                                property_edges_added += 1
                            except Exception as e:
                                print(f"Error adding property edge {prop_uri} from {domain} to {range_cls}: {e}")
                            
                            if property_edges_added >= max_property_edges:
                                break
        
        print(f"Added {edge_count} subclass relationships and {property_edges_added if include_properties else 0} property relationships")
        
        # Add legend
        legend_html = """
        <div style="position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.8); 
                    color: white; padding: 15px; border-radius: 10px; font-family: Arial; z-index: 1000;">
            <h3 style="margin-top: 0;">CDM Ontology Legend</h3>
            <div><span style="color: #FF6B6B;">●</span> Work (FRBR Core)</div>
            <div><span style="color: #4ECDC4;">●</span> Expression (FRBR Core)</div>
            <div><span style="color: #45B7D1;">●</span> Manifestation (FRBR Core)</div>
            <div><span style="color: #96CEB4;">●</span> Item (FRBR Core)</div>
            <div><span style="color: #FFEAA7;">●</span> Agent Classes</div>
            <div><span style="color: #DDA0DD;">●</span> Dossier Classes</div>
            <div><span style="color: #F39C12;">●</span> Event Classes</div>
            <div><span style="color: #74B9FF;">●</span> Other CDM Classes</div>
            <hr style="border-color: #666;">
            <div><span style="color: #848484;">→</span> subClassOf</div>
            <div><span style="color: #FFA726;">⇢</span> Object Property</div>
            <p style="font-size: 12px; margin-bottom: 0;">
                [FRBR] = FRBR Core Classes<br/>
                Node size = Number of subclasses<br/>
                Hover for details
            </p>
        </div>
        """
        
        # Save the visualization
        net.save_graph(output_file)
        
        # Add legend to the HTML file
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Insert legend after the body tag
        content = content.replace('<body>', f'<body>{legend_html}')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Interactive visualization saved to {output_file}")
        print(f"Visualization includes {len(selected_classes)} nodes and {edge_count + (property_edges_added if include_properties else 0)} edges")
        
        return output_file
    
    def parse(self) -> Dict[str, Any]:
        """Main parsing method"""
        from datetime import datetime
        
        print("Starting CDM ontology parsing...")
        
        # Set generation timestamp
        self.ontology["metadata"]["generation_timestamp"] = datetime.now().isoformat()
        
        # Load all RDF files
        self.load_rdf_files()
        
        # Set total files processed
        rdf_files = list(self.ontology_dir.glob("*.rdf"))
        self.ontology["metadata"]["total_files_processed"] = len(rdf_files)
        
        # Extract all components
        self.extract_namespaces()
        self.extract_ontology_metadata()
        self.extract_classes()
        self.extract_properties()
        self.extract_individuals()
        self.assign_ontology_type_labels()
        self.build_class_hierarchy_tree()
        self.calculate_statistics()
        
        print("Parsing completed successfully!")
        return self.ontology
    
    def save_json(self, output_file: str = "../output/eu_cdm_ontology.json"):
        """Save ontology to JSON file"""
        print(f"Saving ontology to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.ontology, f, indent=2, ensure_ascii=False)
        
        print(f"Ontology saved to {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")

    def assign_ontology_type_labels(self):
        """Assign ontology_type labels to classes based on their source components"""
        print("Assigning ontology type labels to classes...")
        
        # Build mapping of URIs to ontology types based on file analysis
        uri_to_ontology_type = {}
        
        # Analyze each RDF file to determine which classes belong to which ontology type
        rdf_files = list(self.ontology_dir.glob("*.rdf"))
        
        for rdf_file in rdf_files:
            try:
                temp_graph = Graph()
                temp_graph.parse(rdf_file, format="xml")
                
                # Get ontology type for this file
                ontology_type = self.classify_ontology_type(rdf_file.name, temp_graph)
                
                # Find all classes defined in this file
                for cls in temp_graph.subjects(RDF.type, OWL.Class):
                    if isinstance(cls, URIRef):
                        uri_to_ontology_type[str(cls)] = ontology_type
                
                # Also check for RDFS classes
                for cls in temp_graph.subjects(RDF.type, RDFS.Class):
                    if isinstance(cls, URIRef):
                        if str(cls) not in uri_to_ontology_type:  # Don't overwrite OWL class classification
                            uri_to_ontology_type[str(cls)] = ontology_type
                            
            except Exception as e:
                print(f"  Warning: Could not analyze {rdf_file.name} for type assignment: {e}")
        
        # Assign ontology_type to classes
        assigned_count = 0
        for class_uri, class_data in self.ontology["classes"].items():
            if class_uri in uri_to_ontology_type:
                class_data["ontology_type"] = uri_to_ontology_type[class_uri]
                assigned_count += 1
            else:
                # Default classification based on URI patterns
                if class_uri.startswith(str(CDM)):
                    class_data["ontology_type"] = "cdm_core"
                elif "skos" in class_uri.lower():
                    class_data["ontology_type"] = "external_standard"
                elif "foaf" in class_uri.lower():
                    class_data["ontology_type"] = "external_standard"
                elif "dcat" in class_uri.lower():
                    class_data["ontology_type"] = "external_standard"
                elif "org" in class_uri.lower():
                    class_data["ontology_type"] = "external_standard"
                elif "w3.org" in class_uri:
                    class_data["ontology_type"] = "w3c_standard"
                else:
                    class_data["ontology_type"] = "unclassified"
        
        print(f"Assigned ontology types to {assigned_count} classes")
        
        # Also assign to properties
        self.assign_property_ontology_types()
    
    def assign_property_ontology_types(self):
        """Assign ontology_type labels to properties"""
        property_collections = [
            ("object_properties", self.ontology["object_properties"]),
            ("datatype_properties", self.ontology["datatype_properties"]),
            ("annotation_properties", self.ontology["annotation_properties"])
        ]
        
        for prop_type_name, properties in property_collections:
            for prop_uri, prop_data in properties.items():
                # Classification based on URI patterns
                if prop_uri.startswith(str(CDM)):
                    prop_data["ontology_type"] = "cdm_core"
                elif "skos" in prop_uri.lower():
                    prop_data["ontology_type"] = "external_standard"
                elif "foaf" in prop_uri.lower():
                    prop_data["ontology_type"] = "external_standard"
                elif "dcat" in prop_uri.lower():
                    prop_data["ontology_type"] = "external_standard"
                elif "org" in prop_uri.lower():
                    prop_data["ontology_type"] = "external_standard"
                elif "w3.org" in prop_uri:
                    prop_data["ontology_type"] = "w3c_standard"
                else:
                    prop_data["ontology_type"] = "unclassified"

def main():
    """Main function"""
    parser = CDMOntologyParser()
    
    try:
        # Parse the ontology
        ontology = parser.parse()
        
        # Save to JSON
        parser.save_json()
        
        # Create interactive visualization
        print("\n" + "="*50)
        print("CREATING INTERACTIVE VISUALIZATION")
        print("="*50)
        visualization_file = parser.create_ontology_visualization(
            output_file="../output/cdm_ontology_graph.html",
            include_properties=True
        )
        
        # Print summary
        stats = ontology["statistics"]
        print("\n" + "="*50)
        print("CDM ONTOLOGY EXTRACTION SUMMARY")
        print("="*50)
        print(f"Generated: {ontology['metadata']['generation_timestamp']}")
        print(f"Graph Type: {ontology['metadata']['graph_type']}")
        print(f"Files Processed: {ontology['metadata']['total_files_processed']}")
        print(f"Total Classes: {stats['total_classes']}")
        print(f"CDM Classes: {stats['cdm_classes']}")
        print(f"Object Properties: {stats['total_object_properties']}")
        print(f"Datatype Properties: {stats['total_datatype_properties']}")
        print(f"Annotation Properties: {stats['total_annotation_properties']}")
        print(f"Individuals: {stats['total_individuals']}")
        print(f"Total RDF Triples: {stats['total_triples']}")
        print(f"Namespaces: {stats['namespaces_count']}")
        print(f"Ontology Components: {stats['ontology_components_count']}")
        
        # Show component type breakdown
        if stats.get('component_types'):
            print(f"\nComponent Types:")
            for comp_type, type_info in stats['component_types'].items():
                print(f"  {comp_type}: {type_info['count']} files, {type_info['total_classes']} classes")
        
        if stats['frbr_core_classes']:
            print(f"\nFRBR Core Classes Found:")
            for frbr_class in stats['frbr_core_classes']:
                label = ontology["classes"][frbr_class].get("label", "No label")
                print(f"  - {frbr_class.split('#')[-1]}: {label}")
        
        print("\nFiles generated successfully!")
        print(f"JSON Ontology: eu_cdm_ontology.json")
        print(f"Interactive Visualization: {visualization_file}")
        print(f"Open {visualization_file} in your browser to explore the ontology!")
        print(f"\nTo add more RDF files: Place them in {parser.ontology_dir} and re-run this script")
        print(f"The script will automatically detect and process all *.rdf files in the directory")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 