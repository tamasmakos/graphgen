## EU Common Data Model (CDM) - Detailed Ontology Structure Analysis

The EU Common Data Model (CDM) is indeed a comprehensive ontological framework that encompasses multiple interconnected ontologies and schema components rather than a single monolithic ontology[1][2]. Here's a detailed breakdown of each ontological structure and schema within the CDM:

### Core FRBR-Based Foundation

**FRBR Model Implementation**

The CDM is fundamentally built upon the Functional Requirements for Bibliographic Records (FRBR) model[3][4], which provides four hierarchical abstraction levels:

- **Work (W)**: The distinct intellectual or artistic creation - the most abstract level representing conceptual content[5][6]
- **Expression (E)**: The intellectual or artistic realization of a work in a particular form, including language variations[4][3] 
- **Manifestation (M)**: The physical embodiment of an expression of a work (physical or digital format)[3][4]
- **Item (I)**: A single exemplar or copy of a manifestation[3][6]

This WEMI hierarchy allows the CDM to systematically describe multilingual and multiformat publications throughout their lifecycle[3][2].

### Document Type Specializations

**Legal Domain Classes**

The CDM describes more than 200 different types of documents across the EU institutional landscape[1]. These include specialized subclasses for:

- **Official Journal documents**: Legal acts, notices, and announcements published in the EU's Official Journal[1][3]
- **Case law**: Court decisions, judgments, and legal precedents[1][2] 
- **Consolidated legislation**: Compiled and updated versions of legal acts[1][2]
- **Preparatory acts**: Legislative proposals and draft documents[1][2]

**General Publications Classes**

Beyond legal documents, the CDM covers general publications such as[1][2]:
- Reports and studies
- Periodical serials
- Books and brochures
- Catalogued publications with specific numbering systems

### Agent Hierarchy

**Institutional and Personal Agents**

The CDM includes comprehensive agent classifications representing entities involved in document creation and management[3]:

- **Institutional agents**: EU institutions, bodies, and agencies
- **Personal agents**: Individual authors, editors, and contributors
- **Corporate agents**: Organizations and entities outside EU institutions
- **Specialized agent roles**: Publishers, translators, reviewers, and other contributors

The agent hierarchy uses **authority tables (NALs)** to ensure consistent identification and avoid linguistic or orthographic variations[1][7].

### Event and Procedural Hierarchies

**Dossier-Event Structure**

The CDM models institutional workflows through hierarchical event structures[3]:

- **Dossiers**: Cover the Work role in legislative procedures and may contain multiple events
- **Events**: Cover the Expression role, representing specific procedural steps or temporal occurrences
- **Top-level events**: Standalone events that can exist independently of dossiers[3][8]

**Legislative Procedure Classifications**

Specialized event types include[3]:
- Inter-institutional legislative procedures
- Internal institutional processes
- Publication and dissemination events
- Amendment and revision procedures

### Authority Tables and Controlled Vocabularies

**Named Authority Lists (NALs)**

The CDM integrates extensive controlled vocabulary systems[1][7][9]:

- **Geographic authorities**: Countries, regions, and administrative territorial units[7]
- **Linguistic authorities**: Language codes and multilingual labels[7][9]
- **Institutional authorities**: Standardized names for EU entities[7]
- **Subject authorities**: Thematic and topical classifications[7]
- **Technical authorities**: File formats, access rights, and other technical metadata[7]

These NALs use resource IRIs following the pattern `http://publications.europa.eu/resource/authority/[authority-name]`[1].

### Property Structure and Relationships

**Object Properties (~1000 Relations)**

The CDM contains approximately 1000 object properties that create semantic relationships between entities[1]. These include:

- **Hierarchical relationships**: Connecting Works to Expressions, Expressions to Manifestations, etc.[3]
- **Procedural relationships**: Linking documents to their creation processes and events[1]
- **Agent relationships**: Connecting documents to their creators, publishers, and other responsible parties[1]
- **Temporal relationships**: Establishing chronological connections between related documents[1]
- **Derivative relationships**: Linking original works to translations, amendments, and other derived versions[1]

**Data Properties (~900 Properties)**

The CDM includes approximately 900 data properties that define metadata constraints and data types[1]:

- **Temporal properties**: Dates, durations, and time-related metadata with specific format requirements (e.g., xsd:gYear)[1]
- **Identifier properties**: Various numbering systems, catalog numbers, and reference codes[1]
- **Descriptive properties**: Titles, descriptions, and textual metadata in multiple languages[1]
- **Technical properties**: Format specifications, file sizes, and technical characteristics[1]
- **Classification properties**: Subject codes, thematic classifications, and categorical metadata[1]

### Cardinality and Constraint Rules

**Ontological Restrictions**

The CDM implements sophisticated cardinality constraints that ensure data integrity[1]:

- **Minimum cardinality**: Ensures required relationships exist (e.g., every Item must belong to at least 1 Manifestation)[1]
- **Exact cardinality**: Enforces precise relationship counts (e.g., every Manifestation must relate to exactly 1 Expression)[1]
- **Data type constraints**: Ensures metadata values conform to specified formats and standards[1]

### Integration with External Standards

**Interoperability Frameworks**

The CDM aligns with multiple external ontological and metadata standards[1]:

- **SKOS (Simple Knowledge Organization System)**: For controlled vocabulary management[1]
- **OWL (Web Ontology Language)**: As the primary ontological framework[1][2]
- **RDFS (RDF Schema)**: For basic semantic relationships[1]
- **ELI (European Legislation Identifier)**: For legal document identification and linking[1]
- **Dublin Core**: For basic bibliographic metadata[10]

### Multilingual and Multiformat Support

**Language and Format Handling**

The CDM's FRBR foundation enables sophisticated handling of[3][4]:

- **Multilingual expressions**: Same work content in different EU languages
- **Format variations**: Different technical manifestations (PDF, XML, HTML, etc.)
- **Version management**: Temporal evolution of documents through amendments and updates
- **Cross-reference systems**: Linking related documents across languages and formats

This comprehensive ontological structure makes the CDM a sophisticated semantic framework capable of describing, organizing, and linking the vast documentary output of EU institutions while maintaining semantic interoperability across multiple systems, languages, and use cases[1][2][3].

[1] https://pro.europeana.eu/page/edm-documentation
[2] https://dssc.eu/space/BVE/357075098/Data+Models
[3] https://op.europa.eu/en/web/eu-vocabularies/authority-tables
[4] https://fib-dm.com/ontology-class-and-data-model-entity-hierarchy/
[5] https://www.cde.ual.es/wp-content/uploads/2023/10/cellar-OA0523299ENN.pdf
[6] https://op.europa.eu/hu/web/eu-vocabularies/cdm
[7] https://journals.ala.org/lrts/article/download/5444/6681
[8] https://terminology.hl7.org/5.5.0/artifacts.html
[9] https://dataeuropa.gitlab.io/data-provider-manual/pdf/documentation_data-europa-eu_V1.2.pdf
[10] https://publications.europa.eu/resource/ontology/cdm
[11] https://op.europa.eu/en/web/eu-vocabularies/dataset/-/resource?uri=http%3A%2F%2Fpublications.europa.eu%2Fresource%2Fdataset%2Fcdm
[12] https://op.europa.eu/en/web/eu-vocabularies/cdm
[13] https://fib-dm.com/ontology-object-property-data-model-associative-entities/
[14] https://op.europa.eu/en/web/eu-vocabularies/model/-/resource/dataset/cdm
[15] https://protege.stanford.edu/publications/ontology_development/ontology101.pdf
[16] https://www.dfki.de/~klusch/i2s/CDM-Core_v.2.0.1.pdf
[17] https://dssc.eu/space/bv15e/766067670
[18] https://www.linkedin.com/pulse/ontology-class-data-model-entity-hierarchy-same-jurgen-ziemer
[19] https://www.nature.com/articles/s41597-025-04558-z
[20] https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.282
[21] https://op.europa.eu/en/web/eu-vocabularies/ontologies
[22] https://www.utwente.nl/en/eemcs/fois2024/resources/papers/el-ghosh-et-al-towards-semantic-interoperability-among-heterogeneous-cancer-data-models-using-a-layered-modular-hyper-ontology.pdf
[23] https://learn.microsoft.com/en-us/common-data-model/
[24] https://dev.to/alexmercedcoder/the-role-of-ontologies-in-data-management-2goo
[25] https://cancerimage.eu/wp-content/uploads/2025/05/D5.2-EUCAIM-CDM-and-Hyper-Ontology.pdf
[26] https://www.sciencedirect.com/topics/computer-science/ontological-model
[27] https://lynx-project.eu/data2/reference-ontologies
[28] https://www.w3.org/TR/owl2-primer/
[29] https://eucdm.softdev.eu.com
[30] https://journal.code4lib.org/articles/16491
[31] https://op.europa.eu/documents/d/cellar/cellar_ml_dataset_guide
[32] https://en.wikipedia.org/wiki/Common_data_model
[33] https://op.europa.eu/en/web/eu-vocabularies/semantic-knowledge-base/-/knowledge_base/about-the-use-of-authority-tables
[34] https://www.linkedin.com/pulse/ontology-object-properties-data-model-associative-entities-ziemer-1f
[35] https://asistdl.onlinelibrary.wiley.com/doi/10.1002/bult.2007.1720330604
[36] https://www.ifla.org/files/assets/cataloguing/frbr/frbroo_v2.2.pdf
[37] https://zenodo.org/record/6542791/files/CELLAR_TUTORIAL_KGC_2022.pdf
[38] https://patents.google.com/patent/US20220012426A1/en
[39] https://cdm.unfccc.int/Reference/Notes/info_note02.pdf
[40] https://www.egmontinstitute.be/app/uploads/2021/06/CSDP-HANDBOO-4th-edition.pdf
[41] https://learn.microsoft.com/en-us/common-data-model/creating-schemas
[42] https://op.europa.eu/en/web/cellar/cellar-data
[43] https://op.europa.eu/fr/web/eu-vocabularies/ontologies
[44] https://www.wcoomd.org/-/media/wco/public/global/pdf/topics/wto-atf/dev/eu-launches-a-new-customs-data-model-based-on-wco-standards-wco-news-october-2015.pdf?la=en
[45] https://pmc.ncbi.nlm.nih.gov/articles/PMC10039164/
[46] https://patents.google.com/patent/US20160179982A1/en
[47] https://www.isda.org/category/infrastructure/common-domain-model/
[48] https://reposit.haw-hamburg.de/bitstream/20.500.12738/12594.2/3/CarusJasminMA_geschw%C3%A4rzt.pdf
[49] https://www.etsi.org/deliver/etsi_gs/CDM/001_099/004/01.00.00_60/gs_CDM004v010000p.pdf
[50] https://www.scribd.com/document/748853271/data-quality-framework-eu-medicines-regulation-en
[51] https://pro.europeana.eu/files/Europeana_Professional/Projects/Project_list/EuropeanaLibraries/Deliverables/D5.1%20%20Alignment%20of%20library%20metadata.pdf
[52] https://www.dublincore.org/blog/2024/announcing-openwemi/
[53] https://www.loc.gov/catdir/cpso/frbreng.pdf
[54] https://www.cde.ual.es/wp-content/uploads/2021/10/OAAI21001ENN.en_.pdf
[55] https://ontocommons.eu
[56] https://www.ifla.org/files/assets/cataloguing/isbd/OtherDocumentation/resource-wemi.pdf
[57] https://cancerimage.eu/wp-content/uploads/2025/05/D5.4_Data_processing_with_supplementary_m.pdf
[58] https://symposium.earsel.org/39th-symposium-Salzburg/wp-content/uploads/2019/07/EARSeL-2019-Book-of-Abstracts-Print.pdf
[59] https://eucdm.softdev.eu.com/Archive/EUCDM%206.0/EN/introduction.htm