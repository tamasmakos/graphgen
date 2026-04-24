

# **From Discourse to Structure: A Graph-Based Approach to Topic Modeling in Political Debate**

## **Abstract**

Traditional topic modeling techniques, such as Latent Dirichlet Allocation (LDA), have long served as the standard for thematic analysis of large text corpora. However, their reliance on "bag-of-words" assumptions, which disregard syntax and relational context, limits their ability to capture the nuanced structure of complex discourse. This thesis explores a graph-based paradigm for topic modeling that leverages Large Language Models (LLMs) to construct and analyze a knowledge graph from textual data. The central research questions investigate whether thematic communities identified within this graph can be considered valid "topics" from a linguistic and philosophical standpoint, and whether this structural approach offers a complementary alternative to probabilistic models in terms of interpretability.

The methodology is applied to the verbatim reports of the 'This is Europe' European Parliamentary debate series (2022-2024). The process involves a rigorous pipeline starting with **ontology injection**, where domain-specific class definitions guide the extraction process. An LLM-powered system then extracts entities, relationships, and claims to construct a Knowledge Graph. This approach is particularly well-suited to LLM-based extraction for several reasons: political discourse contains implicit relational structures (speaker argumentation, policy positions, country alliances) that resist simple co-occurrence statistics; the ontological guidance allows for domain-appropriate entity typing beyond generic NER; and the iterative nature of debate requires relational rather than distributional semantics to capture nuance. To refine the structural quality of this graph, **Node2Vec embeddings** are generated to capture topological similarities, which are then used to re-weight edges. Finally, the Leiden community detection algorithm partitions this weighted graph into densely connected, thematically coherent communities.

Recognizing that discourse topics exist at multiple granularities—a limitation noted in Section 1.1—the Leiden algorithm is applied hierarchically, first identifying broad thematic clusters (macro-topics) and then detecting sub-communities within each (sub-topics). Each resulting community is then processed through an **LLM-based summarization pipeline**: the system prompts a language model to analyze the subgraph induced by each community—its entities, relations, and connecting claims—to produce a coherent topic label and descriptive summary. This hierarchical approach addresses the discourse topic hierarchy (Reinhart, 1981) that simpler flat clustering cannot capture, while the LLM summarization provides interpretable labels grounded in the actual relational structure of the extracted knowledge graph rather than in post-hoc keyword analysis.

A key finding of this research is the validation of the extraction methodology through the emergence of **scale-free properties** in the generated graph, a hallmark of organic complex networks. Furthermore, the application of Node2Vec-based edge weighting was shown to significantly enhance the definition of these communities, improving the Leiden modularity score from a baseline of ~0.74 to ~0.82 in later iterations. This structural enhancement allows the model to reconcile a high average semantic similarity (0.57) among topic summaries with the clear identification of distinct, separable thematic clusters.

The thesis concludes that defining topics as structurally coherent communities within a knowledge graph represents a viable complementary approach to probabilistic methods. This exploration offers an alternative perspective on topic modeling—one that prioritizes interpretability and explicit structure—while acknowledging that graph-based methods are not necessarily superior to simpler approaches (Galke & Scherp, 2022).

---

## **Chapter 1: Deconstructing the 'Topic': From Philosophy to Computation**

### **1.1 The Philosophical and Linguistic Foundations of a 'Topic'**

Before a computational model can claim to identify "topics," it is imperative to establish a rigorous, non-computational understanding of what a topic is. The term is often used imprecisely in data science, treated as a mere label for a cluster of co-occurring words. However, its roots in linguistics and the philosophy of language reveal a far more structured and profound concept, one that is central to the organization of information and the construction of meaning in human communication.

From a linguistic perspective, the foundational distinction is between the *topic* (or *theme*) and the *comment* (or *rheme*) (Halliday, 1985; Firbas, 1992). The topic is what a sentence or clause is *about*; it is the entity or concept that anchors the discourse, providing the subject of the predication. The comment is what is being said *about* that topic; it is the new information, the assertion, or the description being provided. This division, known as information structure, posits that communication is not an unstructured stream of words but a deliberate organization of information into old (the topic, which connects to the existing discourse) and new (the comment) (Halliday, 1985). This fundamental structure implies that a topic is not a standalone artifact but exists in relation to the propositions made about it.

The philosophy of language deepens this understanding through the concept of "aboutness" (Reinhart, 1981, 1982). Reinhart argues that "aboutness" is the defining characteristic of a topic, moving beyond purely grammatical definitions of a subject to a pragmatic one based on communicative intent. The topic is the entity that the speaker directs the hearer's attention to, about which they intend to convey information. This philosophical framing is critical because it sets a higher bar for topic modeling: the goal is not merely to find clusters of words but to identify the primary subjects of "aboutness" that structure a body of text.

Furthermore, a crucial distinction must be made between a *sentence topic* and a *discourse topic* (Reinhart, 1981). A sentence topic is the constituent that a specific sentence is about, whereas a discourse topic is what an entire conversation or text is about. For example, in a debate about European energy policy, the discourse topic is "European Energy Policy." Within this discourse, individual sentences may have sentence topics like "natural gas reserves," "renewable energy investment," or "Russian dependency." Traditional computational models often struggle to separate these levels, conflating high-frequency terms associated with the overarching discourse topic with the more specific subjects of individual arguments. An effective topic model must be capable of resolving this hierarchy.

These concepts can be synthesized through the lens of Foucauldian discourse theory. Foucault defines a discourse not as a simple collection of statements, but as a "system of thoughts composed of ideas, attitudes, courses of action, beliefs, and practices that systematically construct the subjects and the worlds of which they speak" (Foucault, 1972). In this view, a discourse creates its own objects and concepts through the regulated interplay of statements. A "topic," therefore, is not just a word or a concept but a node within this system, defined by its relationships to other nodes. It is a representation of one of these constructed subjects. This provides a powerful theoretical framework: a true topic model should aim to uncover these systems of thought, revealing not just *what* is being discussed, but *how* the subjects of the discourse are constructed through the relationships between different ideas and entities.

### **1.2 Computational Approaches to Topic Modeling**

The abstract, theoretical concept of a topic must be operationalized to be computationally tractable. While the dominant approach for the last two decades has been the probabilistic paradigm, exemplified by Latent Dirichlet Allocation (LDA) (Blei, Ng, & Jordan, 2003), alternative structural approaches have gained interest in recent years.

Probabilistic models like LDA define a topic as a probability distribution over a vocabulary, inferred from the co-occurrence of words within documents. This relies on the "bag-of-words" assumption, which treats text as an unordered collection of terms, disregarding the rich relational structure of language. Despite its limitations, LDA remains a strong baseline—recent work by Galke and Scherp (2022) demonstrated that simple bag-of-words approaches can often outperform more complex graph-based methods in text classification tasks, raising important questions about the practical necessity of graph constructions.

In recent years, graph-based approaches have emerged as an alternative paradigm. TextGCN (Yao, Mao, & Luo, 2019) constructed heterogeneous graphs linking words and documents, demonstrating that graph convolutional networks could capture document-level topics. The Graph Neural Topic Model (GNTM) (Xie et al., 2021) integrated graph-based regularization into neural topic models. These approaches define topics not as statistical abstractions but as components of an explicit knowledge structure.

This thesis explores a **structural paradigm** that defines a topic as a community within a knowledge graph. The formal definition is as follows:

*A topic is a densely interconnected community of entities (nodes) and their relationships (edges) within a knowledge graph, which is algorithmically identified through community detection and can be articulated through a natural language summary.*

This approach moves the unit of analysis from words to entities—real-world concepts, people, places, and organizations—and their explicit, labeled relationships. It represents one possible operationalization of the theoretical concept of a "system of thoughts," where meaning is derived from the structure of connections. It is important to note that this is not claimed as a replacement for probabilistic methods but rather as an alternative exploration—one that prioritizes interpretability and explicit structure over computational efficiency.

---

## **Chapter 2: A Graph-Based Epistemology: Constructing Knowledge from Discourse**

### **2.1 The Corpus: The 'This is Europe' Parliamentary Debates**

The corpus selected for this study consists of the verbatim reports of the 'This is Europe' debate series held in the European Parliament. Between April 2022 and March 2024, 13 EU Heads of State or Government addressed the Parliament to present their visions for the future of the European Union.

The context of these debates is critical. They commenced shortly after Russia's full-scale invasion of Ukraine, a geopolitical event that reshaped European priorities overnight. They also ran concurrently with the conclusion of the Conference on the Future of Europe (CoFoE), a citizen-led initiative to guide EU policy. This backdrop ensures that while each leader brought a national perspective, the speeches were anchored in a shared set of urgent and overarching challenges.

An analysis of the debates conducted by the European Parliamentary Research Service (EPRS) identifies six recurring themes: (i) the value of EU membership, (ii) defending EU values, (iii) the main challenges facing the EU, (iv) delivering for EU citizens, (v) next steps in EU integration, and (vi) the importance of EU unity (European Parliamentary Research Service, 2024). The EPRS report further breaks down the specific topics addressed, noting that Ukraine, enlargement, and energy were the most frequently and extensively discussed subjects across all speeches.

This corpus is particularly well-suited for this thesis for two reasons. First, its thematic cohesion provides a stringent test for any topic model. The high degree of thematic overlap and shared vocabulary makes it difficult for purely statistical models to disentangle nuanced sub-topics. The observed average cosine similarity of 0.57 between the generated topic summaries is a direct quantitative reflection of this shared context, confirming that, at a high level, all debates are indeed "about Europe." Second, the detailed EPRS briefing serves as an expert-curated "ground truth." It provides an independent, qualitative baseline against which the computationally derived topics can be validated, allowing for a robust assessment of the model's accuracy and real-world relevance.

### **2.2 From Text to Graph: An LLM-Powered Pipeline**

The transformation of unstructured text from the parliamentary debates into a structured knowledge graph is the foundational step of the methodology. This process leverages the advanced natural language understanding capabilities of LLMs, guided by a predefined domain ontology, to create a rich, relational representation of the discourse. The pipeline consists of several sequential stages.

1.  **Ontology Injection & Schema Definition:** Before any text is processed, the pipeline ingests a domain-specific ontology (defined in OWL/RDF formats). A dedicated `OntologyLabelExtractor` parses these files to extract a set of valid class labels (e.g., *Person*, *Policy*, *Organization*, *GeopoliticalEntity*). These labels serve as a strict schema that constrains the subsequent extraction steps. This "injection" of structured knowledge ensures that the LLM does not hallucinate arbitrary types but maps concepts to a standardized vocabulary, essential for downstream graph analysis.

2.  **Text Chunking:** The raw verbatim transcripts are first segmented into smaller, manageable text chunks. The size of these chunks is a critical design parameter. Smaller chunks, such as 600 tokens, tend to yield a higher density of extracted entity references, improving the granularity of the resulting graph. However, this comes at the cost of increased LLM API calls and processing time. Conversely, longer chunks are more cost-effective but may risk losing recall of information mentioned early in the chunk.

3.  **Constrained Entity and Relationship Extraction:** Each text chunk is processed by a hybrid pipeline. First, a Named Entity Recognition (NER) model (GLiNER) scans the text to identify surface-level entities, constrained by the injected ontology labels. These detected entities serve as high-confidence "hints." Next, an LLM is prompted to perform the final structured extraction. Crucially, the LLM is provided with the *allowed nodes* list from the ontology and the *pre-identified entities* from the NER step. It extracts relationships between pairs of entities, outputting triplets (Source, Target, Relation). By strictly adhering to the ontology, this step ensures that the resulting graph is semantically consistent.

4.  **Claim Extraction:** Beyond simple entity-relationship pairs, the LLM is also tasked with extracting important factual statements, or "claims," associated with the entities. These claims capture specific details like dates, events, quantitative data, and direct quotes. They are stored as attributes (covariates) of the entity nodes in the graph, enriching the model with specific, verifiable information from the source text.

5.  **Graph Assembly:** Finally, the extracted elements from all chunks are aggregated to form a single, unified knowledge graph. Entities become the nodes (V) of the graph, and the aggregated relationships form the edges (E). Duplicate entities are merged, and claims are stored as node attributes. This assembled graph serves as the structured, machine-readable representation of the entire corpus.

### **2.3 The Emergence of Structure: Scale-Free Properties in Extracted Knowledge**

A significant finding that validates the integrity of this graph construction process is the topological nature of the resulting network. Analysis of the graph's degree distribution—the probability P(k) that a randomly chosen node has k connections—reveals that it follows a power law, where P(k)∼k−γ. This is the defining characteristic of a scale-free network (Barabási & Albert, 1999; Newman, 2005).

The study of scale-free networks was pioneered by Albert-László Barabási and Réka Albert (1999), who discovered that this topology is not a mathematical curiosity but a ubiquitous feature of real-world complex systems, including the World Wide Web, social networks, and biological protein-interaction networks. The emergence of this structure is explained by two simple, yet powerful, underlying mechanisms:

**growth** and **preferential attachment** (Barabási & Albert, 1999).

* **Growth:** Real networks are rarely static; they expand over time through the addition of new nodes.  
* **Preferential Attachment:** New nodes are more likely to connect to existing nodes that are already highly connected. This "rich-get-richer" phenomenon leads to the formation of a few highly connected "hubs" that dominate the network's structure.

The appearance of a scale-free topology in the knowledge graph extracted from the parliamentary debates is not a random artifact. It is a profound reflection of the fundamental dynamics of discourse itself. A series of political debates is a growing system: each speech adds new concepts and arguments to the existing network of ideas (growth). When speakers contribute, they do not introduce concepts in a vacuum. To be relevant and persuasive, they must connect their arguments to the central, most salient themes of the ongoing discussion—the established hubs like 'Ukraine', 'energy dependency', or 'EU values' (preferential attachment) (Barabási & Albert, 1999).

Therefore, the very process of building a coherent, multi-speaker discourse is a network-generating mechanism that naturally follows the principles of scale-free models. The appearance of this exact topology in our generated graph acts as a **structural validation of the extraction pipeline**. It confirms that the ontology-guided extraction did not impose an artificial or rigid structure, but rather successfully captured the organic, self-organizing nature of the political debate. A graph that appeared random (Poisson degree distribution) or regular (lattice-like) would indicate a failure of the extraction process to identify the true semantic hubs. The scale-free property confirms that the "hubs" in our graph (e.g., *Ukraine*, *Energy Crisis*) correspond to the actual semantic anchors of the real-world discourse.

### **2.4 Ensuring Coherence: Entity Resolution with Knowledge Graph Embeddings**

The integrity of the graph's structure, particularly the accurate identification of hubs, depends on a crucial data processing step: Entity Resolution (ER). Raw LLM output can be inconsistent, creating multiple nodes for the same real-world entity (e.g., "Olaf Scholz," "the German Chancellor," "Mr. Scholz"). ER is the process of identifying and merging these duplicate nodes to ensure that each unique entity is represented by a single node in the graph.

A powerful technique for performing ER on knowledge graphs is the use of Knowledge Graph Embeddings (KGEs). This approach maps the symbolic components of the graph—its entities and relationships—into a low-dimensional, continuous vector space. The core principle of KGE models like TransE is that the geometric relationships between vectors in this space should reflect the semantic relationships in the original graph (Bordes et al., 2013). For a given triple (head, relation, tail), the model learns vector representations such that the vector for the head plus the vector for the relation is approximately equal to the vector for the tail: h+r≈t (Bordes et al., 2013).

This embedding process captures the topological neighborhood of each entity. Entities that are connected to similar entities via similar relationships will have their vectors mapped to nearby points in the embedding space. Consequently, similarity between two entities can be efficiently calculated as the distance (e.g., cosine similarity or Euclidean distance) between their corresponding vectors. By setting a similarity threshold, likely duplicates can be automatically identified and flagged for merging. This approach is part of a broader family of embedding models designed for knowledge graph completion and analysis (Nguyen, 2020).

This process is vital for the validity of the overall analysis. Without effective ER, a central concept like 'Ukraine' might be fragmented into dozens of low-degree nodes, obscuring its true role as a major hub in the network. By consolidating these fragments into a single, high-degree node, ER ensures that the scale-free analysis is accurate and that the subsequent community detection operates on a topologically coherent and semantically meaningful graph.

### **2.5 Exhaustive Filtering Mechanisms for Topic Modeling**

The knowledge graph construction pipeline incorporates a multi-stage filtering system designed to extract meaningful abstract concepts while eliminating noise, function words, and spurious associations. This exhaust filter approach serves as a critical preprocessing step for topic modeling, ensuring that the resulting communities represent substantive thematic structures rather than artifacts of the extraction process.

**Filtering Architecture**: The filtering pipeline operates across three dimensions: speech-level filtering, grammatical filtering, and semantic filtering. Each dimension serves a distinct purpose in refining the entity and relationship set before community detection.

**Speech-Level Entity Filtering**: At the corpus aggregation stage, entities that appear fewer than twice across the entire corpus are removed unless they participate in at least one relationship. This heuristic balances two competing concerns: (i) entities mentioned only once may represent noise or idiosyncratic references that do not generalize across the debate series, and (ii) entities that participate in relationships—even if mentioned only once—may serve important structural roles as bridges between thematic clusters. The threshold of two mentions was determined empirically and represents a conservative compromise that retains potentially significant entities while filtering clear outliers.

**Grammatical Filtering**: The pipeline employs a series of grammatical rules to identify valid abstract concepts from candidate words. These rules, implemented in the `is_grammatically_valid_concept` function, include:

1. **Length Filter**: Words shorter than three characters are rejected as insufficiently meaningful.
2. **Alphabetic Filter**: Only purely alphabetic tokens are retained, excluding numeric values and special characters.
3. **Part-of-Speech Filter**: Only common nouns (NN, NNS) are accepted; proper nouns are excluded as they represent specific entities rather than abstract concepts.
4. **Morphological Filter**: Words ending in suffixes such as *-ing*, *-ed*, *-er*, *-est*, or *-ly* are rejected, as these typically indicate verb-derived forms, comparatives, or adverbs rather than abstract nouns.
5. **Lexical Validity Filter**: WordNet synset analysis ensures that the candidate word has at least one valid noun sense.
6. **Semantic Depth Filter**: Abstract concepts are identified by their position in the semantic hierarchy. Words whose WordNet hypernym paths connect to abstract root synsets (e.g., *abstract_entity.n.01*, *attribute.n.01*) with path depth greater than four are retained as abstract concepts.

**Additional Linguistic Filters**: The pipeline further excludes:

- **Pronouns and Determiners**: Words with POS tags in the set {PRP, PRP$, WP, WP$, DT, WDT, PDT, CD, MD} are filtered as function words that lack semantic content.
- **Temporal and Spatial Deictics**: Words whose semantic paths connect to *time.n.01*, *space.n.01*, *location.n.01*, or *temporal_relation.n.01* are excluded, as these represent contextual anchors rather than substantive topics.
- **Modal and Auxiliary Concepts**: Words primarily classified as verbs in WordNet are excluded, as they represent actions rather than abstract concepts.

These exhaust filters serve a dual purpose in the topic modeling pipeline. First, they reduce the dimensionality of the graph by eliminating nodes that would otherwise fragment community structure through spurious connections. Second, they ensure that the resulting communities are composed of entities with genuine semantic content, making the downstream summarization more coherent and interpretable.

### **2.6 Topic Modeling as a Use-Case: Structural versus Distributional Approaches**

The application of community detection to the filtered knowledge graph represents a structural alternative to traditional distributional topic modeling. Whereas LDA and its variants define topics as probability distributions over words—inferring latent structures from co-occurrence patterns—the structural paradigm defined in this thesis treats topics as explicit subgraphs identified through community detection. This distinction has important implications for interpretability, validation, and the philosophical foundations of what constitutes a "topic."

**Structural Topics as Relational Clusters**: In the structural paradigm, a topic is defined as a densely interconnected subgraph whose nodes share more edges with each other than with the rest of the graph. This definition operationalizes the philosophical notion of a topic as a coherent system of "aboutness" relations. The Leiden algorithm's guarantee of connected communities ensures that each identified topic forms a single, cohesive unit—no disconnected clusters that would violate the intuition of a unified theme.

**Validation Through Structural Properties**: The structural approach enables validation through network properties that are unavailable to distributional methods. The emergence of scale-free topology in the extracted graph serves as a strong form of structural validation: if the extraction pipeline has successfully captured the underlying discourse structure, the resulting network should exhibit the same statistical properties as real-world complex networks. The observed power-law degree distribution confirms that the ontology-guided extraction has produced a graph with organic structure rather than artificial clustering.

**Complementarity with Distributional Methods**: The structural approach does not necessarily supplant distributional methods but rather complements them. Where LDA infers topics from word co-occurrence patterns without explicit relational structure, the graph-based approach leverages the full richness of entity-relationship triples. The two paradigms may be fruitfully combined: distributional similarity measures could inform edge weights, while structural community detection could provide topic boundaries that constrain distributional inference.

The exhaust filtering mechanisms described above play a critical role in this use-case. By ensuring that only semantically substantive entities participate in the graph, the filters reduce the risk that community detection will identify spurious structures formed by noise entities. The resulting topics are more likely to correspond to genuine thematic divisions in the discourse rather than artifacts of the extraction process.

**A Note on Comparison with Probabilistic Methods**: A full empirical comparison with LDA would require topic coherence metrics (such as C_v, which uses normalized pointwise mutual information) rather than standard clustering metrics like silhouette scores, which are inappropriate for evaluating probability-distribution-based topic models (Mimno et al., 2011). Such a comparison is left for future work, as it would require re-extracting the corpus using distributional representations compatible with LDA's bag-of-words assumptions—a fundamentally different preprocessing pipeline than the relational knowledge graph construction used here.

---

## **Chapter 3: Uncovering Thematic Structures: Community Detection and Summarization**

### **3.1 The Leiden Algorithm: Partitioning the Graph into Thematic Communities**

Once a validated and coherent knowledge graph is constructed, the core task of identifying topics begins. In the structural paradigm, this is framed as a problem of community detection. A community in a network is defined as a set of nodes that are more densely connected to each other than they are to the rest of the network. This graph-theoretic concept maps directly onto the intuitive notion of a topic: a collection of closely related ideas and concepts.

The algorithm employed for this task is the Leiden algorithm, a state-of-the-art method for community detection (Traag, Waltman, & van Eck, 2019). The Leiden algorithm is an iterative process that aims to find a partition of the graph's nodes that maximizes a quality function known as *modularity* (Newman & Girvan, 2004; Newman, 2006). Modularity measures the difference between the density of edges within the detected communities and the density that would be expected if the edges were distributed randomly, preserving node degrees. A high modularity score indicates a strong, non-random community structure.

The Leiden algorithm improves upon its well-known predecessor, the Louvain method, in a critical way. While Louvain is effective, it can sometimes produce communities that are poorly connected or even internally disconnected. The Leiden algorithm introduces an intermediate refinement phase into its iterative process, which explicitly checks the internal connectivity of communities and may split them to resolve such issues. As a result, the Leiden algorithm **guarantees that all detected communities are well-connected subgraphs** (Traag, Waltman, & van Eck, 2019).

This guarantee is not merely a technical improvement; it provides a graph-theoretic enforcement of *thematic coherence* for the identified topics. As established in Chapter 1, a topic must be a coherent set of related concepts. In the language of graph theory, coherence is synonymous with connectivity. A disconnected community would represent a "topic" containing two or more sets of ideas with no explicit path of relationship between them, violating the fundamental definition of a single, unified topic. By ensuring that every node in a community is part of a single connected component, the Leiden algorithm algorithmically enforces this principle.

#### **3.1.1 Enhancing Structure with Node2Vec Embeddings**

While the Leiden algorithm is effective on unweighted graphs, this research postulates that the *strength* of the relationship between concepts is as important as its existence. To capture this, the pipeline integrates **Node2Vec**, an algorithmic framework for learning continuous feature representations for nodes in networks (Grover & Leskovec, 2016).

The process involves three steps:
1.  **Structural Embedding:** Node2Vec is trained on the graph to generate low-dimensional vector embeddings for every node. These embeddings capture the topological neighborhood of each node; nodes that share similar connections (homophily) or structural roles are mapped to nearby points in vector space.
2.  **Edge Reweighting:** The cosine similarity between the embeddings of connected nodes is calculated. This similarity score is then applied as a weight to the edge connecting them.
3.  **Weighted Detection:** The Leiden algorithm is run on this weighted graph. Edges connecting structurally similar nodes are treated as "stronger," encouraging the algorithm to keep them in the same community.

Experimental results from this study confirm the validity of this approach. As the size of the graph grew over iterative experiments, the **modularity score**—the primary metric for community quality—showed a marked improvement when using Node2Vec weights compared to an unweighted baseline. In later iterations (e.g., Iteration 9), the modularity improved from a baseline of **0.748** to **0.823** (+0.075). This quantitative lift indicates that the topological information captured by Node2Vec helps the community detection algorithm to carve out sharper, more distinct thematic boundaries, effectively "denoising" the graph structure.

### **3.2 Hierarchical Abstraction of Discourse**

Human discourse is often organized hierarchically. A broad theme, such as "EU's Main Challenges," can be broken down into more specific sub-themes like "The Energy Crisis," "Inflation," and "Disinformation," which themselves can be further decomposed. The Leiden algorithm naturally accommodates this nested structure.

The algorithm's process of local node moving, refinement, and aggregation can be applied recursively. After an initial partition of the graph is found, each resulting community can be treated as a new, smaller graph. The Leiden algorithm can then be run again on these subgraphs to identify finer-grained sub-communities within them. This hierarchical application produces a nested partitioning of the data, providing a multi-resolution view of the thematic landscape. This allows an analyst to explore the discourse at different levels of abstraction, from the highest-level themes that span the entire corpus down to the most specific sub-topics discussed within a particular line of argument, mirroring the natural organization of complex information.

### **3.3 Generating Topic Narratives: Structure-Aware Hierarchical Summarization**

The output of the community detection process is a set of subgraphs, where each subgraph represents a topic. While this structured representation is powerful for analysis, it is not immediately interpretable to a human user. The final stage of the methodology bridges this gap through a novel, structure-aware summarization process that explicitly leverages the graph topology to generate rich, multi-faceted community reports.

#### **3.3.1 Hierarchical Processing: From Subtopics to Topics**

Recognizing that discourse exhibits natural hierarchical organization, the summarization pipeline implements a strict bottom-up processing order. The system first identifies and processes all leaf-level communities (subtopics), generating comprehensive summaries for each. Only after subtopic summarization is complete does the pipeline proceed to higher-level communities (topics), where the summaries of constituent subtopics are explicitly provided as contextual input. This hierarchical composition ensures that topic-level summaries are genuine abstractions that synthesize insights from their component parts, mirroring how human analysts construct understanding at multiple levels of granularity.

#### **3.3.2 Structure-Aware Context Formatting**

Unlike traditional summarization approaches that operate solely on concatenated text, this methodology explicitly surfaces the graph structure to the LLM. For each community, the input context is formatted using XML markup to delineate distinct information layers:

1. **Community Structure**: An explicit enumeration of entities within the community, sorted by degree (topological prominence), accompanied by their ontological types. This provides the LLM with immediate visibility into the semantic actors involved.

2. **Relationship Network**: A structured listing of internal relationships (edges) between community members, formatted as source-relation-target triplets. This captures the specific ways in which entities interact within the thematic cluster.

3. **Sub-Community Summaries** (for hierarchical topics): Previously generated summaries of child communities, providing compositional context.

4. **Textual Evidence**: A curated selection of text chunks from the original corpus where these entities co-occur, serving as grounding evidence.

By providing this multi-modal input—combining structural metadata with textual evidence—the summarization process becomes *structure-aware*, enabling the LLM to generate summaries that reflect both the topological salience of entities and their discursive context.

#### **3.3.3 Structured Report Generation with Explicit Findings**

The summarization prompt employs a rigorous template that mandates a structured JSON output format. Rather than generating a simple descriptive paragraph, the LLM is instructed to produce a formal analytical report containing:

- **Title**: A concise, descriptive label for the community (3–10 words).
- **Executive Summary**: A comprehensive 3–5 sentence overview synthesizing the core theme and key dynamics.
- **Detailed Findings**: A structured list of 3–5 specific insights, each comprising:
  - A summary statement identifying a key pattern or observation.
  - An explanation paragraph that cites specific entities, relationships, or text evidence supporting the finding.

This format enforces analytical rigor, ensuring that summaries are not merely descriptive but provide interpretable, evidence-grounded insights. The structured findings serve as auditable claims that can be traced back to the underlying graph structure, enhancing the transparency and verifiability of the topic model.

#### **3.3.4 Prompt Engineering for Analytical Depth**

The prompt itself is carefully engineered to elicit high-quality analytical output. It employs several best practices:

- **Role Assignment**: The LLM is assigned the role of an "expert intelligence analyst specializing in graph-based pattern detection," establishing an analytical rather than merely descriptive framing.
- **Explicit Constraints**: The prompt includes explicit instructions on grounding (all findings must be supported by provided evidence), completeness (prefer synthesis over listing), and tone (professional, analytical, objective).
- **Output Format Specification**: The required JSON schema is explicitly defined, with field-by-field descriptions, ensuring consistent structure across all generated reports.

This structured prompting approach represents a significant methodological advancement over generic "summarize this text" instructions, aligning the summarization task with the graph-theoretic foundations of the overall methodology.

---

## **Chapter 4: Analysis of Graph-Derived Topics from the 'This is Europe' Debates**

### **4.1 The Global Thematic Landscape: Interpreting Quantitative Results**

The quantitative results from the refined experimental pipeline provide a nuanced and, in some respects, challenging view of the corpus's thematic structure. Analysis of the revised `iterative_experiment_results.csv` (5 iterations, 15 speeches) reveals both the strengths and inherent limitations of applying community detection to highly cohesive political discourse.

#### 4.1.1 Structural Coherence vs. Semantic Distinctiveness: A Critical Dichotomy

The experimental results expose a fundamental tension between two dimensions of topic quality: *structural coherence* (as measured by modularity) and *semantic distinctiveness* (as measured by silhouette scores and topic separation metrics). This dichotomy demands careful interpretation.

**Structural Integrity Remains Strong**: Across all five iterations, the Leiden algorithm consistently identified communities with high modularity scores, ranging from **0.774** (Iteration 1) to **0.780** (Iteration 5), with peaks reaching **0.827** (Iteration 2). These values significantly exceed random graph baselines (0.716–0.756), confirming that the detected communities represent genuine structural partitions of the knowledge graph. The community structure is *graph-theoretically valid*—entities within communities are indeed more densely interconnected than would be expected by chance.

**Semantic Overlap Persists and Intensifies**: However, the semantic distinctiveness metrics paint a markedly different picture. The **topic separation** score—a measure of the average pairwise distance between topic summary embeddings—degraded from **0.620** (Iteration 1) to **0.481** (Iteration 5), while the corresponding **topic overlap** increased from **0.380** to **0.519**. More concerning are the **silhouette scores**, which remained consistently negative throughout: entity-level silhouette ranges from **0.077** to **0.003**, while subcommunity-level silhouette deteriorated from **-0.091** to **-0.184**.

Silhouette coefficients quantify how well-separated clusters are,with values near +1 indicating clear separation, 0 indicating overlapping clusters, and negative values indicating potential misclassification (Rousseeuw, 1987). The persistent negative silhouette scores indicate that entities, when embedded in semantic vector space, are often closer to entities in *other* communities than to entities within their *own* community. This reveals the core challenge: the parliamentary debates exhibit *structural modularity* (entities co-occur in distinct patterns) but *semantic homogeneity* (the entities themselves are discussed using highly similar vocabulary and argumentative frames).

#### 4.1.2 The Degradation Pattern: Accumulation vs. Differentiation

A striking empirical finding is that topic distinctiveness systematically *degrades* as the corpus grows. Between Iterations 1 and 5:

- Entity silhouette: **0.077 → 0.003** (-96%)
- Subcommunity silhouette: **-0.091 → -0.184** (-102% worsening)
- Topic overlap: **0.380 → 0.519** (+37%)

This pattern suggests that as additional speeches are incorporated, they introduce entities and relationships that bridge previously distinct thematic clusters, creating a progressively more interconnected discourse network. This observation aligns with sociolinguistic theories of discourse convergence in deliberative settings (Giles, 2016): speakers in parliamentary debates do not introduce wholly novel themes but rather reinforce, reframe, and connect existing ones. The accumulation of such cross-cutting references increases structural *density* (beneficial for modularity) but reduces semantic *differentiation* (detrimental for silhouette scores).

#### 4.1.3 Interpreting "Overlap" in Political Discourse: Consensus as a Confound

The negative silhouette scores must be interpreted in the specific context of political discourse analysis. Standard clustering evaluation assumes that well-defined clusters should exhibit clear boundaries in feature space. However, political debates—particularly those conducted within a parliamentary institution around a shared set of crises—do not conform to this assumption. The "This is Europe" series occurred during an unprecedented confluence of challenges: the Russian invasion of Ukraine, the energy crisis, post-pandemic economic recovery, and migration pressures. These issues are not independent topics but fundamentally interconnected dimensions of European policy.

Thematic overlap in this corpus is not a modeling failure but a substantive property of the data. When 14 EU leaders discuss energy security, nearly all reference Ukraine, NATO, and strategic autonomy. When discussing migration, they invoke energy prices, economic stability, and geopolitical threats. The semantic embeddings—which capture distributional semantics of language use—accordingly project these topics into overlapping regions of vector space. The negative silhouette scores thus quantify the degree of *thematic consilience* in European political discourse during this period.

Crucially, the model's high modularity scores demonstrate that despite this semantic overlap, the *structural patterns* of entity co-occurrence remain distinct. This suggests that while speakers use similar vocabularies, they construct different relational narratives. The graph-based approach captures this structural variance even when semantic variance is minimal—a capability that purely distributional models (which rely on word co-occurrence alone) would lack.

#### 4.1.4 Qualitative Analysis of Generated Community Reports

The hierarchical, structure-aware summarization pipeline produces rich analytical reports that reveal both the methodology's strengths and the inherent challenges of the corpus. Examination of the generated summaries from Iteration 5 demonstrates that the pipeline successfully operationalizes the graph structure into interpretable narratives, but also exposes the fundamental thematic convergence of the debates.

**Analytical Depth and Structural Grounding**: Each generated community report adheres to the mandated structure, producing titles, executive summaries, and detailed findings. For example, TOPIC_0 is titled "European Union Dynamics and Challenges," and includes five granular findings: "Autonomy and Food Security," "Migration Policies and Border Control," "Energy Security and Climate Resilience," "Socioeconomic Stability and Integration," and "Geopolitical Challenges and Autonomy." Each finding is supported by explicit references to entities (e.g., "European Union, NATO, individual member states") and relationships drawn from the graph.

The hierarchical composition is evident when comparing subtopic and topic summaries. SUBTOPIC_0_0 focuses narrowly on "European Union Autonomy and Food Security," identifying 11 entities with specific discussions on defense spending and unity. Its parent, TOPIC_0, synthesizes this alongside summaries from 15 other subtopics into a broader narrative encompassing 69 entities and 45 relationships. This demonstrates successful abstraction: the topic-level summary is not simply a repetition of subtopic content but a genuine compositional synthesis.

**The Convergence Problem: Semantic Homogeneity Across Topics**: Despite this structural sophistication, a critical examination reveals pervasive thematic redundancy. Of the 14 detected topics in Iteration 5, 12 explicitly mention "energy," 11 mention "migration," and 10 mention "European unity" or "security" in their titles or summaries. Representative examples include:

- TOPIC_1: "EU's Resilience and Unity in the Face of External Threats" (energy, migration, Russian aggression)
- TOPIC_2: "European Community Dynamics: Unity, Autonomy, and Sustainability" (energy crisis, migration, Ukraine)
- TOPIC_3: "European Unity and Security in the Face of External Threats" (Russian aggression, migration, energy)
- TOPIC_4: "European Politics and Integration Community" (strategic autonomy, energy crisis, migration)
- TOPIC_5: "EU Energy Security and Self-Sufficiency Community" (renewable energy, pandemic, geopolitical threats)
- TOPIC_6: "EU Dynamics and Energy Transition" (energy, security, migration)

This pattern reveals the limitations of community detection when applied to a corpus with such profound thematic consilience. While the Leiden algorithm successfully identifies structurally distinct subgraphs (hence high modularity), these subgraphs describe semantically overlapping themes (hence negative silhouettes). The graph structure differentiates *perspectives* and *emphases* rather than wholly distinct topics.

**The Methodological Contribution**: Despite the thematic homogeneity, the structured summarization pipeline fulfills its primary objective: it accurately *describes* the communities as they exist in the graph. The redundancy is not a failure of the summarization method but a faithful representation of a highly interconnected discourse network. This honest reflection of the data's properties is itself valuable, as it prevents false claims of topic distinctiveness where none exists. The method's contribution lies in its transparency and auditability—every claim in a summary can be traced to specific entities and relationships—rather than in its ability to artificially impose separability on an inherently cohesive corpus.

### **4.2 Qualitative Validation Against Ground Truth**

To validate the real-world relevance of the computationally derived topics, a qualitative comparison was performed against the themes identified in the independent analysis by Drachenberg & Bącal for the European Parliamentary Research Service (EPRS) (2024). The LLM-generated summaries and the key hub entities (nodes with the highest degree) for each major community were mapped to the corresponding themes described in the EPRS briefing document. The results of this mapping demonstrate a strong alignment between the model's output and the expert analysis, confirming that the method identifies human-salient and politically relevant topics.

| Topic ID | Generated Title & Summary (Excerpt) | Key Hub Entities | Analysis of Match |
| :---- | :---- | :---- | :---- |
| **Topic 0** | **"Europe's Unity and Security in the Face of Global Challenges"**<br>Focuses on the Russian aggression against Ukraine, emphasizing the need for EU unity, sanctions against Russia, and comprehensive support. | *Ukraine, Russia, Sanctions, EU Unity* | **Direct Match**: Aligns perfectly with the EPRS finding that Ukraine was the #1 topic, mentioned by 100% of speakers and accounting for 14% of total attention. |
| **Topic 40** | **"EU Motor Insurance Sector: Regulatory Challenges and National Identity"**<br>Discusses specific regulatory issues within the motor insurance market, likely highlighting national frictions. | *Motor Insurance, Regulation, Member States* | **Niche Discovery**: This aligns with the "National policies" category identified by EPRS, which accounted for 7% of attention and was driven by specific speakers (e.g., Mitsotakis). |
| **Topic 42** | **"Bulgaria's Fight Against Corruption: A 13-Year Struggle for Justice"**<br>A highly specific community focusing on the rule of law and corruption challenges within Bulgaria. | *Bulgaria, Corruption, Justice Reform, Rule of Law* | **High Granularity**: Shows the model can isolate national-level concerns embedded within the broader European debate, separating "Bulgarian Corruption" from general "Rule of Law." |
| **Topic 20** | **"Strengthening European Food Security and Sustainable Energy Practices"**<br>Links the energy crisis to broader food security concerns, reflecting the compound nature of the 2022 crisis. | *Food Security, Energy, Sustainability* | **Thematic Synthesis**: Correctly identifies the intersection of two major EPRS themes (Energy and Economic Challenges). |
| **Topic 3** | **"Europe's Unity and Resilience in the Face of Global Challenges"**<br>A recursive theme similar to Topic 0 but likely with a different rhetorical focus (Resilience vs Security). | *Resilience, Global Challenges, Unity* | **Rhetorical Variation**: Captures the subtle difference in framing between "Security" (Hard Power) and "Resilience" (Systemic Strength). |

This table provides concrete, qualitative proof that the graph-based model is not merely identifying statistically interesting patterns but is successfully extracting the same real-world, substantive topics that were independently identified by political analysts. Crucially, it distinguishes between the **Core Discourse** (Topic 0, Topic 3) which dominates the similarity matrix, and the **Specific Policy Debates** (Topic 40, Topic 42) which reside in the "long tail" of the distribution. This ability to maintain high coherence for niche topics while acknowledging the broad unity of the main discourse is a distinct advantage of the structural approach.

### 4.3 Similarity Analysis and Distribution

The analysis of the `topic_similarity_matrix` reveals a similarity range from **0.17** to **1.0**. The lower bound of 0.17 is significant; it proves that despite the "about Europe" blanket, there are topics that the model considers mathematically distinct. For instance, the discourse on *Motor Insurance Regulation* (Topic 40) shares very little structural or semantic overlap with *Europe's Geopolitical Imperative* (Topic 6).

If the model were failing (i.e., everything is just "Europe"), we would expect the minimum similarity to be much higher (e.g., >0.50). The presence of these low-similarity pairs confirms that the **Node2Vec-enhanced Leiden algorithm** is successfully "cutting" the graph at the joints of the discourse, separating technical policy discussions from high-level political rhetoric.

---



## **Chapter 5: Conclusion**

### **5.1 Summary of Research**

This thesis has pursued a single, coherent research question: whether a graph-based approach to topic modeling, grounded in explicit knowledge representation, offers substantive advantages over traditional probabilistic methods for analyzing political discourse. The investigation proceeded from a theoretical foundation in linguistics and the philosophy of language, through methodological development, and culminated in empirical evaluation on a corpus of European Parliamentary debates.

The central theoretical contribution is a reformulation of "topic" from a probabilistic construct—a distribution over words—into a structural one: a densely connected subgraph within a knowledge graph. This redefinition aligns more closely with intuitive notions of thematic coherence, wherein a topic comprises not merely co-occurring terms but a network of interrelated concepts bound by explicit semantic relationships.

The methodology developed for this thesis integrates several components into a unified pipeline: ontology-guided entity and relation extraction using large language models, multi-stage filtering to ensure semantic quality, community detection via the Leiden algorithm, and hierarchical summarization that leverages the extracted graph structure. Empirical evaluation on the "This is Europe" debate corpus (2022–2024) yielded several key findings.

First, the extracted knowledge graphs exhibit **scale-free topology**, with degree distributions following power laws consistent with those observed in organic complex networks. This structural property serves as validation of the extraction pipeline: rather than producing artificial or random graphs, the ontology-guided LLM extraction captures the hub-and-spoke dynamics of political discourse, where certain entities—such as "Ukraine," "energy," and "European unity"—consistently serve as semantic anchors around which arguments coalesce.

Second, community detection identifies **structurally coherent** partitions with high modularity scores (0.774–0.827 across iterations). The Leiden algorithm's guarantee of connected communities ensures that each identified topic forms a single, cohesive unit, satisfying the graph-theoretic counterpart to the philosophical requirement of thematic coherence.

Third, the **hierarchical, structure-aware summarization pipeline** successfully translates graph topology into interpretable, multi-faceted analytical reports. By providing the language model with explicit information about entity degrees, relationship triplets, and subtopic summaries, the system generates outputs in which every claim can be traced back to specific structural features of the underlying graph. This transparency represents a significant advantage over probabilistic topic models, where the relationship between a topic's label and the documents that instantiate it is mediated by opaque probability distributions.

Fourth, and perhaps most revealing, the evaluation exposed a fundamental **tension between structural coherence and semantic distinctiveness**. While modularity scores indicate strong community structure, silhouette scores remain negative (−0.091 to −0.184 at the subcommunity level), and topic overlap increases as the corpus grows. This divergence is not a methodological failure but a substantive finding: the "This is Europe" debates occurred during a period of interconnected crises—Russian aggression against Ukraine, the energy crisis, migration pressures, and post-pandemic recovery—such that European leaders discussed not independent topics but multiple facets of a unified geopolitical predicament. The negative silhouette scores quantify the degree of **thematic consilience** in European political discourse during this period.

### 5.2 Theoretical and Methodological Contributions

This research makes three contributions, each of which should be understood as modest steps toward understanding the potential of graph-based approaches rather than definitive advances.

**Methodologically**, this thesis demonstrates that graph-based topic modeling is a feasible and interpretable alternative to probabilistic approaches. The hierarchical summarization pipeline developed here provides one template—among possible alternatives—for translating complex graph structures into human-readable analytical reports, with auditability built into every step. Unlike LDA and its variants, which produce probability distributions whose interpretability requires additional post-processing, the graph-based approach generates explicit knowledge structures where each identified topic is a concrete subgraph that can be examined, validated, and traced back to source documents. Whether this property constitutes a genuine advantage in practice remains an empirical question.

**Theoretically**, this work offers one perspective on the relationship between structural coherence and semantic distinctiveness in topic modeling. The observation that standard clustering evaluation metrics—silhouette scores, separation measures—may be inappropriate for deliberative discourse adds to a growing body of literature questioning the applicability of generic clustering metrics to text analysis. In domains where consensus-building and shared framing are central features, "overlap" between topics may reflect a substantive property of the discourse rather than a methodological deficiency. The graph-based approach's ability to distinguish between *relational perspectives* and *semantic topics* offers one possible lens for analyzing discourse: where distributional models would see homogeneity, the structural approach can reveal variance in how entities are connected and positioned within argumentative networks. This interpretation, while plausible, would benefit from validation across additional corpora and discourse types.

**Practically**, the approach may be particularly well-suited for in-depth analysis of complex, interconnected discourse in domains where interpretability and verifiability are paramount. Political science, policy analysis, and legal text analysis are examples where the ability to trace every claim back to specific entities and relationships may be more valuable than the computational efficiency of probabilistic methods. This applicability remains a hypothesis to be tested in future applications.

### **5.3 Limitations**

The study's findings should be interpreted in light of several limitations.

**Parameter Sensitivity**: The optimal resolution parameter for community detection may vary across corpora. The current approach used fixed or empirically tuned parameters, but a more principled approach to parameter selection—perhaps informed by domain-specific considerations or cross-validation procedures—could improve robustness.

**Semantic Enrichment**: The current pipeline extracts entities and relationships but does not generate or embed entity descriptions (e.g., "European Union—a political and economic union of 27 member states"). Such descriptions could enrich the graph with semantic content, potentially improving alignment between structural and semantic topic quality.

**Temporal Dynamics**: The analysis treats the corpus as a static snapshot, but political discourse evolves over time. The observed degradation pattern—where topic distinctiveness decreases as the corpus grows—suggests that temporal dynamics play a significant role. A longitudinal analysis, tracking topic emergence, merging, and divergence across time, could provide additional insights.

**Evaluation Frameworks**: The study relied on standard clustering metrics (modularity, silhouette scores) that were designed for domains with clearly separated classes. For deliberative discourse, where overlap is expected and substantive, alternative evaluation frameworks that capture argumentative diversity, perspectival variance, or deliberative quality would be more appropriate. Developing such metrics is an important direction for future research.

**Corpus Scope**: The "This is Europe" corpus, while well-suited for testing the method's ability to handle thematic consilience, may not be representative of all political discourse. Application to corpora with more distinct thematic divisions—such as debates on specific policy areas or cross-national comparisons—could validate whether the observed tension between structural coherence and semantic distinctiveness is domain-specific.

### **5.4 Future Research Directions**

Several promising directions emerge from this research.

**Knowledge Graph Embeddings and Link Prediction**: The current pipeline does not exploit knowledge graph embeddings (KGEs) for link prediction. Models such as TransE, ComplEx, and RotatE learn vector space representations that capture complex relational patterns and can predict missing edges with high accuracy. Applying these techniques could enable (i) inference of implicitly mentioned relationships that the LLM extraction missed, (ii) identification of structurally anomalous connections, and (iii) generation of entity descriptions through neighborhood aggregation.

**Graph Neural Topic Models**: The topic modeling literature has seen advances in integrating graph-based regularization into neural topic models. The Graph Neural Topic Model (GNTM) uses document graphs to regularize topic distributions, while TextGCN builds heterogeneous graphs of words and documents for joint learning of word and document embeddings. The Node2Vec approach used in this thesis can be viewed as a simpler, unsupervised precursor to these more sophisticated architectures.

**Retrieval-Augmented Generation**: Recent developments in graph-based RAG systems—such as GraphRAG and LightRAG—demonstrate how entity knowledge graphs with hierarchical community summaries can improve question answering over large text corpora. These developments validate the central insight of this thesis: structuring text as a knowledge graph enables more principled and interpretable access to large discourse corpora. Future work could implement a GraphRAG-style interface for querying the parliamentary debates using natural language.

**Dynamic Graph Analysis**: Developing techniques to track topic evolution over time would address the temporal limitations of the current analysis. Dynamic graph models, incremental community detection, and temporal embedding methods could provide insights into how political discourse shifts in response to events and policy developments.

**Cross-Lingual and Multimodal Extensions**: The methodology could be extended to multilingual corpora to explore how topics are framed differently across languages. Integration of multimodal data—voting records, policy documents, or visual elements—could provide richer bases for analysis.

**Causal and Argumentative Analysis**: The explicit relational structure of the knowledge graph could be leveraged for tasks beyond topic identification, such as causal claim extraction, argumentative structure mapping, stance detection, and influence network analysis. These tasks would be difficult or impossible with purely distributional models.

### 5.5 Final Reflections

This thesis has explored whether defining topics as structurally coherent communities within a knowledge graph offers a viable alternative to the dominant probabilistic paradigm. The approach developed here is computationally more demanding than LDA or similar methods, but offers gains in explainability and auditability—qualities that may be valuable in domains where interpretability is paramount. It should be emphasized that, as Galke and Scherp (2022) demonstrated, simpler distributional methods often perform equally well or better in practice; this work does not claim superiority but rather contributes to a broader exploration of alternative paradigms.

The tension between structural coherence and semantic distinctiveness observed in the "This is Europe" corpus merits careful interpretation. Rather than viewing negative silhouette scores as a methodological shortcoming, this thesis proposes they may reflect a substantive property of the discourse: a period characterized by interconnected crises and consilient framing among European leaders. The graph-based approach's ability to represent this overlap faithfully—rather than artificially imposing separability—suggests potential value as a diagnostic tool for understanding discourse structure, though this claim would require further validation across additional corpora.

Future work in knowledge graph-based discourse analysis, graph neural topic models, and graph-enhanced retrieval systems may build on the insights developed here. More broadly, this thesis contributes to an ongoing conversation in the computational linguistics community about the role of structural representations in natural language understanding—understanding that remains, as yet, incomplete.

---

## **References**

Barabási, A.-L. & Albert, R. (1999) 'Emergence of scaling in random networks', *Science*, 286(5439), pp. 509-512.

Blei, D. M. (2012) 'Probabilistic topic models', *Communications of the ACM*, 55(4), pp. 77-84.

Blei, D. M., Ng, A. Y. & Jordan, M. I. (2003) 'Latent Dirichlet Allocation', *Journal of Machine Learning Research*, 3, pp. 993-1022.

Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J. & Yakhnenko, O. (2013) 'Translating embeddings for modeling multi-relational data', *Advances in Neural Information Processing Systems*, 26, pp. 2787-2795.

European Parliamentary Research Service (2024) *'This is Europe' debates: Analysis of EU leaders' speeches*. Drachenberg, R. & Bącal, P. Available at: https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI(2024)757844 (Accessed: 15 January 2024).

Firbas, J. (1992) *Functional sentence perspective in written and spoken communication*. Cambridge: Cambridge University Press.

Foucault, M. (1972) *The archaeology of knowledge*. New York: Pantheon Books.

Galke, L. & Scherp, A. (2022) 'Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP', *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 2365-2376.

Griffiths, T. L. & Steyvers, M. (2004) 'Finding scientific topics', *Proceedings of the National Academy of Sciences*, 101(suppl 1), pp. 5228-5235.

Halliday, M. A. K. (1985) *An introduction to functional grammar*. London: Edward Arnold.

Jolliffe, I. T. (2002) *Principal component analysis*. 2nd edn. New York: Springer.

Zhao, H., Phung, D., Huynh, V., Jin, Y., Du, L. & Buntine, W. (2021) 'Topic Modelling Meets Deep Neural Networks: A Survey', *arXiv preprint arXiv:2103.00498*.

Wu, X., Nguyen, T. & Luu, A. T. (2024) 'A Survey on Neural Topic Models: Methods, Applications, and Challenges', *arXiv preprint arXiv:2401.15351*.

Newman, M. E. J. (2005) 'Power laws, Pareto distributions and Zipf's law', *Contemporary Physics*, 46(5), pp. 323-351.

Newman, M. E. J. (2006) 'Modularity and community structure in networks', *Proceedings of the National Academy of Sciences*, 103(23), pp. 8577-8582.

Newman, M. E. J. & Girvan, M. (2004) 'Finding and evaluating community structure in networks', *Physical Review E*, 69(2), p. 026113.

Nguyen, D. Q. (2020) 'A survey of embedding models of entities and relationships for knowledge graph completion', *arXiv preprint arXiv:2003.08001*.

Reinhart, T. (1981) 'Pragmatics and linguistics: An analysis of sentence topics', *Philosophica*, 27(1), pp. 53-94.

Reinhart, T. (1982) 'Pragmatics and linguistics: An analysis of sentence topics', *Distributed by the Indiana University Linguistics Club*.

Salton, G. & McGill, M. J. (1986) *Introduction to modern information retrieval*. New York: McGraw-Hill.

Traag, V. A., Waltman, L. & van Eck, N. J. (2019) 'From Louvain to Leiden: guaranteeing well-connected communities', *Scientific Reports*, 9(1), p. 5233.

Grover, A. & Leskovec, J. (2016) 'node2vec: Scalable feature learning for networks', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 855–864.

Giles, H. (2016) 'Communication Accommodation Theory', in Berger, C. R. & Roloff, M. E. (eds.) *The International Encyclopedia of Interpersonal Communication*. Hoboken: Wiley, pp. 1-18.

Rousseeuw, P. J. (1987) 'Silhouettes: A graphical aid to the interpretation and validation of cluster analysis', *Journal of Computational and Applied Mathematics*, 20, pp. 53-65.

**Knowledge Graphs and Graph Neural Networks**

Ji, S., Pan, S., Cambria, E., Zhou, B. & Bian, Y. (2022) 'A Survey on Knowledge Graphs: Representation, Acquisition, and Applications', *IEEE Transactions on Neural Networks and Learning Systems*, 33(2), pp. 494-514. DOI: 10.1109/TNNLS.2021.3070843.

Sun, Z., Deng, Z.-H., Nie, J.-Y. & Tang, J. (2019) 'RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space', *Proceedings of the 7th International Conference on Learning Representations (ICLR)*.

Trouillon, T., Welbl, J., Riedel, S., Gaussier, E. & Bouchard, G. (2016) 'Complex Embeddings for Simple Link Prediction', *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, pp. 2075-2084.

Xie, Z., Wang, J., Huang, Y., Peng, H. & Liu, Y. (2021) 'Knowledge Graph Enhanced Neural Topic Model', *arXiv preprint arXiv:2110.11942*.

Yao, L., Shu, C., Liu, S., Zou, Y., Yang, J. & Liu, M. (2019) 'TextGCN: Word Graph Convolutional Network for Text Classification', *Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI)*, pp. 7456-7463.

**Graph-Based Retrieval and Topic Modeling**

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D. & Larson, J. (2024) 'From Local to Global: A Graph RAG Approach to Query-Focused Summarization', *arXiv preprint arXiv:2404.16130*.

Guo, Z., Xia, L., Yu, Y., Ao, T. & Huang, C. (2024) 'LightRAG: Simple and Fast Retrieval-Augmented Generation', *arXiv preprint arXiv:2410.05779*.


