

# **From Discourse to Structure: A Graph-Based Approach to Topic Modeling in Political Debate**

## **Abstract**

Traditional topic modeling techniques, such as Latent Dirichlet Allocation (LDA), have long served as the standard for thematic analysis of large text corpora. However, their reliance on "bag-of-words" assumptions, which disregard syntax and relational context, limits their ability to capture the nuanced structure of complex discourse. This thesis explores a graph-based paradigm for topic modeling that constructs and analyzes a knowledge graph from political text. The central research questions investigate whether thematic communities identified within this graph can be considered valid "topics" from a linguistic and philosophical standpoint, and whether this structural approach offers a complementary alternative to probabilistic models in terms of interpretability.

The methodology is applied to the verbatim reports of the 'This is Europe' European Parliamentary debate series (2022-2024). In the validated implementation path, ontology-derived labels guide a hybrid extraction pipeline that combines **GLiNER2** entity detection with **DSPy-based** relation extraction. The resulting graph is then processed through Node2Vec-informed edge weighting and Leiden community detection. This design is motivated by the nature of political discourse: debates contain persistent entities, implicit relations, and recurring cross-references that are not well captured by simple co-occurrence statistics alone.

Recognizing that discourse topics exist at multiple granularities—a limitation noted in Section 1.1—the Leiden algorithm is applied hierarchically, first identifying broad thematic clusters (macro-topics) and then detecting sub-communities within each (sub-topics). Each resulting community is then processed through an LLM-based summarization pipeline that generates interpretable labels and descriptive reports from graph structure and supporting textual evidence.

The strongest empirical result of the current validation phase is operational rather than inferential. On resource-constrained local hardware, the non-iterative pipeline scaled from 48 to 224 lexical chunks while producing steadily larger graphs: from 531 nodes / 1341 edges to 1916 nodes / 5550 edges. Across the same runs, the system produced between 49 and 122 topic-like community nodes (topics plus subtopics), indicating that the architecture can generate hierarchical thematic structure under realistic local constraints. Topic-separation reports were successfully produced, but they consistently indicated insufficient data for strong statistical interpretation. Accordingly, the present evidence supports structural feasibility and pipeline scalability more strongly than definitive claims about semantic separability.

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

This corpus is particularly well-suited for this thesis for two reasons. First, its thematic cohesion provides a stringent test for any topic model. The high degree of thematic overlap and shared vocabulary makes it difficult for purely statistical models to disentangle nuanced sub-topics. In practice, this means that strong structural partitions may coexist with weak semantic separation, making the corpus a useful stress test for graph-based topic modeling. Second, the detailed EPRS briefing serves as an expert-curated "ground truth." It provides an independent, qualitative baseline against which the computationally derived topics can be assessed, allowing for a cautious comparison between model output and expert political analysis.

### **2.2 From Text to Graph: A Hybrid Ontology-Guided Pipeline**

The transformation of unstructured text from the parliamentary debates into a structured knowledge graph is the foundational step of the methodology. In the validated implementation, this process combines ontology guidance, neural entity detection, and constrained relation extraction to create a relational representation of the discourse. The pipeline consists of several sequential stages.

1.  **Ontology-Guided Schema Definition:** Before any text is processed, the pipeline ingests a domain-specific ontology (defined in OWL/RDF formats). A dedicated `OntologyLabelExtractor` parses these files to extract a set of valid class labels (e.g., *Person*, *Policy*, *Organization*, *GeopoliticalEntity*). These labels serve as a schema that constrains subsequent extraction steps and reduces type drift in downstream graph construction.

2.  **Text Chunking:** The raw verbatim transcripts are segmented into manageable text chunks. Chunk size is a practical design parameter because it governs both computational cost and extraction granularity. In the local validation runs reported later, larger chunk sizes were preferred in order to keep the non-iterative pipeline tractable on CPU-only hardware.

3.  **Constrained Entity and Relationship Extraction:** Each text chunk is processed by a hybrid extraction pipeline. First, **GLiNER2** scans the text for ontology-compatible entity surfaces. These detections are normalized into entity hints. Next, a **DSPy-based relation extractor** is prompted with both ontology classes and the detected entity hints to produce grounded relation triplets. In the current implementation, relations whose endpoints cannot be aligned conservatively to the detected hints are dropped, which favors precision over aggressive graph expansion.

4.  **Graph Assembly:** The extracted elements from all chunks are aggregated into a single knowledge graph. Entities become nodes and extracted relations become edges. The result is a structured, machine-readable representation of the debate corpus that can be used for community detection, summarization, and downstream analysis.

### **2.3 The Emergence of Structure: Scale-Free Properties in Extracted Knowledge**

A significant finding that validates the integrity of this graph construction process is the topological nature of the resulting network. Analysis of the graph's degree distribution—the probability P(k) that a randomly chosen node has k connections—reveals that it follows a power law, where P(k)∼k−γ. This is the defining characteristic of a scale-free network (Barabási & Albert, 1999; Newman, 2005).

The study of scale-free networks was pioneered by Albert-László Barabási and Réka Albert (1999), who discovered that this topology is not a mathematical curiosity but a ubiquitous feature of real-world complex systems, including the World Wide Web, social networks, and biological protein-interaction networks. The emergence of this structure is explained by two simple, yet powerful, underlying mechanisms:

**growth** and **preferential attachment** (Barabási & Albert, 1999).

* **Growth:** Real networks are rarely static; they expand over time through the addition of new nodes.  
* **Preferential Attachment:** New nodes are more likely to connect to existing nodes that are already highly connected. This "rich-get-richer" phenomenon leads to the formation of a few highly connected "hubs" that dominate the network's structure.

The appearance of a scale-free topology in the knowledge graph extracted from the parliamentary debates is not a random artifact. It is a profound reflection of the fundamental dynamics of discourse itself. A series of political debates is a growing system: each speech adds new concepts and arguments to the existing network of ideas (growth). When speakers contribute, they do not introduce concepts in a vacuum. To be relevant and persuasive, they must connect their arguments to the central, most salient themes of the ongoing discussion—the established hubs like 'Ukraine', 'energy dependency', or 'EU values' (preferential attachment) (Barabási & Albert, 1999).

Therefore, the very process of building a coherent, multi-speaker discourse is a network-generating mechanism that naturally follows the principles of scale-free models. The appearance of this exact topology in our generated graph acts as a **structural validation of the extraction pipeline**. It confirms that the ontology-guided extraction did not impose an artificial or rigid structure, but rather successfully captured the organic, self-organizing nature of the political debate. A graph that appeared random (Poisson degree distribution) or regular (lattice-like) would indicate a failure of the extraction process to identify the true semantic hubs. The scale-free property confirms that the "hubs" in our graph (e.g., *Ukraine*, *Energy Crisis*) correspond to the actual semantic anchors of the real-world discourse.

### **2.4 Ensuring Coherence: Conservative Entity Normalization and Resolution**

The integrity of the graph's structure, particularly the accurate identification of hubs, depends on a crucial data processing step: Entity Resolution (ER). Raw extraction output can be inconsistent, creating multiple nodes for the same real-world entity (e.g., "Olaf Scholz," "the German Chancellor," "Mr. Scholz"). ER is the process of identifying and merging such duplicates so that each unique entity is represented as consistently as possible.

In the validated local pipeline, this step is implemented conservatively. Rather than treating knowledge graph embeddings as the primary validated mechanism for large-scale automatic merging, the current workflow relies mainly on normalization, strict similarity thresholds, and downstream graph cleaning to avoid unsupported merges. This design choice reflects the practical priority of preserving trustworthy graph structure during local validation.

This conservative behavior is visible in the reported local scaling runs, where the entity-resolution stage produced stable graphs without aggressive automatic collapsing of nodes. The interpretation of these results should therefore be modest: the current evidence validates extraction-and-assembly stability more strongly than it validates a large-scale automated canonicalization system.

Knowledge graph embeddings remain relevant as a future direction for entity resolution and link prediction (Nguyen, 2020), but they are not the main validated mechanism in the non-iterative local pipeline emphasized in this thesis.

### **2.5 Conservative Filtering and Graph Cleanup**

The knowledge graph construction pipeline incorporates several filtering and cleanup mechanisms intended to reduce noise and preserve interpretable structure before community detection. In the current implementation, this is less an exhaustive linguistic filter stack than a conservative graph-construction strategy.

The main design principle is precision over aggressive abstraction. Ontology-derived labels constrain the entity search space, DSPy relation extraction is grounded in detected entity hints, and weak or unsupported structure is pruned during downstream graph cleanup. Similarity thresholds and minimum-component rules are used to avoid over-merging and to reduce fragmentation caused by low-value nodes or edges.

These mechanisms serve two related purposes. First, they reduce the chance that community structure is driven by spurious extractions or accidental bridges. Second, they make the downstream summaries more interpretable by ensuring that the graph presented to the community-detection stage is already a conservative representation of the discourse. In a local validation setting, this trade-off is desirable: it is preferable to retain a slightly incomplete graph than to create topic communities from unsupported structure.

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

In the validated non-iterative local runs reported in this thesis, Node2Vec remains part of the implemented community-detection pipeline as a structurally motivated weighting layer. Across the successful scaling probes, the weighted pipeline remained stable while graph size increased substantially. However, these local manifests do not yet show a positive modularity delta relative to the unweighted baseline. Accordingly, Node2Vec is best presented here as a validated architectural component and a theoretically motivated structural prior, rather than as a locally re-demonstrated source of modularity improvement in the current capped runs.

### **3.2 Hierarchical Abstraction of Discourse**

Human discourse is often organized hierarchically. A broad theme, such as "EU's Main Challenges," can be broken down into more specific sub-themes like "The Energy Crisis," "Inflation," and "Disinformation," which themselves can be further decomposed. The Leiden algorithm naturally accommodates this nested structure.

In the implementation validated here, the hierarchical procedure is applied in two levels. After an initial partition of the graph is found, each resulting community is treated as a smaller graph and processed again to identify sub-communities. This produces a nested partitioning of the data, providing a multi-resolution view of the thematic landscape without requiring an open-ended recursive depth. The result is still useful for analysis: it allows the discourse to be examined both at the level of broader themes and at the level of finer-grained sub-topics.

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

### **4.1 Local Scaling Behaviour of the Validated Pipeline**

The current validation phase focuses on a non-iterative local pipeline rather than on cumulative iterative experimentation. The aim is correspondingly narrower: to assess whether the implemented GLiNER2 + DSPy + Node2Vec path continues to produce coherent graph structure and hierarchical community summaries as chunk counts increase under local execution constraints.

#### 4.1.1 Structural growth under chunk scaling

Across the completed local probes, graph size increases consistently as chunk count is raised. A 48-chunk run produced a graph with **531 nodes** and **1341 edges**; a 64-chunk run produced **661 nodes** and **1733 edges**; a 96-chunk run produced **923 nodes** and **2529 edges**; a 128-chunk run produced **1155 nodes** and **3239 edges**; a 160-chunk run produced **1377 nodes** and **3946 edges**; a 187-chunk run produced **1617 nodes** and **4670 edges**; and a 224-chunk run produced **1916 nodes** and **5550 edges**. This monotonic growth is an important engineering validation because it suggests that the extraction and graph-assembly stages continue to add usable structure without obvious collapse, runaway sparsification, or instability on the available local hardware.

The extraction statistics show a similar progression. The same sequence of runs yielded **293**, **380**, **549**, **767**, **856**, **1072**, and **1246** extracted entities, alongside **145**, **208**, **321**, **411**, **463**, **547**, and **637** extracted relations. These counts indicate that larger chunk budgets do not merely inflate the lexical scaffold; they also translate into additional semantic content in the assembled graph.

Table 4.1 summarizes the validated local scaling ladder.

| Run | Chunks | Entities | Relations | Nodes | Edges | Topics | Subtopics | Modularity | Topic analysis |
| :-- | -----: | -------: | --------: | ----: | ----: | -----: | --------: | ---------: | :------------- |
| v2 | 48 | 293 | 145 | 531 | 1341 | 10 | 39 | 0.6312 | Insufficient data for analysis |
| scale1 | 64 | 380 | 208 | 661 | 1733 | 14 | 47 | 0.6748 | Insufficient data for analysis |
| scale2 | 96 | 549 | 321 | 923 | 2529 | 16 | 66 | 0.7108 | Insufficient data for analysis |
| scale3 | 128 | 767 | 411 | 1155 | 3239 | 14 | 72 | 0.6567 | Insufficient data for analysis |
| scale4 | 160 | 856 | 463 | 1377 | 3946 | 14 | 80 | 0.6593 | Insufficient data for analysis |
| scale5 | 187 | 1072 | 547 | 1617 | 4670 | 17 | 91 | 0.6526 | Insufficient data for analysis |
| scale6 | 224 | 1246 | 637 | 1916 | 5550 | 18 | 104 | 0.6711 | Insufficient data for analysis |

#### 4.1.2 Growth of hierarchical community structure

The number of generated community summaries also rises across these probes. The completed runs produced **10 topics / 39 subtopics** at 48 chunks, **14 / 47** at 64 chunks, **16 / 66** at 96 chunks, **14 / 72** at 128 chunks, **14 / 80** at 160 chunks, **17 / 91** at 187 chunks, and **18 / 104** at 224 chunks. Expressed as total topic-like units, this corresponds to **49**, **61**, **82**, **86**, **94**, **108**, and **122** topic-like nodes respectively.

This pattern does not in itself prove semantic distinctiveness, but it does show that the non-iterative pipeline continues to materialize a usable hierarchical thematic representation as the local sample grows. In practical terms, the summarization stage is not merely surviving larger runs; it is producing increasingly rich topic and subtopic inventories that can be inspected qualitatively.

#### 4.1.3 Stability of community quality metrics

The modularity values remain within a relatively narrow and consistently positive range, from **0.6312** to **0.7108** across the validated runs. This suggests that larger local runs do not destroy the structural coherence of the detected communities even as the graph expands substantially. At the same time, the local manifests report **modularity deltas of 0.0** between the weighted and baseline variants, meaning that in the current capped local experiments Node2Vec is best interpreted as a stable architectural component rather than as a newly re-demonstrated source of modularity gains.

This is an important result for thesis interpretation. The current evidence supports structural robustness under scaling, but it does not justify a stronger claim that local scale-up alone improves community quality. In other words, the graph becomes larger and richer while community quality remains broadly stable rather than dramatically better.

#### 4.1.4 Current limits of semantic evaluation

The present local outputs do not yet support strong claims about semantic separation between the detected communities. In all completed runs inspected here, the automatically generated topic-separation reports return the judgment **"Insufficient data for analysis"**, with global separation and overlap both reported as **0.0**. This should not be interpreted as a pipeline failure. Rather, it indicates that the reporting and analytics stages are functioning, but that the current local validation regime does not yet provide a sufficiently informative basis for robust inferential claims about semantic topic separation.

Accordingly, the strongest conclusion from the current results is that the validated non-iterative pipeline is operationally stable and structurally scalable on modest hardware. The evidence is much stronger for feasibility, graph growth, and hierarchical summary generation than for definitive semantic separability.

### **4.2 Qualitative Validation Against External Analysis**

A full qualitative alignment study against the EPRS taxonomy has not yet been rerun on the validated local non-iterative path. Such a comparison remains feasible, but the present chapter limits itself to reporting structural scaling behaviour and cautious methodological interpretation. This is a deliberate choice: it is preferable to separate implementation-faithful validation from broader substantive claims until the same comparison has been repeated on the currently supported pipeline.

### **4.3 Similarity Analysis and Distribution**

At the current validation stage, similarity-based interpretation remains provisional. Although the pipeline successfully generates topic and subtopic summaries, the corresponding topic-separation reports for the completed local runs indicate insufficient data for robust pairwise analysis. It would therefore be premature to infer strong semantic distance or overlap patterns from the present local outputs alone. A more defensible use of the current results is to show that the graph-based pipeline can assemble, summarize, and preserve hierarchical structure at increasing scale, while leaving stronger claims about topic separation for later full-corpus evaluation.

---



## **Chapter 5: Conclusion**

### **5.1 Summary of Research**

This thesis has pursued a single, coherent research question: whether a graph-based approach to topic modeling, grounded in explicit knowledge representation, offers substantive advantages over traditional probabilistic methods for analyzing political discourse. The investigation proceeded from a theoretical foundation in linguistics and the philosophy of language, through methodological development, and culminated in empirical evaluation on a corpus of European Parliamentary debates.

The central theoretical contribution is a reformulation of "topic" from a probabilistic construct—a distribution over words—into a structural one: a densely connected subgraph within a knowledge graph. This redefinition aligns more closely with intuitive notions of thematic coherence, wherein a topic comprises not merely co-occurring terms but a network of interrelated concepts bound by explicit semantic relationships.

The methodology developed for this thesis integrates several components into a unified pipeline: ontology-guided schema construction, GLiNER2-based entity detection, DSPy-based relation extraction, community detection via the Leiden algorithm, Node2Vec-informed structural weighting, and hierarchical summarization that leverages the extracted graph structure. Empirical evaluation on the "This is Europe" debate corpus (2022–2024) yielded several cautious findings.

First, the extracted knowledge graphs exhibit sustained structural growth under increasing local chunk budgets. In the validated non-iterative runs, graph size increased from **531 nodes / 1341 edges** at 48 chunks to **1916 nodes / 5550 edges** at 224 chunks. This behaviour suggests that the pipeline can continue to add entities, relations, and communities without obvious instability on modest local hardware.

Second, community detection produced consistently interpretable hierarchical outputs. Across the same runs, the summarization stage generated between **10 and 18 topics** and between **39 and 104 subtopics**, indicating that the graph structure can support multi-level thematic abstraction as the corpus sample grows.

Third, the study supports a modest practical claim about feasibility and interpretability rather than a strong inferential claim about topic separation. The pipeline now produces graph artifacts, community summaries, and topic-separation reports reliably in a non-iterative setting. However, the available topic-separation reports remain statistically inconclusive, repeatedly indicating insufficient data for robust semantic interpretation.

Fourth, the current evidence suggests an important asymmetry between structural coherence and semantic distinctiveness. The pipeline is already capable of generating coherent graph structure and hierarchical summaries, but stronger claims about semantic separation require larger or differently structured evaluation settings. This should be treated as a limitation of the present validation stage rather than as proof of either success or failure for the broader graph-based paradigm.

### 5.2 Theoretical and Methodological Contributions

This research makes three contributions, each of which should be understood as modest steps toward understanding the potential of graph-based approaches rather than definitive advances.

**Methodologically**, this thesis demonstrates that graph-based topic modeling is a feasible and interpretable alternative to probabilistic approaches. The hierarchical summarization pipeline developed here provides one template—among possible alternatives—for translating complex graph structures into human-readable analytical reports, with auditability built into every step. Unlike LDA and its variants, which produce probability distributions whose interpretability requires additional post-processing, the graph-based approach generates explicit knowledge structures where each identified topic is a concrete subgraph that can be examined, validated, and traced back to source documents. Whether this property constitutes a genuine advantage in practice remains an empirical question.

**Theoretically**, this work offers one perspective on the relationship between structural coherence and semantic distinctiveness in topic modeling. The observation that standard clustering evaluation metrics—silhouette scores, separation measures—may be inappropriate for deliberative discourse adds to a growing body of literature questioning the applicability of generic clustering metrics to text analysis. In domains where consensus-building and shared framing are central features, "overlap" between topics may reflect a substantive property of the discourse rather than a methodological deficiency. The graph-based approach's ability to distinguish between *relational perspectives* and *semantic topics* offers one possible lens for analyzing discourse: where distributional models would see homogeneity, the structural approach can reveal variance in how entities are connected and positioned within argumentative networks. This interpretation, while plausible, would benefit from validation across additional corpora and discourse types.

**Practically**, the approach may be particularly well-suited for in-depth analysis of complex, interconnected discourse in domains where interpretability and verifiability are paramount. Political science, policy analysis, and legal text analysis are examples where the ability to trace every claim back to specific entities and relationships may be more valuable than the computational efficiency of probabilistic methods. This applicability remains a hypothesis to be tested in future applications.

### **5.3 Limitations**

The study's findings should be interpreted in light of several limitations.

**Parameter Sensitivity**: The optimal resolution parameter for community detection may vary across corpora. The current approach used fixed or empirically tuned parameters, but a more principled approach to parameter selection—perhaps informed by domain-specific considerations or cross-validation procedures—could improve robustness.

**Local Validation Scope**: The present validation is based on capped, non-iterative local runs rather than on a single full-corpus benchmark. This means that the strongest current conclusions concern operational stability, graph growth, and summary generation under constrained conditions, not exhaustive corpus-wide topic evaluation.

**Conservative Entity Resolution**: The validated local path prioritizes precision over aggressive canonicalization. This reduces the risk of unsupported merges, but it also means that the current experiments should not be interpreted as a full validation of large-scale automated entity resolution.

**Semantic Enrichment**: The current pipeline extracts entities and relationships but does not yet enrich the graph with explicit entity descriptions or other auxiliary semantic metadata that might improve alignment between structural and semantic topic quality.

**Evaluation Frameworks**: The study relied on topic-separation and clustering-style diagnostics that, in the present local runs, remain statistically underdetermined. For deliberative discourse, where overlap is expected and substantive, alternative evaluation frameworks that capture argumentative diversity, perspectival variance, or deliberative quality may prove more informative.

**Corpus Scope**: The "This is Europe" corpus, while well-suited for testing the method's ability to handle thematic consilience, may not be representative of all political discourse. Application to corpora with more distinct thematic divisions—such as debates on specific policy areas or cross-national comparisons—could clarify whether the present combination of structural coherence and weak semantic separation is domain-specific.

### **5.4 Future Research Directions**

Several promising directions emerge from this research.

**Knowledge Graph Embeddings and Link Prediction**: The current pipeline does not exploit knowledge graph embeddings (KGEs) as a mainline validated component for link prediction or entity resolution. Models such as TransE, ComplEx, and RotatE learn vector space representations that capture complex relational patterns and could support (i) inference of implicitly mentioned relationships, (ii) identification of structurally anomalous connections, and (iii) more ambitious automated entity canonicalization. These possibilities should be understood as extensions to the presently validated GLiNER2 + DSPy + Node2Vec local pipeline, not as part of the current mainline result.

**Graph Neural Topic Models**: The topic modeling literature has seen advances in integrating graph-based regularization into neural topic models. The Graph Neural Topic Model (GNTM) uses document graphs to regularize topic distributions, while TextGCN builds heterogeneous graphs of words and documents for joint learning of word and document embeddings. The Node2Vec approach used in this thesis can be viewed as a simpler, unsupervised precursor to these more sophisticated architectures.

**Retrieval-Augmented Generation**: Recent developments in graph-based RAG systems—such as GraphRAG and LightRAG—demonstrate how entity knowledge graphs with hierarchical community summaries can improve question answering over large text corpora. These developments validate the central insight of this thesis: structuring text as a knowledge graph enables more principled and interpretable access to large discourse corpora. Future work could implement a GraphRAG-style interface for querying the parliamentary debates using natural language.

**Dynamic and Incremental Graph Analysis**: The present thesis validates a non-iterative pipeline, but future work could study how graphs grow across time in a more principled way. A Cognee-like accumulation perspective—where entity resolution, upsert logic, and merge eligibility are treated as first-class problems—could provide a better framework for longitudinal graph growth than the earlier iteration-as-evaluation approach. Dynamic graph models, incremental community detection, and temporal embedding methods could then be used to analyze how discourse changes in response to events and policy developments.

**Cross-Lingual and Multimodal Extensions**: The methodology could be extended to multilingual corpora to explore how topics are framed differently across languages. Integration of multimodal data—voting records, policy documents, or visual elements—could provide richer bases for analysis.

**Causal and Argumentative Analysis**: The explicit relational structure of the knowledge graph could be leveraged for tasks beyond topic identification, such as causal claim extraction, argumentative structure mapping, stance detection, and influence network analysis. These tasks would be difficult or impossible with purely distributional models.

### 5.5 Final Reflections

This thesis has explored whether defining topics as structurally coherent communities within a knowledge graph offers a viable alternative to the dominant probabilistic paradigm. The approach developed here is computationally more demanding than LDA or similar methods, but offers gains in explainability and auditability—qualities that may be valuable in domains where interpretability is paramount. It should be emphasized that, as Galke and Scherp (2022) demonstrated, simpler distributional methods often perform equally well or better in practice; this work does not claim superiority but rather contributes to a broader exploration of alternative paradigms.

The tension between structural coherence and semantic distinctiveness in the "This is Europe" corpus merits careful interpretation. The current local results show that graph structure and hierarchical summaries can be produced reliably, but they do not yet support strong inferential claims about semantic separation. This asymmetry should be understood neither as a decisive failure nor as conclusive proof of superiority. Rather, it suggests that graph-based topic modeling may be especially valuable as a structural and interpretive tool, while the question of semantic distinctiveness requires broader validation across additional corpora and evaluation settings.

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


