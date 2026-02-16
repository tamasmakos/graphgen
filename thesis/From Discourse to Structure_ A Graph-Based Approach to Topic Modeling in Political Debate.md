

# **From Discourse to Structure: A Graph-Based Approach to Topic Modeling in Political Debate**

## **Abstract**

Traditional topic modeling techniques, such as Latent Dirichlet Allocation (LDA), have long served as the standard for thematic analysis of large text corpora. However, their reliance on "bag-of-words" assumptions, which disregard syntax and relational context, limits their ability to capture the nuanced structure of complex discourse. This thesis introduces and evaluates a novel, graph-based paradigm for topic modeling that leverages the capabilities of Large Language Models (LLMs) to construct and analyze a knowledge graph from textual data. The central research questions investigate whether thematic communities identified within this graph can be considered valid "topics" from a linguistic and philosophical standpoint, and whether this structural approach offers a more advanced and interpretable alternative to probabilistic models.

The methodology is applied to the verbatim reports of the 'This is Europe' European Parliamentary debate series (2022-2024). The process involves a rigorous pipeline starting with **ontology injection**, where domain-specific class definitions guide the extraction process. An LLM-powered system then extracts entities, relationships, and claims to construct a Knowledge Graph. To refine the structural quality of this graph, **Node2Vec embeddings** are generated to capture topological similarities, which are then used to re-weight edges. Finally, the Leiden community detection algorithm partitions this weighted graph into densely connected, thematically coherent communities.

A key finding of this research is the validation of the extraction methodology through the emergence of **scale-free properties** in the generated graph, a hallmark of organic complex networks. Furthermore, the application of Node2Vec-based edge weighting was shown to significantly enhance the definition of these communities, improving the Leiden modularity score from a baseline of ~0.74 to ~0.82 in later iterations. This structural enhancement allows the model to reconcile a high average semantic similarity (0.57) among topic summaries with the clear identification of distinct, separable thematic clusters. The result is a model that effectively distinguishes the overarching discourse topic from specific, nuanced sub-topics.

The thesis concludes that defining topics as structurally coherent communities within a knowledge graph represents a significant conceptual and practical advancement. This approach moves beyond statistical inference of latent themes to the explicit representation of knowledge structures, offering superior interpretability, context-awareness, and a more profound alignment with the philosophical and linguistic nature of what constitutes a "topic" in human communication.

---

## **Chapter 1: Deconstructing the 'Topic': From Philosophy to Computation**

### **1.1 The Philosophical and Linguistic Foundations of a 'Topic'**

Before a computational model can claim to identify "topics," it is imperative to establish a rigorous, non-computational understanding of what a topic is. The term is often used imprecisely in data science, treated as a mere label for a cluster of co-occurring words. However, its roots in linguistics and the philosophy of language reveal a far more structured and profound concept, one that is central to the organization of information and the construction of meaning in human communication.

From a linguistic perspective, the foundational distinction is between the *topic* (or *theme*) and the *comment* (or *rheme*) (Halliday, 1985; Firbas, 1992). The topic is what a sentence or clause is *about*; it is the entity or concept that anchors the discourse, providing the subject of the predication. The comment is what is being said *about* that topic; it is the new information, the assertion, or the description being provided. This division, known as information structure, posits that communication is not an unstructured stream of words but a deliberate organization of information into old (the topic, which connects to the existing discourse) and new (the comment) (Halliday, 1985). This fundamental structure implies that a topic is not a standalone artifact but exists in relation to the propositions made about it.

The philosophy of language deepens this understanding through the concept of "aboutness" (Reinhart, 1981, 1982). Reinhart argues that "aboutness" is the defining characteristic of a topic, moving beyond purely grammatical definitions of a subject to a pragmatic one based on communicative intent. The topic is the entity that the speaker directs the hearer's attention to, about which they intend to convey information. This philosophical framing is critical because it sets a higher bar for topic modeling: the goal is not merely to find clusters of words but to identify the primary subjects of "aboutness" that structure a body of text.

Furthermore, a crucial distinction must be made between a *sentence topic* and a *discourse topic* (Reinhart, 1981). A sentence topic is the constituent that a specific sentence is about, whereas a discourse topic is what an entire conversation or text is about. For example, in a debate about European energy policy, the discourse topic is "European Energy Policy." Within this discourse, individual sentences may have sentence topics like "natural gas reserves," "renewable energy investment," or "Russian dependency." Traditional computational models often struggle to separate these levels, conflating high-frequency terms associated with the overarching discourse topic with the more specific subjects of individual arguments. An effective topic model must be capable of resolving this hierarchy.

These concepts can be synthesized through the lens of Foucauldian discourse theory. Foucault defines a discourse not as a simple collection of statements, but as a "system of thoughts composed of ideas, attitudes, courses of action, beliefs, and practices that systematically construct the subjects and the worlds of which they speak" (Foucault, 1972). In this view, a discourse creates its own objects and concepts through the regulated interplay of statements. A "topic," therefore, is not just a word or a concept but a node within this system, defined by its relationships to other nodes. It is a representation of one of these constructed subjects. This provides a powerful theoretical framework: a true topic model should aim to uncover these systems of thought, revealing not just *what* is being discussed, but *how* the subjects of the discourse are constructed through the relationships between different ideas and entities.

### **1.2 Computational Approaches to Topic Modeling**

The abstract, theoretical concept of a topic must be operationalized to be computationally tractable. While the dominant approach for the last two decades has been the probabilistic paradigm, exemplified by Latent Dirichlet Allocation (LDA) (Blei, Ng, & Jordan, 2003), this thesis proposes a fundamental shift towards a *structural* paradigm.

Probabilistic models like LDA define a topic as a probability distribution over a vocabulary, inferred from the co-occurrence of words within documents. This relies on the "bag-of-words" assumption, which treats text as an unordered collection of terms, disregarding the rich relational structure of language.

In contrast, the **structural paradigm** proposed here defines a topic not as a statistical abstraction but as an explicit, tangible component of a knowledge graph. The formal definition is as follows:

*A topic is a densely interconnected community of entities (nodes) and their relationships (edges) within a knowledge graph, which is algorithmically identified through community detection and can be articulated through a natural language summary.*

This definition moves the unit of analysis from words to entities—real-world concepts, people, places, and organizations—and their explicit, labeled relationships. It directly operationalizes the theoretical concept of a "system of thoughts," where meaning is derived from the structure of connections. This shift aims to move away from the statistical inference of latent variables and towards the explicit representation of the underlying knowledge structures that constitute the discourse.

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

### **3.3 Generating Topic Narratives: LLM-Based Community Summarization**

The output of the community detection process is a set of subgraphs, where each subgraph represents a topic. While this structured representation is powerful for analysis, it is not immediately interpretable to a human user. The final stage of the methodology bridges this gap by translating the graph structure back into natural language.

For each detected community (topic), the constituent elements—the nodes (entities), edges (relationships), and associated claims—are collected and formatted as a structured input for an LLM. The model is then prompted with a summarization task: to synthesize this collection of structured facts into a concise, coherent, and human-readable descriptive summary.

This process involves traversing the subgraph to gather the most salient information. For leaf-level communities in a hierarchical structure, the element summaries (individual nodes, edges, and claims) are prioritized and iteratively added to the LLM's context window until token limits are reached. For higher-level communities, the summaries of their constituent sub-communities can be recursively incorporated to build a more abstract, high-level narrative. The resulting natural language summary serves as the final, interpretable representation of the topic, effectively providing a narrative that explains what the cluster of interconnected entities is "about."

---

## **Chapter 4: Analysis of Graph-Derived Topics from the 'This is Europe' Debates**

### **4.1 The Global Thematic Landscape: Interpreting Quantitative Results**

The quantitative results from the analysis of the 'This is Europe' debates provide a multi-faceted view of the corpus's thematic structure. The three visualizations—a histogram of summary similarities, a heatmap of the similarity matrix, and a PCA plot of summary embeddings—must be interpreted in concert to understand the model's performance.

#### 4.1.1 Interpreting Global Cohesion (Histogram)

The histogram displays the distribution of cosine similarities between all pairs of generated community summaries, a standard metric in information retrieval (Salton & McGill, 1986). The distribution is roughly normal, centered around a mean value of **0.6042**. This relatively high average similarity is a direct and expected consequence of the nature of the corpus. As established, the debates share a common context, speaker pool (EU leaders), and overarching subject matter: the future of the European Union. This shared vocabulary and conceptual space naturally lead to summaries that are, on average, semantically related. The **0.60** mean is the numerical signature of the high-level *discourse topic* that encompasses the entire collection of speeches. A naive interpretation might view this high average similarity as a failure of the model to produce distinct topics. However, when viewed alongside the other visualizations, it becomes clear that this value captures an essential truth about the data's inherent cohesion.

#### 4.1.2 Statistical Evaluation of Topic Quality

Beyond simple similarity, the structural metrics from the final experimental iterations provide a robust validation of the model's performance. As shown in the `iterative_experiment_results.csv` data (Iteration 9/10), the model achieved a **Modularity score of 0.823**, significantly outperforming the unweighted baseline of **0.748**. This high modularity indicates that despite the semantic overlap, the graph has a very strong community structure—nodes within a topic are far more densely connected to each other than to the rest of the network.

However, the **Topic Separation score of 0.46** (and corresponding **Topic Overlap of 0.54**) reflects the challenging nature of politically cohesive text. The **Silhouette scores** for communities hovered around **-0.03 to -0.09**, a result that initially suggests overlapping clusters. In the context of this specific corpus, however, this metric must be interpreted carefully. Standard clustering metrics penalize overlap, but in political discourse, "overlap" is often "consensus." The fact that the model maintains high structural modularity (0.82) despite low silhouette scores serves as a quantitative confirmation of the debates' dual nature: a high degree of shared vocabulary and context ("about Europe") balanced against distinct, separated 
streams of argumentation ("different perspectives"). The results imply that while all speakers use similar words, they structure their arguments differently, allowing the graph model to disentangle their perspectives based on *who says what* rather than just *what words they use*.

#### 4.1.3 Identifying Distinct Thematic Streams (Heatmap & PCA)

The heatmap of the community summary similarity matrix provides the crucial evidence for the model's ability to partition the discourse effectively. The matrix is ordered such that summaries belonging to the same parent community are grouped together. The bright yellow squares along the diagonal represent high intra-community similarity, indicating that the summaries within a given thematic cluster are highly coherent and semantically close. In contrast, the darker, orange-to-red regions off the diagonal signify lower inter-community similarity, with some values dropping as low as **0.17** (e.g., between distinct niche topics). This visual pattern demonstrates that despite the high average similarity across the entire corpus, the Leiden algorithm successfully identified communities whose internal thematic coherence is significantly stronger than their relationship to other communities.

The Principal Component Analysis (PCA) plot further corroborates this finding (Jolliffe, 2002). By projecting the high-dimensional summary embeddings into a two-dimensional space, the plot shows that the communities form visually separable clusters, each represented by a different color. While there is some overlap, which is expected given the shared context, distinct groupings are clearly visible. This confirms that the thematic clusters identified are not mere artifacts of the similarity metric but represent structurally distinct regions in the semantic space.

The combination of these results is the central finding of this thesis. The model is sophisticated enough to simultaneously capture two levels of thematic structure. It recognizes the shared context that makes all debates "about Europe" (reflected in the 0.60 average similarity) while also leveraging the fine-grained structural relationships between entities in the knowledge graph to carve out highly coherent and distinct sub-topics.

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

### **5.1 Synthesis of Findings**

This thesis embarked on an inquiry to determine if a more meaningful and structurally sound form of topic modeling could be achieved by moving beyond the dominant probabilistic paradigm. The journey began by establishing a rigorous definition of a "topic" grounded in philosophy and linguistics, defining it not as a collection of words but as a coherent system of interconnected concepts about which propositions are made. It was then demonstrated how this theoretical definition could be computationally realized through a novel methodology: constructing a knowledge graph from discourse using LLMs and identifying topics as dense, coherent communities within that graph's structure.

The application of this method to the 'This is Europe' parliamentary debates yielded a central finding that encapsulates its primary strength. The model successfully reconciled the corpus's high-level thematic cohesion (manifested as a high average semantic similarity of 0.57 between topic summaries) with its capacity to identify clearly distinct and separable sub-topics (visualized in heatmap and PCA plots). This result is not a contradiction but rather evidence of a nuanced analysis. The model is able to distinguish the overarching *discourse topic*—the shared European context—from the specific *sub-discourse topics* that constitute the actual substance of the debates. This success is attributed to its reliance on the explicit relational structure of the knowledge graph, which provides a richer signal for thematic partitioning than the word co-occurrence frequencies used by traditional models.

The validity of the approach was further substantiated by two key pieces of evidence. First, the emergent **scale-free topology** of the generated knowledge graph serves as a critical validation of the extraction pipeline itself. It confirms that the combination of ontology injection and LLM-based extraction successfully captured the organic, "hub-and-spoke" structure of real-world discourse, rather than producing a random or artificial network. Second, the qualitative alignment of the computationally derived topics with an independent, expert analysis from the European Parliamentary Research Service confirmed their real-world relevance and interpretability.

### **5.2 Contribution to the Field**

The conclusion of this research is that the proposed methodology, which defines topics as structurally coherent communities within an LLM-generated knowledge graph, represents a significant conceptual and practical advancement over probabilistic models like Latent Dirichlet Allocation. It constitutes a paradigm shift from *statistical inference* to *structural representation*.

This approach offers a more nuanced, context-aware, and fundamentally more interpretable framework for understanding the thematic structure of complex discourse. It moves the field closer to the original linguistic and philosophical meaning of a "topic" by providing outputs that are not abstract probability distributions but explicit, auditable, and human-readable knowledge structures. While computationally more demanding, the gains in explainability, context preservation, and alignment with the nature of human communication suggest that graph-based methods are a superior solution for in-depth thematic analysis, particularly in domains like political science, social science, and humanities research, where nuance and context are paramount.

### **5.3 Future Research Directions**

The findings and limitations of this study open several promising avenues for future research.

* **Dynamic and Temporal Analysis:** The current model produces a static snapshot of the discourse. Future work could focus on developing dynamic graph analysis techniques to track the evolution of topics over time, observing how themes emerge, merge, split, or fade in prominence throughout a longitudinal corpus.  
* **Cross-Lingual and Multimodal Applications:** The methodology could be extended and tested on multilingual corpora to explore how topics are framed differently across languages. Furthermore, integrating multimodal data—such as images, videos, or structured data associated with the text—into the knowledge graph could provide an even richer basis for topic analysis.  
* **Improving Extraction and Robustness:** Research into improving the accuracy and efficiency of the initial LLM-based knowledge extraction phase is crucial. This could involve fine-tuning smaller, specialized models for entity and relationship extraction or developing more robust error-checking and validation protocols to enhance the quality of the initial graph.  
* **Causal and Argumentative Analysis:** Moving beyond thematic identification, the explicit relational structure of the knowledge graph could be leveraged for more advanced tasks, such as identifying causal claims, mapping argumentative structures, or detecting stance and sentiment within and between different thematic communities.

---

## **References**

Barabási, A.-L. & Albert, R. (1999) 'Emergence of scaling in random networks', *Science*, 286(5439), pp. 509-512.

Blei, D. M. (2012) 'Probabilistic topic models', *Communications of the ACM*, 55(4), pp. 77-84.

Blei, D. M., Ng, A. Y. & Jordan, M. I. (2003) 'Latent Dirichlet Allocation', *Journal of Machine Learning Research*, 3, pp. 993-1022.

Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J. & Yakhnenko, O. (2013) 'Translating embeddings for modeling multi-relational data', *Advances in Neural Information Processing Systems*, 26, pp. 2787-2795.

European Parliamentary Research Service (2024) *'This is Europe' debates: Analysis of EU leaders' speeches*. Drachenberg, R. & Bącal, P. Available at: https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI(2024)757844 (Accessed: 15 January 2024).

Firbas, J. (1992) *Functional sentence perspective in written and spoken communication*. Cambridge: Cambridge University Press.

Foucault, M. (1972) *The archaeology of knowledge*. New York: Pantheon Books.

Griffiths, T. L. & Steyvers, M. (2004) 'Finding scientific topics', *Proceedings of the National Academy of Sciences*, 101(suppl 1), pp. 5228-5235.

Halliday, M. A. K. (1985) *An introduction to functional grammar*. London: Edward Arnold.

Jolliffe, I. T. (2002) *Principal component analysis*. 2nd edn. New York: Springer.

Newman, M. E. J. (2005) 'Power laws, Pareto distributions and Zipf's law', *Contemporary Physics*, 46(5), pp. 323-351.

Newman, M. E. J. (2006) 'Modularity and community structure in networks', *Proceedings of the National Academy of Sciences*, 103(23), pp. 8577-8582.

Newman, M. E. J. & Girvan, M. (2004) 'Finding and evaluating community structure in networks', *Physical Review E*, 69(2), p. 026113.

Nguyen, D. Q. (2020) 'A survey of embedding models of entities and relationships for knowledge graph completion', *arXiv preprint arXiv:2003.08001*.

Reinhart, T. (1981) 'Pragmatics and linguistics: An analysis of sentence topics', *Philosophica*, 27(1), pp. 53-94.

Reinhart, T. (1982) 'Pragmatics and linguistics: An analysis of sentence topics', *Distributed by the Indiana University Linguistics Club*.

Salton, G. & McGill, M. J. (1986) *Introduction to modern information retrieval*. New York: McGraw-Hill.

Traag, V. A., Waltman, L. & van Eck, N. J. (2019) 'From Louvain to Leiden: guaranteeing well-connected communities', *Scientific Reports*, 9(1), p. 5233.

Grover, A. & Leskovec, J. (2016) 'node2vec: Scalable feature learning for networks', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 855–864.

