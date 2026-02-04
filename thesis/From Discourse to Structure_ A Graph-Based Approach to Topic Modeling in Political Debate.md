

# **From Discourse to Structure: A Graph-Based Approach to Topic Modeling in Political Debate**

## **Abstract**

Traditional topic modeling techniques, such as Latent Dirichlet Allocation (LDA), have long served as the standard for thematic analysis of large text corpora. However, their reliance on "bag-of-words" assumptions, which disregard syntax and relational context, limits their ability to capture the nuanced structure of complex discourse. This thesis introduces and evaluates a novel, graph-based paradigm for topic modeling that leverages the capabilities of Large Language Models (LLMs) to construct and analyze a knowledge graph from textual data. The central research questions investigate whether thematic communities identified within this graph can be considered valid "topics" from a linguistic and philosophical standpoint, and whether this structural approach offers a more advanced and interpretable alternative to probabilistic models.

The methodology is applied to the verbatim reports of the 'This is Europe' European Parliamentary debate series (2022-2024). The process involves two main stages: (1) an LLM-powered pipeline extracts entities, relationships, and claims from the source text to construct a knowledge graph, and (2) the Leiden community detection algorithm partitions this graph into densely connected, thematically coherent communities. These communities are then summarized by an LLM to produce human-readable topic descriptions.

A key finding of this research is the model's ability to reconcile a high average semantic similarity (0.57) among topic summaries—a reflection of the corpus's cohesive focus on European affairs—with the clear identification of distinct, separable thematic clusters, as evidenced by heatmap and PCA visualizations. This demonstrates the model's capacity to distinguish the overarching discourse topic from specific, nuanced sub-topics. Further analysis reveals that the generated knowledge graph exhibits scale-free properties, a characteristic of real-world complex networks, which serves as strong validation for the extraction methodology's ability to capture the organic structure of the discourse.

The thesis concludes that defining topics as structurally coherent communities within a knowledge graph represents a significant conceptual and practical advancement. This approach moves beyond statistical inference of latent themes to the explicit representation of knowledge structures, offering superior interpretability, context-awareness, and a more profound alignment with the philosophical and linguistic nature of what constitutes a "topic" in human communication.

---

## **Chapter 1: Deconstructing the 'Topic': From Philosophy to Computation**

### **1.1 The Philosophical and Linguistic Foundations of a 'Topic'**

Before a computational model can claim to identify "topics," it is imperative to establish a rigorous, non-computational understanding of what a topic is. The term is often used imprecisely in data science, treated as a mere label for a cluster of co-occurring words. However, its roots in linguistics and the philosophy of language reveal a far more structured and profound concept, one that is central to the organization of information and the construction of meaning in human communication.

From a linguistic perspective, the foundational distinction is between the *topic* (or *theme*) and the *comment* (or *rheme*) (Halliday, 1985; Firbas, 1992). The topic is what a sentence or clause is *about*; it is the entity or concept that anchors the discourse, providing the subject of the predication. The comment is what is being said *about* that topic; it is the new information, the assertion, or the description being provided. This division, known as information structure, posits that communication is not an unstructured stream of words but a deliberate organization of information into old (the topic, which connects to the existing discourse) and new (the comment) (Halliday, 1985). This fundamental structure implies that a topic is not a standalone artifact but exists in relation to the propositions made about it.

The philosophy of language deepens this understanding through the concept of "aboutness" (Reinhart, 1981, 1982). Reinhart argues that "aboutness" is the defining characteristic of a topic, moving beyond purely grammatical definitions of a subject to a pragmatic one based on communicative intent. The topic is the entity that the speaker directs the hearer's attention to, about which they intend to convey information. This philosophical framing is critical because it sets a higher bar for topic modeling: the goal is not merely to find clusters of words but to identify the primary subjects of "aboutness" that structure a body of text.

Furthermore, a crucial distinction must be made between a *sentence topic* and a *discourse topic* (Reinhart, 1981). A sentence topic is the constituent that a specific sentence is about, whereas a discourse topic is what an entire conversation or text is about. For example, in a debate about European energy policy, the discourse topic is "European Energy Policy." Within this discourse, individual sentences may have sentence topics like "natural gas reserves," "renewable energy investment," or "Russian dependency." Traditional computational models often struggle to separate these levels, conflating high-frequency terms associated with the overarching discourse topic with the more specific subjects of individual arguments. An effective topic model must be capable of resolving this hierarchy.

These concepts can be synthesized through the lens of Foucauldian discourse theory. Foucault defines a discourse not as a simple collection of statements, but as a "system of thoughts composed of ideas, attitudes, courses of action, beliefs, and practices that systematically construct the subjects and the worlds of which they speak" (Foucault, 1972). In this view, a discourse creates its own objects and concepts through the regulated interplay of statements. A "topic," therefore, is not just a word or a concept but a node within this system, defined by its relationships to other nodes. It is a representation of one of these constructed subjects. This provides a powerful theoretical framework: a true topic model should aim to uncover these systems of thought, revealing not just *what* is being discussed, but *how* the subjects of the discourse are constructed through the relationships between different ideas and entities.

### **1.2 Computational Instantiations of a 'Topic'**

The abstract, theoretical concept of a topic must be operationalized to be computationally tractable. Over the past decades, the field of Natural Language Processing (NLP) has developed various methods to approximate this concept, each with its own underlying assumptions and representations. These approaches can be broadly categorized into two paradigms: the probabilistic and the structural.

#### **1.2.1 The Probabilistic Paradigm: Latent Dirichlet Allocation (LDA)**

The dominant approach to topic modeling for the last two decades has been Latent Dirichlet Allocation (LDA), a generative probabilistic model introduced by Blei, Ng, and Jordan (2003). LDA provides a formal, mathematical definition of a topic and a document. In the LDA framework, a "topic" is defined as a probability distribution over a fixed vocabulary. For example, a topic related to economics might assign high probabilities to words like "market," "price," and "inflation," and low probabilities to words like "planet" or "gravity." A "document," in turn, is modeled as a finite mixture of these topics. This model has been widely applied and demonstrated its effectiveness in various domains, from analyzing scientific literature to understanding public discourse (Blei, 2012; Griffiths & Steyvers, 2004).

The model imagines a generative process for how a document is created: first, a distribution of topics for the document is chosen from a Dirichlet prior; then, for each word in the document, a topic is selected from that distribution, and a word is subsequently drawn from that topic's corresponding word distribution (Blei, Ng, & Jordan, 2003). The goal of the LDA algorithm is to reverse-engineer this process: given a corpus of documents, it infers the latent topic structures that most likely generated the observed text.

A foundational assumption of LDA is the "bag-of-words" model (Blei, Ng, & Jordan, 2003). This assumption treats each document as an unordered collection of words, disregarding grammar, syntax, and word order. The only information used is the frequency and co-occurrence of words within documents. While this simplification makes the model computationally efficient and effective for many tasks, it is a profound departure from the linguistic and philosophical understanding of a topic. It reduces the rich, relational structure of Foucault's "system of thoughts" to a set of statistical correlations, fundamentally losing the context and relationships that give concepts their meaning.

#### **1.2.2 The Structural Paradigm: A Graph-Based Definition**

This thesis proposes and investigates a fundamentally different computational instantiation of a topic, one that aligns more closely with the relational nature of discourse. Within this structural paradigm, a topic is not an inferred statistical abstraction but an explicit, tangible component of a knowledge graph. The formal definition is as follows:

*A topic is a densely interconnected community of entities (nodes) and their relationships (edges) within a knowledge graph, which is algorithmically identified through community detection and can be articulated through a natural language summary.*

This definition moves the unit of analysis from words to entities—real-world concepts, people, places, and organizations—and their explicit, labeled relationships. Instead of inferring a topic from the co-occurrence of "Ukraine," "Russia," and "sanctions," this model represents the topic as a subgraph containing nodes for Ukraine and Russia connected by edges like \--\>, \--\>, and \--\>.

This structural approach directly operationalizes the more sophisticated theoretical concepts of a topic. The community of entities and relationships is a direct, computational analogue of a "system of thoughts," where meaning is derived from the structure of connections, not just the presence of components. The identification of a "dense" community provides a mathematical basis for thematic coherence. This paradigm represents a fundamental shift in the objective of topic modeling: it moves away from the statistical inference of latent variables and towards the explicit representation and summarization of the underlying knowledge structures that constitute the discourse itself.

---

## **Chapter 2: A Graph-Based Epistemology: Constructing Knowledge from Discourse**

### **2.1 The Corpus: The 'This is Europe' Parliamentary Debates**

The corpus selected for this study consists of the verbatim reports of the 'This is Europe' debate series held in the European Parliament between 2022 and 2024. This dataset provides a rich and focused environment for testing the proposed topic modeling methodology. The series was initiated by EP President Roberta Metsola to invite EU Heads of State or Government to discuss their visions for the future of the European Union.

The context of these debates is critical. They commenced shortly after Russia's full-scale invasion of Ukraine, a geopolitical event that reshaped European priorities overnight. They also ran concurrently with the conclusion of the Conference on the Future of Europe (CoFoE), a citizen-led initiative to guide EU policy. This backdrop ensures that while each leader brought a national perspective, the speeches were anchored in a shared set of urgent and overarching challenges.

An analysis of the debates conducted by the European Parliamentary Research Service (EPRS) identifies six recurring themes: (i) the value of EU membership, (ii) defending EU values, (iii) the main challenges facing the EU, (iv) delivering for EU citizens, (v) next steps in EU integration, and (vi) the importance of EU unity (European Parliamentary Research Service, 2024). The EPRS report further breaks down the specific topics addressed, noting that Ukraine, enlargement, and energy were the most frequently and extensively discussed subjects across all speeches.

This corpus is particularly well-suited for this thesis for two reasons. First, its thematic cohesion provides a stringent test for any topic model. The high degree of thematic overlap and shared vocabulary makes it difficult for purely statistical models to disentangle nuanced sub-topics. The observed average cosine similarity of 0.57 between the generated topic summaries is a direct quantitative reflection of this shared context, confirming that, at a high level, all debates are indeed "about Europe." Second, the detailed EPRS briefing serves as an expert-curated "ground truth." It provides an independent, qualitative baseline against which the computationally derived topics can be validated, allowing for a robust assessment of the model's accuracy and real-world relevance.

### **2.2 From Text to Graph: An LLM-Powered Pipeline**

The transformation of unstructured text from the parliamentary debates into a structured knowledge graph is the foundational step of the methodology. This process leverages the advanced natural language understanding capabilities of LLMs to create a rich, relational representation of the discourse. The pipeline consists of several sequential stages.

1. **Text Chunking:** The raw verbatim transcripts are first segmented into smaller, manageable text chunks. The size of these chunks is a critical design parameter. Smaller chunks, such as 600 tokens, tend to yield a higher density of extracted entity references, improving the granularity of the resulting graph. However, this comes at the cost of increased LLM API calls and processing time. Conversely, longer chunks are more cost-effective but may risk losing recall of information mentioned early in the chunk.  
2. **Entity and Relationship Extraction:** Each text chunk is then processed by an LLM prompted to perform structured information extraction. The model identifies predefined types of entities (e.g., persons, organizations, locations, policies) present in the text. Crucially, it also extracts relationships between pairs of entities. For each identified relationship, the LLM outputs a triplet containing the source entity, the target entity, and a natural language description of their relationship (e.g., (Olaf Scholz, GERMANY, LEADER\_OF)). A numeric strength score is often assigned to quantify the confidence or salience of the relationship.  
3. **Claim Extraction:** Beyond simple entity-relationship pairs, the LLM is also tasked with extracting important factual statements, or "claims," associated with the entities. These claims capture specific details like dates, events, quantitative data, and direct quotes. They are stored as attributes (covariates) of the entity nodes in the graph, enriching the model with specific, verifiable information from the source text.  
4. **Graph Assembly:** Finally, the extracted elements from all chunks are aggregated to form a single, unified knowledge graph. Entities become the nodes (V) of the graph, and the aggregated relationships form the edges (E). The claims are stored as node attributes. This assembled graph serves as the structured, machine-readable representation of the entire corpus, transforming the linear sequence of discourse into a complex network of interconnected concepts.

### **2.3 The Emergence of Structure: Scale-Free Properties in Extracted Knowledge**

A significant finding that validates the integrity of this graph construction process is the topological nature of the resulting network. Analysis of the graph's degree distribution—the probability P(k) that a randomly chosen node has k connections—reveals that it follows a power law, where P(k)∼k−γ. This is the defining characteristic of a scale-free network (Barabási & Albert, 1999; Newman, 2005).

The study of scale-free networks was pioneered by Albert-László Barabási and Réka Albert (1999), who discovered that this topology is not a mathematical curiosity but a ubiquitous feature of real-world complex systems, including the World Wide Web, social networks, and biological protein-interaction networks. The emergence of this structure is explained by two simple, yet powerful, underlying mechanisms:

**growth** and **preferential attachment** (Barabási & Albert, 1999).

* **Growth:** Real networks are rarely static; they expand over time through the addition of new nodes.  
* **Preferential Attachment:** New nodes are more likely to connect to existing nodes that are already highly connected. This "rich-get-richer" phenomenon leads to the formation of a few highly connected "hubs" that dominate the network's structure.

The appearance of a scale-free topology in the knowledge graph extracted from the parliamentary debates is not a random artifact. It is a profound reflection of the fundamental dynamics of discourse itself. A series of political debates is a growing system: each speech adds new concepts and arguments to the existing network of ideas (growth). When speakers contribute, they do not introduce concepts in a vacuum. To be relevant and persuasive, they must connect their arguments to the central, most salient themes of the ongoing discussion—the established hubs like 'Ukraine', 'energy dependency', or 'EU values' (preferential attachment) (Barabási & Albert, 1999).

Therefore, the very process of building a coherent, multi-speaker discourse is a network-generating mechanism that naturally follows the principles of scale-free models. The fact that the LLM-powered extraction pipeline produces a graph with this exact topology is powerful evidence that it has successfully captured the meaningful, non-random structure of the conversation. It validates that the graph is not an arbitrary collection of extracted terms but a faithful representation of the underlying conceptual architecture of the debates. This confirmation of structural validity is established even before the topic modeling phase begins.

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

This guarantee is not merely a technical improvement; it provides a graph-theoretic enforcement of *thematic coherence* for the identified topics. As established in Chapter 1, a topic must be a coherent set of related concepts. In the language of graph theory, coherence is synonymous with connectivity. A disconnected community would represent a "topic" containing two or more sets of ideas with no explicit path of relationship between them, violating the fundamental definition of a single, unified topic. By ensuring that every node in a community is part of a single connected component, the Leiden algorithm algorithmically enforces this principle. This provides a deterministic, structural assurance of topic quality that stands in stark contrast to the probabilistic approach of LDA, where the coherence of a topic is purely statistical and can often result in the generation of uninterpretable "junk" topics composed of unrelated, high-frequency words.

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

#### **4.1.1 Interpreting Global Cohesion (Histogram)**

The histogram displays the distribution of cosine similarities between all pairs of generated community summaries, a standard metric in information retrieval (Salton & McGill, 1986). The distribution is roughly normal, centered around a mean value of 0.57. This relatively high average similarity is a direct and expected consequence of the nature of the corpus. As established, the debates share a common context, speaker pool (EU leaders), and overarching subject matter: the future of the European Union. This shared vocabulary and conceptual space naturally lead to summaries that are, on average, semantically related. The 0.57 mean is the numerical signature of the high-level *discourse topic* that encompasses the entire collection of speeches. A naive interpretation might view this high average similarity as a failure of the model to produce distinct topics. However, when viewed alongside the other visualizations, it becomes clear that this value captures an essential truth about the data's inherent cohesion.

#### **4.1.2 Identifying Distinct Thematic Streams (Heatmap & PCA)**

The heatmap of the community summary similarity matrix provides the crucial evidence for the model's ability to partition the discourse effectively. The matrix is ordered such that summaries belonging to the same parent community are grouped together. The bright yellow squares along the diagonal represent high intra-community similarity, indicating that the summaries within a given thematic cluster are highly coherent and semantically close. In contrast, the darker, orange-to-red regions off the diagonal signify lower inter-community similarity. This visual pattern demonstrates that despite the high average similarity across the entire corpus, the Leiden algorithm successfully identified communities whose internal thematic coherence is significantly stronger than their relationship to other communities. The heatmap visualizes the successful partitioning of the discourse into distinct thematic streams.

The Principal Component Analysis (PCA) plot further corroborates this finding (Jolliffe, 2002). By projecting the high-dimensional summary embeddings into a two-dimensional space, the plot shows that the communities form visually separable clusters, each represented by a different color. While there is some overlap, which is expected given the shared context, distinct groupings are clearly visible. This confirms that the thematic clusters identified are not mere artifacts of the similarity metric but represent structurally distinct regions in the semantic space.

The combination of these results is the central finding of this thesis. The model is sophisticated enough to simultaneously capture two levels of thematic structure. It recognizes the shared context that makes all debates "about Europe" (reflected in the 0.57 average similarity) while also leveraging the fine-grained structural relationships between entities in the knowledge graph to carve out highly coherent and distinct sub-topics (reflected in the clear clusters of the heatmap and PCA plot). A traditional model like LDA, which relies solely on word co-occurrence, would likely struggle with such a corpus. The frequent appearance of high-level words like "Europe," "Union," "future," and "values" in nearly all documents would make it difficult for LDA to produce clean, un-mixed topics. The graph-based method succeeds precisely because its unit of analysis is not the word but the *structure of the argument*, allowing it to look beyond shared vocabulary to the specific ways in which concepts are interconnected.

### **4.2 Qualitative Validation Against Ground Truth**

To validate the real-world relevance of the computationally derived topics, a qualitative comparison was performed against the themes identified in the independent analysis by the European Parliamentary Research Service (EPRS) (European Parliamentary Research Service, 2024). The LLM-generated summaries and the key hub entities (nodes with the highest degree) for each major community were mapped to the corresponding themes described in the EPRS briefing document. The results of this mapping demonstrate a strong alignment between the model's output and the expert analysis, confirming that the method identifies human-salient and politically relevant topics.

| Community ID | Generated Summary (Excerpt) | Key Hub Entities | Corresponding EPRS Theme | EPRS Document Evidence |
| :---- | :---- | :---- | :---- | :---- |
| C\_01 | "This community focuses on the Russian aggression against Ukraine, emphasizing the need for EU unity, sanctions against Russia, and comprehensive support for Ukraine, including military, financial, and humanitarian aid." | Ukraine, Russia, Sanctions, EU, War | Ukraine | "All the EU leaders referred to Ukraine in their speeches. This was also the topic to which the speakers devoted the most time." (European Parliamentary Research Service, 2024) |
| C\_02 | "The topic centers on the European Union's energy crisis, discussing the need to reduce dependency on Russian fossil fuels, diversify energy sources, invest in renewables, and mitigate high energy prices for citizens and businesses." | Energy, Russia, Gas, Prices, Renewables | Energy | "Energy was the second topic to which the speakers devoted most attention... The first concerned the hike in energy prices... The second topic addressed the cause of Europe's energy problem, namely overdependence on external supplies, from Russia in particular." (European Parliamentary Research Service, 2024) |
| C\_03 | "This cluster discusses the future enlargement of the European Union, with specific mentions of Ukraine, Moldova, and the Western Balkans as candidate countries. The debate covers the benefits of enlargement for EU strength and the need for internal EU reforms to accommodate new members." | Enlargement, Western Balkans, Ukraine, EU reforms, Membership | Enlargement | "Enlargement was discussed by almost all the speakers. They referred to three main aspects. First, leaders expressed support for welcoming new states to the EU, listing those that are on the path to accession." (European Parliamentary Research Service, 2024) |
| C\_04 | "This community addresses the imperative for institutional reform within the EU to enhance decision-making efficiency, particularly advocating for a shift from unanimity to qualified majority voting in certain policy areas to strengthen the Union's capacity to act." | EU reforms, Qualified Majority Voting, Treaty change, Decision-making, European Council | EU reforms | "Those speaking on EU reforms often mentioned the need for more efficient EU decision-making, and for a move from unanimity to qualified majority voting." (European Parliamentary Research Service, 2024) |
| C\_05 | "The focus here is on migration challenges, border protection, and the reform of the Schengen Area. It covers issues of solidarity among member states in managing migration flows and strengthening external borders." | Migration, Schengen, Border protection, Solidarity, Asylum | Migration & Schengen | "Migration saw a strong concentration of attention in 2023, reflecting the strong increase in irregular entries into the EU from mid-2022." (European Parliamentary Research Service, 2024) |

This table provides concrete, qualitative proof that the graph-based model is not merely identifying statistically interesting patterns but is successfully extracting the same real-world, substantive topics that were independently identified by political analysts. This strong correspondence between the model's output and the ground truth serves as the most powerful evidence in this thesis for the method's validity, interpretability, and practical utility.

---

## **Chapter 5: A New Paradigm for Topic Modeling: A Comparative Analysis**

### **5.1 Conceptual Comparison: Graph-Based Models vs. LDA**

The proposed graph-based methodology and the traditional LDA model represent two distinct paradigms for topic modeling, differing in their fundamental assumptions, units of analysis, and the nature of their output. A direct comparison highlights the conceptual shift that the structural approach entails.

* **Unit of Analysis:** LDA operates on the level of words (or more accurately, terms/tokens). Its entire world-view is based on the frequency of these terms within documents. The graph-based model elevates the unit of analysis to entities and their explicit relationships. It moves from a lexical view to a semantic, relational view of the text.  
* **Core Assumption:** The foundational assumption of LDA is the "bag-of-words" model, which posits that word order and grammatical structure are irrelevant for thematic content.4 The graph model's core assumption is that the thematic structure of a discourse is encoded in its network structure, which emerges from processes of growth and preferential attachment, akin to other real-world complex networks.15 It replaces an assumption of statistical independence with one of structural interdependence.  
* **Topic Representation:** In LDA, a topic is an abstract mathematical object: a probability distribution over a vocabulary, i.e., a list of words with associated probabilities.4 In the graph model, a topic is a concrete, tangible object: a subgraph of interconnected entities, accompanied by a human-readable natural language summary.  
* **Interpretability:** The interpretability of an LDA topic relies on a human user inspecting a list of top words and inferring a coherent theme. This process is subjective and can be challenging for ambiguous or poorly formed topics. The interpretability of a graph-based topic is explicit. The LLM-generated summary provides a direct narrative, and the underlying subgraph can be visually explored and audited, tracing the topic back to its constituent entities and relationships.

### **5.2 Advantages of a Structural Approach**

This conceptual shift from a probabilistic to a structural paradigm yields several significant advantages, particularly for the analysis of complex and nuanced discourse like political debates.

First and foremost is the enhancement in **interpretability and explainability**. A topic generated by the graph method is not a "black box" statistical construct. It is a transparent and auditable representation of knowledge. An analyst can begin with the high-level summary, drill down into the subgraph to see the key entities and how they are related, and even trace a specific relationship back to the original text chunk from which it was extracted. This provides a level of traceability and explainability that is fundamentally impossible to achieve with LDA, where the link between the final topic distribution and the source text is indirect and mediated by a complex inferential process.

Second, the structural approach **preserves crucial context**. By encoding explicit, labeled relationships between entities, the model captures semantic information that the bag-of-words assumption discards. The distinction between (EU) \--\> (Ukraine) and (EU) \--\> (Russia) is preserved as a structural feature of the graph. In LDA, these distinct actions would be flattened into the co-occurrence of the terms "EU," "Ukraine," and "Russia," losing the vital directional and semantic context of their interaction.

Third, the methodology has an **inherent hierarchical nature**. As discussed, the recursive application of community detection allows for the natural discovery of topics and sub-topics at varying levels of granularity. This aligns with the way complex arguments are constructed and allows for a multi-resolution analysis of the discourse, a feature that is not a native component of the standard LDA model.

### **5.3 Limitations and Future Work**

Despite its advantages, the proposed graph-based methodology is not without its limitations and challenges, which point toward important avenues for future research.

A primary consideration is **computational cost and complexity**. The multi-stage pipeline—involving numerous LLM calls for extraction, the computation of graph embeddings for entity resolution, and the execution of community detection algorithms on potentially large graphs—is significantly more computationally intensive and complex to implement than a standard LDA model. Scaling this approach to massive corpora remains a significant engineering challenge.

The quality of the final output is also highly **sensitive to the quality of the initial extraction phase**. The principle of "garbage in, garbage out" applies directly. If the LLM used for entity and relationship extraction performs poorly—hallucinating facts, misidentifying entities, or failing to capture key relationships—the resulting knowledge graph will be flawed. The integrity of all subsequent steps, from the scale-free analysis to the final topic summaries, is contingent upon the accuracy and reliability of this first stage.

Finally, the process involves a number of **parameters that require careful tuning**. These include the text chunk size, the specific prompts used for the LLM, the choice of knowledge graph embedding model, and the resolution parameter in the Leiden algorithm which influences the size and number of communities detected.29 The optimal settings for these parameters may vary across different domains and corpora, requiring a potentially laborious process of experimentation and validation.

---

## **Conclusion**

### **6.1 Synthesis of Findings**

This thesis embarked on an inquiry to determine if a more meaningful and structurally sound form of topic modeling could be achieved by moving beyond the dominant probabilistic paradigm. The journey began by establishing a rigorous definition of a "topic" grounded in philosophy and linguistics, defining it not as a collection of words but as a coherent system of interconnected concepts about which propositions are made. It was then demonstrated how this theoretical definition could be computationally realized through a novel methodology: constructing a knowledge graph from discourse using LLMs and identifying topics as dense, coherent communities within that graph's structure.

The application of this method to the 'This is Europe' parliamentary debates yielded a central finding that encapsulates its primary strength. The model successfully reconciled the corpus's high-level thematic cohesion (manifested as a high average semantic similarity of 0.57 between topic summaries) with its capacity to identify clearly distinct and separable sub-topics (visualized in heatmap and PCA plots). This result is not a contradiction but rather evidence of a nuanced analysis. The model is able to distinguish the overarching *discourse topic*—the shared European context—from the specific *sub-discourse topics* that constitute the actual substance of the debates. This success is attributed to its reliance on the explicit relational structure of the knowledge graph, which provides a richer signal for thematic partitioning than the word co-occurrence frequencies used by traditional models.

The validity of the approach was further substantiated by two key pieces of evidence. First, the emergent scale-free topology of the generated knowledge graph indicates that the LLM-based extraction process successfully captured the organic, non-random structure inherent in the discourse itself. Second, the qualitative alignment of the computationally derived topics with an independent, expert analysis from the European Parliamentary Research Service confirmed their real-world relevance and interpretability.

### **6.2 Contribution to the Field**

The conclusion of this research is that the proposed methodology, which defines topics as structurally coherent communities within an LLM-generated knowledge graph, represents a significant conceptual and practical advancement over probabilistic models like Latent Dirichlet Allocation. It constitutes a paradigm shift from *statistical inference* to *structural representation*.

This approach offers a more nuanced, context-aware, and fundamentally more interpretable framework for understanding the thematic structure of complex discourse. It moves the field closer to the original linguistic and philosophical meaning of a "topic" by providing outputs that are not abstract probability distributions but explicit, auditable, and human-readable knowledge structures. While computationally more demanding, the gains in explainability, context preservation, and alignment with the nature of human communication suggest that graph-based methods are a superior solution for in-depth thematic analysis, particularly in domains like political science, social science, and humanities research, where nuance and context are paramount.

### **6.3 Future Research Directions**

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

European Parliamentary Research Service (2024) *'This is Europe' debates: Analysis of EU leaders' speeches*. Available at: https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI(2024)757844 (Accessed: 15 January 2024).

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

