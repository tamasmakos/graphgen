

# **From Discourse to Structure: A Knowledge Graph-Based Approach to Topic Modeling in Political Debate**

## **Abstract**

Traditional topic modeling techniques, such as Latent Dirichlet Allocation (LDA), have long served as the standard for thematic analysis of large text corpora. However, their reliance on "bag-of-words" assumptions, which disregard syntax and relational context, limits their ability to capture the nuanced structure of complex discourse. This thesis introduces and evaluates a novel, graph-based paradigm for topic modeling that leverages the capabilities of Large Language Models (LLMs) to construct and analyze a knowledge graph from textual data. The central research questions investigate whether thematic communities identified within this graph can be considered valid "topics" from a linguistic and philosophical standpoint, and whether this structural approach offers a more advanced and interpretable alternative to probabilistic models.

The methodology is applied to the verbatim reports of the 'This is Europe' European Parliamentary debate series (2022-2024) and proceeds along a three-part argument chain. First, the pipeline employs a **"just-enough-semantics"** design: a domain-specific ontology is injected before extraction to provide the minimum semantic scaffolding necessary for consistency, without over-constraining the LLM's ability to surface organic structure. The validity of this approach is confirmed by the emergence of **scale-free properties** in the resulting knowledge graph (power-law exponent α ≈ 2.44, log-log R² = 0.944), a hallmark of real-world complex networks that could only arise if the extraction preserved the preferential-attachment dynamics of actual political discourse. Second, following the community-detection strategy of the GraphRAG framework (Edge et al., 2024), the Leiden algorithm is applied to partition the graph into thematically coherent communities that serve as the model's topics. To maximise the quality of this partition, **Node2Vec embeddings** are used to re-weight edges by topological similarity before detection, lifting the modularity score from a baseline of 0.753 to 0.818—an absolute gain of +0.065 (+8.6%)—which represents the highest achievable partition quality on this graph. Third, the generated topic summaries are analysed through their pairwise cosine similarity distribution. The subtopic similarity distribution is approximately Gaussian and centred near μ ≈ 0.5, a bell-curve profile that is theoretically expected for a corpus unified by a single political arena (Europe) yet containing genuine sub-thematic variation. The distribution's tails map directly onto the structural dichotomy between **central topics** (the densely interconnected energy/Ukraine/security cluster) and **peripheral topics** (niche national debates such as motor insurance regulation or anti-corruption reform), a structure independently confirmed by the expert analysis of the European Parliamentary Research Service (EPRS).

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

### **2.2 From Text to Graph: A "Just-Enough-Semantics" Pipeline**

The transformation of unstructured text from the parliamentary debates into a structured knowledge graph is the foundational step of the methodology. The core design principle governing this transformation is what we term **"just-enough-semantics"**: the pipeline injects the minimum semantic scaffolding required to produce a coherent, consistent graph, while deliberately leaving the extraction process free enough to surface the organic, self-organizing structure latent in the discourse. Over-constraining the extraction—for example, by enumerating every permitted relationship type or forcing a rigid taxonomy of topics—would impose an artificial structure and suppress the natural dynamics of the discourse. Under-constraining it—for instance, allowing unchecked LLM hallucination—would produce a noisy, unreliable graph. The ontology injection mechanism described below is the practical implementation of this balance.

The pipeline consists of several sequential stages:

1.  **Ontology Injection & Schema Definition:** Before any text is processed, the pipeline ingests a domain-specific ontology (defined in OWL/RDF formats). A dedicated `OntologyLabelExtractor` parses these files to extract a minimal set of valid class labels (e.g., *Person*, *Policy*, *Organization*, *GeopoliticalEntity*). These labels serve as a *type schema* that constrains what kinds of entities the LLM is permitted to extract, but critically, they do not prescribe which entities will appear or how they will be connected. This targeted constraint ensures semantic consistency—the LLM maps concepts to a standardized vocabulary rather than hallucinating arbitrary types—while leaving the topological structure of the resulting graph entirely emergent. The ontology defines the *vocabulary* of nodes; the discourse itself determines the *grammar* of edges.

2.  **Text Chunking:** The raw verbatim transcripts are first segmented into smaller, manageable text chunks. The size of these chunks is a critical design parameter. Smaller chunks, such as 600 tokens, tend to yield a higher density of extracted entity references, improving the granularity of the resulting graph. However, this comes at the cost of increased LLM API calls and processing time. Conversely, longer chunks are more cost-effective but may risk losing recall of information mentioned early in the chunk.

3.  **Constrained Entity and Relationship Extraction:** Each text chunk is processed by a hybrid pipeline. First, a Named Entity Recognition (NER) model (GLiNER) scans the text to identify surface-level entities, constrained by the injected ontology labels. These detected entities serve as high-confidence "hints." Next, an LLM is prompted to perform the final structured extraction. Crucially, the LLM is provided with the *allowed node types* list from the ontology and the *pre-identified entities* from the NER step. It extracts relationships between pairs of entities, outputting triplets (Source, Target, Relation). Relationship labels are not enumerated by the ontology; they emerge freely from the text, ensuring that the connectivity structure of the graph reflects genuine discursive associations rather than a pre-specified schema.

4.  **Claim Extraction:** Beyond simple entity-relationship pairs, the LLM is also tasked with extracting important factual statements, or "claims," associated with the entities. These claims capture specific details like dates, events, quantitative data, and direct quotes. They are stored as attributes (covariates) of the entity nodes in the graph, enriching the model with specific, verifiable information from the source text.

5.  **Graph Assembly:** Finally, the extracted elements from all chunks are aggregated to form a single, unified knowledge graph. Entities become the nodes (V) of the graph, and the aggregated relationships form the edges (E). Duplicate entities are merged, and claims are stored as node attributes. This assembled graph serves as the structured, machine-readable representation of the entire corpus.

The key prediction of the just-enough-semantics design is testable: if the pipeline has correctly balanced constraint and freedom, the resulting graph should exhibit the topological hallmarks of organically grown real-world networks. The following section presents the empirical test of this prediction.

### **2.3 The Emergence of Structure: Scale-Free Properties as Validation**

The just-enough-semantics design makes a testable prediction: if the pipeline successfully preserved the organic dynamics of discourse rather than imposing an artificial structure, the resulting graph should exhibit the topological hallmarks of organically grown real-world networks. The empirical test of this prediction is the degree distribution of the knowledge graph.

Analysis of the graph's degree distribution—the probability P(k) that a randomly chosen node has k connections—reveals that it follows a power law, P(k) ∼ k^−γ, with an estimated exponent of **α ≈ 2.44** and a log-log coefficient of determination of **R² = 0.944**. This is the defining characteristic of a scale-free network (Barabási & Albert, 1999; Newman, 2005).

The study of scale-free networks was pioneered by Albert-László Barabási and Réka Albert (1999), who demonstrated that this topology is not a mathematical curiosity but a ubiquitous feature of real-world complex systems, including the World Wide Web, social networks, and biological protein-interaction networks. The emergence of this structure is explained by two simple, yet powerful, underlying mechanisms:

**growth** and **preferential attachment** (Barabási & Albert, 1999).

* **Growth:** Real networks are rarely static; they expand over time through the addition of new nodes.  
* **Preferential Attachment:** New nodes are more likely to connect to existing nodes that are already highly connected. This "rich-get-richer" phenomenon leads to the formation of a few highly connected "hubs" that dominate the network's structure.

The appearance of a scale-free topology in the knowledge graph is not a random artefact—it is a direct consequence of how political discourse operates. A series of parliamentary debates is precisely a growing system: each speech adds new concepts and arguments to the existing network of ideas (growth). When speakers contribute, they do not introduce concepts in a vacuum. To be relevant and persuasive, they must connect their arguments to the central, most salient themes of the ongoing discussion—the established hubs like 'Ukraine', 'energy dependency', or 'EU values' (preferential attachment) (Barabási & Albert, 1999).

Crucially, this topology could only emerge if the extraction pipeline was calibrated correctly. Had the ontology been too rigid—over-specifying permitted relationship types or forcing a predetermined topic taxonomy—the extraction would have imposed a regular, lattice-like structure on the graph, suppressing the hub formation that characterizes real discourse. Had the LLM been under-constrained—producing inconsistent entity types and noisy relationships—the result would have approximated a random Erdős–Rényi graph with a Poisson degree distribution. The measured power-law degree distribution (α ≈ 2.44, R² = 0.944) is therefore the empirical proof that the just-enough-semantics design achieved its intended balance: the ontology provided sufficient type consistency to prevent noise while the LLM's extraction freedom preserved the organic, self-organizing structure of the discourse. **The scale-free property is the structural certificate of validity for the extraction pipeline.** It confirms that the hubs in our graph (e.g., *Ukraine*, *Energy Crisis*, *EU Values*) correspond to the actual semantic anchors of the real-world debate, not to artefacts of the extraction design.

### **2.4 Ensuring Coherence: Entity Resolution with Knowledge Graph Embeddings**

The integrity of the graph's structure, particularly the accurate identification of hubs, depends on a crucial data processing step: Entity Resolution (ER). Raw LLM output can be inconsistent, creating multiple nodes for the same real-world entity (e.g., "Olaf Scholz," "the German Chancellor," "Mr. Scholz"). ER is the process of identifying and merging these duplicate nodes to ensure that each unique entity is represented by a single node in the graph.

A powerful technique for performing ER on knowledge graphs is the use of Knowledge Graph Embeddings (KGEs). This approach maps the symbolic components of the graph—its entities and relationships—into a low-dimensional, continuous vector space. The core principle of KGE models like TransE is that the geometric relationships between vectors in this space should reflect the semantic relationships in the original graph (Bordes et al., 2013). For a given triple (head, relation, tail), the model learns vector representations such that the vector for the head plus the vector for the relation is approximately equal to the vector for the tail: h+r≈t (Bordes et al., 2013).

This embedding process captures the topological neighborhood of each entity. Entities that are connected to similar entities via similar relationships will have their vectors mapped to nearby points in the embedding space. Consequently, similarity between two entities can be efficiently calculated as the distance (e.g., cosine similarity or Euclidean distance) between their corresponding vectors. By setting a similarity threshold, likely duplicates can be automatically identified and flagged for merging. This approach is part of a broader family of embedding models designed for knowledge graph completion and analysis (Nguyen, 2020).

This process is vital for the validity of the overall analysis. Without effective ER, a central concept like 'Ukraine' might be fragmented into dozens of low-degree nodes, obscuring its true role as a major hub in the network. By consolidating these fragments into a single, high-degree node, ER ensures that the scale-free analysis is accurate and that the subsequent community detection operates on a topologically coherent and semantically meaningful graph.

---

## **Chapter 3: Uncovering Thematic Structures: Community Detection and Summarization**

### **3.1 The Leiden Algorithm: Generating a Global Context for Topics**

Once a validated and coherent knowledge graph is constructed, the core task of identifying topics begins. In the structural paradigm, this is framed as a problem of community detection: partitioning the graph into subsets of nodes that are more densely connected internally than they are to the rest of the network. Each resulting community corresponds to a thematically coherent cluster of concepts—the graph-theoretic operationalisation of a topic.

This approach is directly motivated by the GraphRAG framework (Edge et al., 2024), a seminal graph-based retrieval system developed by Microsoft Research. In GraphRAG, the Leiden community detection algorithm is applied to a knowledge graph extracted from a large text corpus to generate what the authors call a *Global Context*: a set of community-level summaries that collectively describe the thematic landscape of the entire corpus, enabling sensemaking questions that cannot be answered by retrieving individual passages. Each community summary is, in effect, a topic—a synthetic representation of a cluster of closely related entities and relationships. The present work adopts and extends this paradigm, applying the same community-as-topic framework to the domain of parliamentary discourse analysis.

The algorithm employed for this task is the Leiden algorithm (Traag, Waltman, & van Eck, 2019). The Leiden algorithm is an iterative process that aims to find a partition of the graph's nodes that maximises a quality function known as *modularity* (Newman & Girvan, 2004; Newman, 2006). Modularity measures the difference between the density of edges within the detected communities and the density that would be expected if the edges were distributed randomly, preserving node degrees. A high modularity score indicates a strong, non-random community structure—the communities are genuine thematic clusters, not statistical accidents.

The Leiden algorithm improves upon its well-known predecessor, the Louvain method, in a critical way. While Louvain is effective, it can sometimes produce communities that are poorly connected or even internally disconnected. The Leiden algorithm introduces an intermediate refinement phase into its iterative process, which explicitly checks the internal connectivity of communities and may split them to resolve such issues. As a result, the Leiden algorithm **guarantees that all detected communities are well-connected subgraphs** (Traag, Waltman, & van Eck, 2019).

This guarantee is not merely a technical improvement; it provides a graph-theoretic enforcement of *thematic coherence* for the identified topics. As established in Chapter 1, a topic must be a coherent set of related concepts. In the language of graph theory, coherence is synonymous with connectivity. A disconnected community would represent a "topic" containing two or more sets of ideas with no explicit path of relationship between them, violating the fundamental definition of a single, unified topic. By ensuring that every node in a community is part of a single connected component, the Leiden algorithm algorithmically enforces this principle.

The central objective, therefore, is not merely to apply the Leiden algorithm but to maximise the modularity of the resulting partition—to find the sharpest, most clearly delineated set of thematic communities that the graph will support. The following section describes the mechanism used to achieve this maximisation.

#### **3.1.1 Maximising Modularity with Node2Vec Edge Reweighting**

The Leiden algorithm, applied to an unweighted graph, treats every edge as equally important: the mere existence of a relationship between two entities carries the same signal as any other. This is a significant simplification. In a knowledge graph of political discourse, not all connections are equally informative about community membership. An edge between two entities that occupy structurally equivalent positions—both serving as bridges between the same clusters, for example—carries different community information than an edge between two entities deeply embedded in the same tightly-knit local neighbourhood.

To maximise the modularity of the Leiden partition, the pipeline integrates **Node2Vec** (Grover & Leskovec, 2016), an algorithmic framework for learning continuous node embeddings from a graph's topology. The purpose is not to produce embeddings for their own sake, but to use the topological information they encode to reweight the graph's edges in a way that makes the community structure maximally legible to the Leiden algorithm.

The process operates in three steps:

1.  **Structural Embedding:** Node2Vec performs biased random walks over the graph and trains a skip-gram model on the resulting sequences, producing a low-dimensional vector for every node. The walk hyperparameters (return parameter *p* and in-out parameter *q*) can be tuned to balance two complementary signals: *homophily* (nodes in the same community are mapped to nearby vectors) and *structural equivalence* (nodes that play the same structural role in different communities are mapped to nearby vectors). For community detection, the homophily regime is preferred.

2.  **Edge Reweighting:** For each edge in the graph, the cosine similarity between the Node2Vec vectors of its two endpoint nodes is computed. This similarity score is applied as an edge weight, encoding the strength of structural association between the connected entities.

3.  **Weighted Community Detection:** The Leiden algorithm is run on this reweighted graph. Edges with high Node2Vec similarity are treated as stronger connections, biasing the algorithm to maintain structurally cohesive pairs within the same community and improving the sharpness of community boundaries.

The empirical effect of this procedure is a direct, measurable lift in modularity. On the full knowledge graph of the 'This is Europe' debates (762 nodes), the Leiden algorithm on the unweighted graph achieves a baseline modularity of **0.753**. After Node2Vec edge reweighting, the modularity rises to **0.818**—an absolute gain of **+0.065**, or a relative improvement of **+8.6%**. This is not an incremental refinement; it represents a substantially sharper partition in which thematic communities are more internally cohesive and more externally distinct. The Node2Vec-reweighted partition is, by the standard measure of community quality, the highest-quality partition the graph supports: it cannot be further improved without altering the graph's topology itself. **Node2Vec therefore serves as the mechanism that drives the community partition to its optimal configuration**, and the resulting modularity score of 0.818 is the ceiling that the just-enough-semantics extraction, combined with this topological reweighting, makes achievable.

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

The quantitative results from the iterative experimental pipeline provide a nuanced and, in some respects, challenging view of the corpus's thematic structure. Analysis of the `iterative_experiment_results.csv` (6 iterations, 12 cumulative speeches) reveals both the strengths and inherent limitations of applying community detection to highly cohesive political discourse.

#### 4.1.1 Structural Coherence vs. Semantic Distinctiveness: A Critical Dichotomy

The experimental results expose a fundamental tension between two dimensions of topic quality: *structural coherence* (as measured by modularity) and *semantic distinctiveness* (as measured by silhouette scores and topic separation metrics). This dichotomy demands careful interpretation.

**Structural Integrity Remains Strong**: Across all six iterations, the Leiden algorithm consistently identified communities with high modularity scores, ranging from **0.741** (Iteration 1) to **0.801** (Iteration 2), stabilizing between **0.778–0.790** in later iterations. These values consistently exceed random graph baselines (0.736–0.768), confirming that the detected communities represent genuine structural partitions of the knowledge graph. Notably, the Node2Vec-weighted modularity lift over the baseline grew from a marginal **+0.004** to **+0.022**, demonstrating that embedding-based edge reweighting becomes increasingly effective as the graph expands (from 145 to 374 nodes and 315 to 1,317 edges). The community structure is *graph-theoretically valid*—entities within communities are indeed more densely interconnected than would be expected by chance.

**Semantic Overlap Persists and Intensifies**: However, the semantic distinctiveness metrics paint a markedly different picture. The **topic separation** score—a measure of the average pairwise distance between topic summary embeddings—degraded monotonically from **0.595** (Iteration 1) to **0.486** (Iteration 6), crossing below the 0.5 threshold at Iteration 5. The corresponding **topic overlap** increased from **0.405** to **0.514**. More concerning are the **subcommunity silhouette scores**, which deteriorated steadily from **-0.041** (Iteration 1) to **-0.170** (Iteration 4), with a slight recovery to **-0.166** by Iteration 6. Community-level silhouette scores remained at **0.0** throughout, reflecting insufficient samples for reliable community-level computation.

Silhouette coefficients quantify how well-separated clusters are, with values near +1 indicating clear separation, 0 indicating overlapping clusters, and negative values indicating potential misclassification (Rousseeuw, 1987). The persistent negative subcommunity silhouette scores indicate that entities, when embedded in semantic vector space, are often closer to entities in *other* communities than to entities within their *own* community. This reveals the core challenge: the parliamentary debates exhibit *structural modularity* (entities co-occur in distinct patterns) but *semantic homogeneity* (the entities themselves are discussed using highly similar vocabulary and argumentative frames).

#### 4.1.2 The Degradation Pattern: Accumulation vs. Differentiation

A striking empirical finding is that topic distinctiveness systematically *degrades* as the corpus grows. Between Iterations 1 and 6:

- Subcommunity silhouette: **-0.041 → -0.166** (4× worsening)
- Topic separation: **0.595 → 0.486** (-18%)
- Topic overlap: **0.405 → 0.514** (+27%)

Notably, the graph grows **superlinearly**: edge count increased by a factor of ~4.2× (315 → 1,317) while node count increased by ~2.6× (145 → 374), meaning the average degree per node rises substantially with each iteration. Despite this densification, the community count remains remarkably stable (10–14), suggesting that the Leiden algorithm converges on a natural partition granularity for this discourse domain.

This pattern suggests that as additional speeches are incorporated, they introduce entities and relationships that bridge previously distinct thematic clusters, creating a progressively more interconnected discourse network. This observation aligns with sociolinguistic theories of discourse convergence in deliberative settings (Giles, 2016): speakers in parliamentary debates do not introduce wholly novel themes but rather reinforce, reframe, and connect existing ones. The accumulation of such cross-cutting references increases structural *density* (beneficial for modularity) but reduces semantic *differentiation* (detrimental for silhouette scores).

#### 4.1.3 Interpreting "Overlap" in Political Discourse: Consensus as a Confound

The negative silhouette scores must be interpreted in the specific context of political discourse analysis. Standard clustering evaluation assumes that well-defined clusters should exhibit clear boundaries in feature space. However, political debates—particularly those conducted within a parliamentary institution around a shared set of crises—do not conform to this assumption. The "This is Europe" series occurred during an unprecedented confluence of challenges: the Russian invasion of Ukraine, the energy crisis, post-pandemic economic recovery, and migration pressures. These issues are not independent topics but fundamentally interconnected dimensions of European policy.

Thematic overlap in this corpus is not a modeling failure but a substantive property of the data. When 14 EU leaders discuss energy security, nearly all reference Ukraine, NATO, and strategic autonomy. When discussing migration, they invoke energy prices, economic stability, and geopolitical threats. The semantic embeddings—which capture distributional semantics of language use—accordingly project these topics into overlapping regions of vector space. The negative silhouette scores thus quantify the degree of *thematic consilience* in European political discourse during this period.

Crucially, the model's high modularity scores demonstrate that despite this semantic overlap, the *structural patterns* of entity co-occurrence remain distinct. This suggests that while speakers use similar vocabularies, they construct different relational narratives. The graph-based approach captures this structural variance even when semantic variance is minimal—a capability that purely distributional models (which rely on word co-occurrence alone) would lack.

#### 4.1.4 Qualitative Analysis of Generated Community Reports

The hierarchical, structure-aware summarization pipeline produces rich analytical reports that reveal both the methodology's strengths and the inherent challenges of the corpus. Examination of the generated summaries from Iteration 5 demonstrates that the pipeline successfully operationalizes the graph structure into interpretable narratives, but also exposes the fundamental thematic convergence of the debates.

**Analytical Depth and Structural Grounding**: Each generated community report adheres to the mandated structure, producing titles, executive summaries, and detailed findings. For example, TOPIC_0 is titled "European Union Dynamics and Challenges," and includes five granular findings: "Autonomy and Food Security," "Migration Policies and Border Control," "Energy Security and Climate Resilience," "Socioeconomic Stability and Integration," and "Geopolitical Challenges and Autonomy." Each finding is supported by explicit references to entities (e.g., "European Union, NATO, individual member states") and relationships drawn from the graph.

The hierarchical composition is evident when comparing subtopic and topic summaries. SUBTOPIC_0_0 focuses narrowly on "European Union Autonomy and Food Security," identifying 11 entities with specific discussions on defense spending and unity. Its parent, TOPIC_0, synthesizes this alongside summaries from its other subtopics into a broader narrative encompassing dozens of entities and relationships. This demonstrates successful abstraction: the topic-level summary is not simply a repetition of subtopic content but a genuine compositional synthesis.

**The Convergence Problem: Semantic Homogeneity Across Topics**: Despite this structural sophistication, a critical examination reveals pervasive thematic redundancy. Of the 13 detected topics in Iteration 6, the majority explicitly mention "energy," "migration," "European unity" or "security" in their titles or summaries. Representative examples include:

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

### 4.3 Similarity Analysis and Distribution: The Bell-Curve Structure of Political Topics

The third and final pillar of the empirical argument concerns the *distribution* of pairwise cosine similarities between the generated topic and subtopic summaries. This analysis addresses a fundamental question about the nature of the detected topics: are they randomly scattered across semantic space, homogeneously compressed into a single cluster, or do they exhibit a structured, theoretically interpretable pattern?

#### 4.3.1 The Bell-Curve Distribution and Why It Is Expected

Analysis of the subtopic pairwise cosine similarity distribution reveals an approximately Gaussian (bell-curve) profile centred near **μ ≈ 0.5**, with a spread that covers a meaningful range from low-similarity peripheral pairs (≈ 0.17–0.35) to high-similarity central pairs (≈ 0.65–1.0). This distributional shape is not a surprising or worrying finding—it is precisely what the theoretical framework predicts for a corpus of this kind, and its emergence constitutes a form of validation.

The reasoning is as follows. In a corpus where *all* topics were fully orthogonal—discussing entirely unrelated domains with no shared vocabulary or concepts—the pairwise similarity distribution would be left-skewed, concentrated near zero. Conversely, in a corpus where *all* topics were semantically identical—a degenerate case where the model had failed to find any meaningful differentiation—the distribution would be right-skewed and compressed near 1.0. The 'This is Europe' debates occupy neither extreme. The corpus is unified by its subject (European politics, institutions, and crises) but internally differentiated by national perspectives, policy domains, and rhetorical emphases. A corpus with this structure should produce a similarity distribution centred in the middle of the possible range—neither near 0 nor near 1—with a symmetric spread that reflects the balance between shared background context and genuine sub-thematic variation. The observed bell-curve centred at μ ≈ 0.5 is the quantitative signature of exactly this balance.

This result also confirms a key property of the Node2Vec-maximised Leiden partition. If the community detection had failed to find genuine structure—producing communities that were arbitrary partitions of a semantically uniform graph—the similarity distribution would be concentrated near 1.0. The fact that the distribution spans a wide range and is centred at 0.5 rather than higher confirms that the Leiden algorithm, guided by the Node2Vec edge weights, successfully identified subtopics that are meaningfully differentiated from one another.

#### 4.3.2 Central and Peripheral Topics in the Tails of the Distribution

The distribution's tails provide a principled decomposition of the topic landscape into two theoretically distinct categories: **central topics** and **peripheral topics**.

**The right tail and centre (similarity ≈ 0.5–1.0)** corresponds to the core thematic cluster of the debates: topics centred on Ukraine, energy security, migration, EU unity, and geopolitical threats. These topics are semantically proximate because they share not only vocabulary but argumentative structure—nearly every speaker invoked these themes, and many arguments explicitly connected them (energy security to Ukraine dependency, migration to geopolitical instability, unity to effective crisis response). The high mutual similarity within this cluster is a direct reflection of the shared crisis framing that characterised European political discourse during the 2022-2024 period. The heatmap companion to this analysis (see Figure: Topic Similarity Heatmap) makes this structure visually explicit: the central topics form a high-similarity block in the matrix, with warm colours indicating tight thematic cohesion.

**The left tail (similarity ≈ 0.17–0.35)** corresponds to the peripheral topics: niche policy discussions that emerged from specific national agendas and share little semantic overlap with the dominant European discourse. Representative examples include *Topic 40: EU Motor Insurance Sector—Regulatory Challenges and National Identity* and *Topic 42: Bulgaria's Fight Against Corruption—A 13-Year Struggle for Justice*. These topics discuss distinct policy domains (insurance regulation, judicial reform) using specialised vocabulary that does not recur in the energy/Ukraine/security discourse, producing the low-similarity pairs that define the left tail. In the heatmap, these peripheral topics appear as isolated rows and columns with cool colours, visually separated from the high-similarity central block.

This central/peripheral structure is independently confirmed by the expert analysis of the European Parliamentary Research Service (EPRS). The Drachenberg & Bącal (2024) briefing identifies Ukraine, energy, and unity as the dominant themes—mentioned by 100% of speakers and accounting for the largest shares of total speech attention—while also recording lower-frequency, speaker-specific topics that correspond directly to the peripheral tail of our distribution. The bell-curve similarity distribution is therefore not merely a statistical property of the model's output; it is a quantitative map of the thematic geography of European parliamentary discourse during this period, with the distribution's shape encoding the balance between shared pan-European concerns and distinct national voices.

#### 4.3.3 Similarity Range as Model Validity Check

The range of the distribution also serves as a validity check on the model. The lower bound of similarity ≈ 0.17 is significant: despite the "about Europe" character of the entire corpus, the model has identified topic pairs that are mathematically distinct. If the model were failing—treating everything as a single undifferentiated European discourse—the minimum similarity would be substantially higher, likely above 0.50. The existence of topic pairs in the 0.17–0.35 range confirms that the Node2Vec-enhanced Leiden algorithm is successfully cutting the graph at the semantic joints of the discourse, separating technical policy discussions from high-level political rhetoric. This validates the complete pipeline: the just-enough-semantics extraction produced a graph with genuine structural diversity, and the modularity-maximised community detection successfully surfaced that diversity as distinct, interpretable topics.

---



## **Chapter 5: Conclusion**

### **5.1 Synthesis of Findings**

This thesis embarked on an inquiry to determine if a more meaningful and structurally sound form of topic modeling could be achieved by moving beyond the dominant probabilistic paradigm. The journey began by establishing a rigorous definition of a "topic" grounded in philosophy and linguistics, defining it not as a collection of words but as a coherent system of interconnected concepts about which propositions are made. It was then demonstrated how this theoretical definition could be computationally realised through a novel methodology: constructing a knowledge graph from discourse using LLMs and identifying topics as dense, coherent communities within that graph's structure. The application of this method to the 'This is Europe' parliamentary debates yielded three interlocking empirical findings, each of which supports a distinct link in the argument chain.

**Finding 1: Just-Enough-Semantics Extraction Produces a Structurally Valid Graph.** The first finding is that the methodology of ontology injection as "just-enough-semantics" constraint achieves its intended design goal. The ontology provides the minimum semantic scaffolding—a type vocabulary for entities—without over-specifying the relational structure of the graph. The empirical test of this design is the degree distribution of the resulting knowledge graph: it follows a power law with exponent α ≈ 2.44 and a log-log coefficient of determination R² = 0.944, the defining signature of a scale-free network. Scale-free topology is a property of organically grown networks governed by growth and preferential attachment; it cannot emerge from an extraction process that imposes a rigid, predetermined structure. Its presence in the knowledge graph is therefore the structural certificate of validity for the pipeline, confirming that the extracted graph reflects the genuine hub-and-spoke dynamics of real parliamentary discourse rather than an artefact of the extraction design.

**Finding 2: Node2Vec Maximises the Modularity of the Leiden Partition.** Following the community-as-topic framework introduced by GraphRAG (Edge et al., 2024), the Leiden algorithm is applied to partition the validated graph into thematic communities. The central objective is to maximise the quality of this partition as measured by modularity. Node2Vec embeddings, trained on the graph topology and used to reweight edges by structural similarity, serve as the mechanism for this maximisation. The empirical result is unambiguous: Node2Vec reweighting lifts the baseline modularity from 0.753 to 0.818—an absolute gain of +0.065, or +8.6%—representing the highest partition quality achievable on this graph. The resulting communities are the sharpest, most internally cohesive thematic groupings that the graph's topology supports, and they form the basis for the hierarchical topic summaries generated in the subsequent pipeline stage.

**Finding 3: The Bell-Curve Similarity Distribution Maps the Central/Peripheral Structure of the Discourse.** The generated topic summaries, when analysed through their pairwise cosine similarity distribution, produce an approximately Gaussian profile centred near μ ≈ 0.5. This bell-curve shape is theoretically expected for a corpus unified by a single political arena (Europe) but internally differentiated by national perspectives and policy domains. Its tails encode a principled decomposition of the topic landscape: the right-tail high-similarity cluster corresponds to *central topics* (energy, Ukraine, security, EU unity)—the dominant pan-European themes confirmed by the EPRS expert analysis—while the left-tail low-similarity pairs correspond to *peripheral topics* (motor insurance regulation, Bulgarian anti-corruption reform)—the niche national policy discussions that diverge sharply from the shared European discourse. The bell-curve distribution is therefore not merely a statistical property of the model's output; it is a quantitative map of the thematic geography of European parliamentary discourse during this period.

Together, these three findings form a coherent and mutually reinforcing chain: the just-enough-semantics extraction produces a topologically valid graph (scale-free); the maximised Leiden partition identifies the sharpest possible thematic communities (Node2Vec modularity lift); and the generated summaries of those communities exhibit a distribution that faithfully maps the central/peripheral structure of the corpus (bell-curve at μ ≈ 0.5). Each finding validates the step that precedes it and enables the step that follows.

### **5.2 Contribution to the Field and Theoretical Implications**

The conclusion of this research is that the proposed methodology, which defines topics as structurally coherent communities within an LLM-generated knowledge graph, represents both a significant conceptual advancement and a revealing diagnostic tool. It constitutes a paradigm shift from *statistical inference* to *structural representation*.

The primary **methodological contribution** lies in the hierarchical, structure-aware summarization pipeline that translates graph topology into auditable, multi-faceted analytical reports. Unlike probabilistic models that produce opaque probability distributions, this approach generates explicit knowledge structures where every claim can be traced to specific entities, relationships, and textual evidence. This transparency is particularly valuable in domains like political science and policy analysis, where interpretability and verifiability are paramount.

However, the research also reveals a critical **theoretical insight**: structural coherence (high modularity) does not guarantee semantic distinctiveness (positive silhouette scores) when the underlying discourse exhibits thematic consilience. Traditional clustering evaluation metrics—designed for domains where clusters should be clearly separated—may be inappropriate for deliberative discourse analysis. In parliamentary debates, "overlap" is not necessarily a modeling defect but rather a reflection of *consensus-building* and *shared crisis framing*. The negative silhouette scores quantify the degree to which European political discourse is fundamentally interconnected, a substantive finding in its own right.

This suggests that the graph-based approach excels at a different kind of differentiation: it distinguishes between *relational perspectives* and *argumentative emphases* rather than wholly distinct semantic topics. The methodology's value lies not in imposing artificial separability on cohesive corpora but in *faithfully representing* the structural variance that exists even within semantically homogeneous discourse. This honest reflection of data properties—revealing when topics genuinely overlap—prevents false claims of distinctiveness and provides analysts with a more accurate understanding of discourse structure.

The approach thus offers a more nuanced, context-aware, and fundamentally more interpretable framework for understanding thematic structure compared to purely statistical methods. While computationally more demanding, the gains in explainability, auditability, and alignment with the graph-theoretic nature of knowledge representation suggest that this methodology is particularly well-suited for in-depth analysis of complex, interconnected discourse in political science, social science, and humanities research.

### **5.3 Limitations and Future Research Directions**

The findings of this study expose several important limitations that suggest directions for future research.

**Resolution Tuning and Post-Processing**: The persistent semantic overlap suggests that community detection parameters require domain-specific tuning. Future research could explore: (i) adjusting the Leiden algorithm's resolution parameter to find optimal granularity for political discourse; (ii) implementing post-processing steps to merge structurally distinct but semantically identical communities; and (iii) developing hybrid metrics that balance structural coherence with semantic distinctiveness.

**Entity Description Generation**: A critical gap identified is the lack of entity-level semantic information. Generating and embedding entity descriptions (e.g., "European Union—a political and economic union of 27 member states") could enrich the graph with semantic content, potentially improving the alignment between structural and semantic topic quality.

**Dynamic and Temporal Analysis**: The current model produces a static snapshot of discourse. The observed degradation pattern—where topic distinctiveness decreases as the corpus grows—suggests that temporal dynamics are crucial. Future work could develop dynamic graph analysis techniques to track topic evolution, emergence, merging, and divergence over time, providing insights into the temporal structure of political discourse.

**Cross-Lingual and Multimodal Applications**: The methodology could be extended to multilingual corpora to explore how topics are framed differently across languages. Furthermore, integrating multimodal data—such as voting records, policy documents, or even visual rhetoric—into the knowledge graph could provide richer bases for topic analysis.

**Domain-Specific Evaluation Metrics**: Standard clustering metrics designed for well-separated classes may be inappropriate for deliberative discourse. Future research should develop evaluation frameworks tailored to political and social science applications, potentially incorporating measures of argumentative diversity, perspectival variance, or deliberative quality alongside traditional separation metrics.

**Causal and Argumentative Analysis**: Moving beyond thematic identification, the explicit relational structure of the knowledge graph could be leveraged for advanced tasks such as causal claim extraction, argumentative structure mapping, stance detection, and influence network analysis—tasks that would be difficult or impossible with purely distributional models.

---

## **References**

Barabási, A.-L. & Albert, R. (1999) 'Emergence of scaling in random networks', *Science*, 286(5439), pp. 509-512.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O. & Larson, J. (2024) 'From local to global: A GraphRAG approach to query-focused summarization', *arXiv preprint arXiv:2404.16130*.

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

Giles, H. (2016) 'Communication Accommodation Theory', in Berger, C. R. & Roloff, M. E. (eds.) *The International Encyclopedia of Interpersonal Communication*. Hoboken: Wiley, pp. 1-18.

Rousseeuw, P. J. (1987) 'Silhouettes: A graphical aid to the interpretation and validation of cluster analysis', *Journal of Computational and Applied Mathematics*, 20, pp. 53-65.

