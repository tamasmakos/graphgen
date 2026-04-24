#!/usr/bin/env python3
"""
LDA Analysis for "This is Europe" Debate Corpus
Run this to compare LDA results with the graph-based topic modeling approach
"""

import os
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Read all debate files
debate_dir = "/root/graphgen/input/txt/translated"
debate_files = sorted([f for f in os.listdir(debate_dir) if f.endswith('.txt')])
print(f"Found {len(debate_files)} debate files")

# Read and preprocess text
documents = []
for f in debate_files:
    with open(os.path.join(debate_dir, f), 'r', encoding='utf-8') as file:
        content = file.read()
        # Remove speaker annotations like [EN] Name | Surname said...
        content = re.sub(r'\[EN\]\s*\w+\s*\|\s*\w+.*?said.*?NULL\s*that\s*', '', content)
        documents.append(content)

print(f"Loaded {len(documents)} documents")
print(f"Sample text length (chars): {len(documents[0])}")

# Create document-topic matrix using CountVectorizer (required for LDA)
vectorizer = CountVectorizer(
    max_df=0.85,          # Ignore terms in >85% of docs
    min_df=2,             # Ignore terms in <2 docs
    max_features=2000,    # Limit vocabulary
    stop_words='english', # Remove English stop words
    ngram_range=(1, 2)    # Include bigrams
)

doc_term_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names)}")
print(f"Document-term matrix shape: {doc_term_matrix.shape}")

# Run LDA with same number of topics as graph-based approach (~13)
n_topics = 13  # Match number of debates

lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=30,
    learning_method='batch',  # Use batch for better results
    random_state=42,
    n_jobs=-1
)

lda_output = lda_model.fit_transform(doc_term_matrix)

print(f"\nLDA model fitted.")
print(f"Log-likelihood: {lda_model.score(doc_term_matrix):.2f}")
print(f"Perplexity: {lda_model.perplexity(doc_term_matrix):.2f}")

# Get top words for each topic
def get_top_words(model, feature_names, n_top_words=15):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[:-n_top_words-1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics[topic_idx] = top_words
    return topics

topic_words = get_top_words(lda_model, feature_names, 15)

# Save results
results = {
    "model": "LDA (Latent Dirichlet Allocation)",
    "n_topics": n_topics,
    "vocabulary_size": len(feature_names),
    "log_likelihood": float(lda_model.score(doc_term_matrix)),
    "perplexity": float(lda_model.perplexity(doc_term_matrix)),
    "topics": {}
}

print("\n" + "="*60)
print("LDA TOPIC RESULTS (13 topics)")
print("="*60)
for topic_idx, words in topic_words.items():
    print(f"\nTopic {topic_idx}: {', '.join(words[:10])}")
    results["topics"][f"topic_{topic_idx}"] = words

# Document-topic distribution
print("\n" + "="*60)
print("DOCUMENT-TOPIC DISTRIBUTION")
print("="*60)
for i, doc_topics in enumerate(lda_output):
    dominant_topic = np.argmax(doc_topics)
    print(f"Doc {i} ({debate_files[i][:20]}...): Topic {dominant_topic} ({doc_topics[dominant_topic]:.2f})")
    results[f"doc_{i}"] = {
        "file": debate_files[i],
        "dominant_topic": int(dominant_topic),
        "topic_distribution": doc_topics.tolist()
    }

# Compare with different topic numbers
print("\n" + "="*70)
print("LDA SCALING ANALYSIS")
print("="*70)

for n_topics in [5, 10, 13, 15, 20]:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=30,
        learning_method='batch',
        random_state=42
    )
    doc_topics = lda.fit_transform(doc_term_matrix)
    
    # Get dominant topic for each document
    dominant_topics = np.argmax(doc_topics, axis=1)
    
    # Compute how many docs have clear dominant topic
    clarity = np.mean(np.max(doc_topics, axis=1))
    
    # Silhouette score
    silhouette = silhouette_score(doc_term_matrix.toarray(), dominant_topics)
    
    print(f"\nn_topics={n_topics}:")
    print(f"  Dominant topic clarity (mean max prob): {clarity:.3f}")
    print(f"  Silhouette score: {silhouette:.3f}")
    print(f"  Unique topics used: {len(set(dominant_topics))}/{n_topics}")
    print(f"  Perplexity: {lda.perplexity(doc_term_matrix):.1f}")

# Save results to JSON
output_path = "/root/graphgen/thesis/lda_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_path}")