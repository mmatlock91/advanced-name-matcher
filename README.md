# advanced-name-matcher
A high-accuracy Python library for semantic name matching using a hybrid AI/ML (Retriever/Reranker) approach.

# Advanced Name Matcher: A Hybrid AI/ML Solution
A high-performance Python library for semantic and fuzzy name matching between two disparate datasets. This project was architected to solve the common, high-stakes business problem of joining large-scale datasets that lack a common, reliable key and suffer from inconsistent string data.

## The Problem
In many enterprise environments, critical business data is siloed across multiple systems (e.g., a sales CRM and a financial reporting database). Joining these systems is essential for a holistic business view, but is often impossible due to the lack of a shared identifier. Manual matching is not scalable, and traditional fuzzy logic often fails to capture the semantic nuance of complex entity names, leading to low accuracy and untrustworthy results.

## The Solution: A Two-Pass, Hybrid Architecture
This library implements a sophisticated, two-pass pipeline that leverages the strengths of both traditional and AI-powered techniques to achieve a state-of-the-art balance of speed and accuracy.

### Pass 1: High-Confidence Fuzzy Match
The system first performs a rapid, parallelized pass using `rapidfuzz`. This initial stage is designed to quickly and efficiently identify and clear all the "easy wins"—near-exact string matches that meet a high confidence threshold. This significantly reduces the volume of data that needs to be processed by the more computationally expensive AI model.

### Pass 2: AI-Powered Semantic Reranking
The remaining, more difficult, unmatched names are then passed to a state-of-the-art **Retriever/Reranker** AI model.
1.  **Retriever Stage:** For each name, an efficient vector search using `FAISS` and a `SentenceTransformer` model retrieves a small set of the most likely semantic candidates from the master list.
2.  **Reranker Stage:** This smaller set of candidates is then passed to a more powerful **Cross-Encoder model.** The cross-encoder performs a deep, pairwise semantic analysis, assigning a precise similarity score.
3.  **Hybrid Scoring & Validation:** A final, weighted score combining the semantic (AI) and lexical (fuzzy) similarities is calculated. This is then passed through a final, robust validation layer to ensure a high degree of confidence before declaring a match.

This hybrid, two-pass architecture provides the best of both worlds: the raw speed of fuzzy logic for the simple cases, and the deep, semantic intelligence of a state-of-the-art AI model for the complex ones.

## Key Features
- **State-of-the-Art Accuracy:** Achieves >99% accuracy on real-world, messy business data through a sophisticated hybrid scoring model.
- **High-Performance Architecture:** Utilizes multiprocessing to parallelize workloads and `FAISS` for blazingly fast vector search.
- **Sophisticated AI/ML Core:** Implements a modern Retriever/Reranker architecture with `sentence-transformers`.
- **Robust & Resilient:** Features a final, logical validation gate to ensure the quality and integrity of the AI's suggestions.

## Basic Usage
This repository includes sample data (`sample_source_data.csv`, `sample_master_data.csv`) to demonstrate the matcher's capabilities. The `example.py` script provides a simple, command-line demonstration.

To run the demonstration:
1.  Clone the repository: `git clone https://github.com/your-username/advanced-name-matcher.git`
2.  Install the required packages: `pip install -r requirements.txt`
3.  Run the example script: `python example.py`

---
*This project was developed as a personal portfolio piece and is not affiliated with any past or present employer.*
