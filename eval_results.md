# RAG Evaluation Benchmark Report

- **Total Test Samples**: `2`
- **Duration**: `136.07s`
- **Average Query Latency**: `68.03s`

## Summary Metric Scores

| Metric | Average Score | Performance Rating |
| :--- | :---: | :--- |
| **Faithfulness** | `50.0%` | Needs Attention |
| **Answer Relevance** | `100.0%` | Excellent |
| **Context Relevance** | `51.5%` | Needs Attention |
| **Semantic Similarity** | `90.4%` | Excellent |

## Detailed Sample Breakdown

### Sample 1: Who created Python and when was it first released?

- **Generated Answer**: Python was created by Guido van Rossum and was first released in 1991.
- **Ground Truth**: Python was created by Guido van Rossum and first released in 1991.
- **Latency**: `7.448s`

**Metric Scores & Reasoning:**
- **Faithfulness** (`100.0%`): The generated answer states that Python was created by Guido van Rossum and first released in 1991, which matches the information in the context. All claims are directly supported, with no unsupported or contradictory statements.
- **Answer Relevance** (`100.0%`): The answer directly and concisely states that Guido van Rossum created Python and specifies the first release year as 1991, fully addressing both parts of the question.
- **Context Relevance** (`33.0%`): Out of the three retrieved chunks, only Chunk 1 contains information that directly answers the question (creator: Guido van Rossum, release year: 1991). Chunks 2 and 3 are unrelated to the question and provide only installation instructions and language syntax details. Therefore, only one third of the chunks are relevant, placing the relevance score in the 0.1–0.5 range.
- **Semantic Similarity** (`99.8%`): Cosine similarity between embeddings: 0.9976

### Sample 2: What built-in data types are available in Python according to the document?

- **Generated Answer**: The document lists the following core built‑in data types in Python:

- **int** – whole numbers  
- **float** – decimal numbers  
- **str** – text strings  
- **bool** – boolean values (`True` or `False`)  
- **list** – ordered, mutable collections  
- **tuple** – ordered, immutable collections  
- **dict** – key–value pair collections  
- **set** – unordered collections of unique items
- **Ground Truth**: The core built-in data types mentioned are int, float, str, bool, list, tuple, dict, and set.
- **Latency**: `10.55s`

**Metric Scores & Reasoning:**
- **Faithfulness** (`0.0%`): HTTPSConnectionPool(host='integrate.api.nvidia.com', port=443): Read timed out. (read timeout=60)
- **Answer Relevance** (`100.0%`): The answer lists all core built‑in data types (int, float, str, bool, list, tuple, dict, set) that the document references, directly and succinctly answering the question with no irrelevant content.
- **Context Relevance** (`70.0%`): Chunks 1 and 2 each contain explicit lists of built‑in data types (int, float, str, bool, list, tuple, dict, set), directly answering the question. Chunk 3 is a generic document header with no type information and is therefore irrelevant. Out of three retrieved chunks, two are relevant, yielding a relevance ratio of 2/3, which falls into the 0.6–0.9 range.
- **Semantic Similarity** (`81.1%`): Cosine similarity between embeddings: 0.8109
