import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from app.models.eval import EvalSample

logger = logging.getLogger(__name__)

# Pre-defined Golden Test Suite for the project's knowledge base
DEFAULT_GOLDEN_DATASET: List[EvalSample] = [
    EvalSample(
        question="Who created Python and when was it first released?",
        ground_truth="Python was created by Guido van Rossum and first released in 1991.",
        expected_source="knowledge.pdf",
        metadata={"category": "general_knowledge", "difficulty": "easy"},
    ),
    EvalSample(
        question="What built-in data types are available in Python according to the document?",
        ground_truth="The core built-in data types mentioned are int, float, str, bool, list, tuple, dict, and set.",
        expected_source="knowledge.pdf",
        metadata={"category": "data_types", "difficulty": "medium"},
    ),
    EvalSample(
        question="How does Python handle errors and exceptions?",
        ground_truth="Python uses try/except/finally blocks to handle exceptions gracefully and prevent programs from crashing.",
        expected_source="knowledge.pdf",
        metadata={"category": "error_handling", "difficulty": "easy"},
    ),
    EvalSample(
        question="What is crucial for navigating tough times according to the handwriting note?",
        ground_truth="Clinging to hope is crucial for navigating tough times. It is the quiet strength that whispers to keep going and fuels resilience.",
        expected_source="Handwriting Practice Paragraphs.jpg",
        metadata={"category": "ocr_image", "difficulty": "medium"},
    ),
    EvalSample(
        question="What does the handwriting practice note say about how every great achievement began?",
        ground_truth="Every great achievement and personal triumph began with someone refusing to succumb to despair.",
        expected_source="Handwriting Practice Paragraphs.jpg",
        metadata={"category": "ocr_image", "difficulty": "medium"},
    ),
    EvalSample(
        question="What is the capital of Mars?",
        ground_truth="I don't know.",
        expected_source=None,
        metadata={"category": "out_of_domain", "difficulty": "hard"},
    ),
]


class DatasetManager:
    """Manages loading, saving, and generating evaluation datasets."""

    @staticmethod
    def get_default_dataset() -> List[EvalSample]:
        """Return the curated golden test dataset."""
        return [sample.model_copy() for sample in DEFAULT_GOLDEN_DATASET]

    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> List[EvalSample]:
        """Load evaluation samples from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]

        return [EvalSample(**item) for item in data]

    @staticmethod
    def save_to_json(samples: List[EvalSample], file_path: Union[str, Path]) -> None:
        """Save evaluation samples to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [sample.model_dump() for sample in samples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[DatasetManager] Saved {len(samples)} samples to {path}")

    @staticmethod
    def generate_synthetic_samples(
        chunks: List[str], llm: Any, samples_per_chunk: int = 1
    ) -> List[EvalSample]:
        """Synthetically generate question-answer evaluation pairs from document text chunks."""
        generated: List[EvalSample] = []
        print(f"[DatasetManager] Generating synthetic eval samples from {len(chunks)} chunks...")

        for idx, chunk in enumerate(chunks[:5]):  # limit to first few chunks
            prompt = f"""Based on the following context passage, generate {samples_per_chunk} clear factual question and its ground truth answer.

Context:
{chunk}

Respond ONLY with a JSON list in this format:
```json
[
  {{
    "question": "Clear question based on the text?",
    "ground_truth": "Concise factual answer directly found in the text."
  }}
]
```"""
            try:
                resp = llm.invoke(prompt)
                content = resp.content if hasattr(resp, "content") else str(resp)
                import re

                match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
                if match:
                    items = json.loads(match.group(1))
                    for it in items:
                        generated.append(
                            EvalSample(
                                question=it["question"],
                                ground_truth=it["ground_truth"],
                                metadata={"source_chunk_index": idx, "type": "synthetic"},
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to generate synthetic sample for chunk {idx}: {e}")

        return generated
