import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from app.models.eval import MetricScore

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    # Try finding markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Try raw curly braces
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Fallback heuristic score parsing if JSON is malformed
    score_match = re.search(r'"?score"?\s*:\s*([0-9]*\.?[0-9]+)', text)
    reason_match = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', text)
    score = float(score_match.group(1)) if score_match else 0.5
    reasoning = reason_match.group(1) if reason_match else text.strip()
    return {"score": min(max(score, 0.0), 1.0), "reasoning": reasoning}


class BaseMetric(ABC):
    """Abstract base class for all RAG evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> MetricScore:
        pass


class FaithfulnessEvaluator(BaseMetric):
    """Evaluates whether the claims in the generated answer are grounded in the retrieved context."""

    def __init__(self, llm: Any):
        self.llm = llm

    @property
    def name(self) -> str:
        return "faithfulness"

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> MetricScore:
        combined_context = "\n---\n".join(contexts)

        prompt = f"""You are an impartial judge evaluating the FAITHFULNESS (groundedness) of a generated answer.
Faithfulness measures whether every claim and statement in the answer can be directly inferred from the provided context.

Context:
{combined_context}

Question:
{question}

Generated Answer:
{answer}

Instructions:
1. Identify all key claims made in the Generated Answer.
2. Check if each claim is supported by the Context.
3. If the answer says "I don't know" and the context does not contain the answer, that is 100% faithful (score 1.0).
4. Assign a score between 0.0 and 1.0:
   - 1.0: Completely faithful, all claims supported by context, no hallucinations.
   - 0.5: Partially faithful, some unsupported claims or exaggerations.
   - 0.0: Completely unfaithful / entirely fabricated or contradicts the context.

Respond ONLY with a valid JSON object in this exact format:
```json
{{
  "score": 1.0,
  "reasoning": "Explanation of faithfulness assessment"
}}
```"""
        try:
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = _extract_json(content)
            score = float(data.get("score", 0.5))
            score = min(max(score, 0.0), 1.0)
            return MetricScore(
                metric_name=self.name,
                score=score,
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"Error in Faithfulness evaluation: {e}")
            return MetricScore(metric_name=self.name, score=0.0, reasoning=str(e))


class AnswerRelevanceEvaluator(BaseMetric):
    """Evaluates whether the generated answer directly and completely addresses the user question."""

    def __init__(self, llm: Any):
        self.llm = llm

    @property
    def name(self) -> str:
        return "answer_relevance"

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> MetricScore:
        prompt = f"""You are an impartial judge evaluating the ANSWER RELEVANCE of a generated response to a question.
Answer Relevance measures how directly, clearly, and completely the answer addresses the specific question asked.

Question:
{question}

Generated Answer:
{answer}

Instructions:
1. Determine if the answer directly addresses the intent of the question.
2. Penalize evasive, irrelevant, or incomplete answers.
3. Assign a score between 0.0 and 1.0:
   - 1.0: Highly relevant, concise, directly answers what was asked.
   - 0.5: Partially relevant, misses parts of the question or contains excessive rambling.
   - 0.0: Irrelevant or completely off-topic.

Respond ONLY with a valid JSON object in this exact format:
```json
{{
  "score": 1.0,
  "reasoning": "Explanation of relevance assessment"
}}
```"""
        try:
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = _extract_json(content)
            score = float(data.get("score", 0.5))
            score = min(max(score, 0.0), 1.0)
            return MetricScore(
                metric_name=self.name,
                score=score,
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"Error in Answer Relevance evaluation: {e}")
            return MetricScore(metric_name=self.name, score=0.0, reasoning=str(e))


class ContextRelevanceEvaluator(BaseMetric):
    """Evaluates the proportion and quality of retrieved context chunks relevant to the question."""

    def __init__(self, llm: Any):
        self.llm = llm

    @property
    def name(self) -> str:
        return "context_relevance"

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> MetricScore:
        formatted_chunks = "\n".join(
            [f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(contexts)]
        )

        prompt = f"""You are an impartial judge evaluating the CONTEXT RELEVANCE of retrieved document chunks for a question.
Context Relevance measures whether the retrieved chunks contain pertinent information needed to answer the question.

Question:
{question}

Retrieved Context Chunks:
{formatted_chunks}

Instructions:
1. Evaluate each chunk to see if it provides useful information for answering the Question.
2. Calculate the fraction of relevant chunks versus noisy/distractor chunks.
3. Assign a score between 0.0 and 1.0:
   - 1.0: All retrieved chunks are highly relevant and informative for the question.
   - 0.6 - 0.9: Majority of chunks are relevant, with minor noise.
   - 0.1 - 0.5: Low relevance, only a fraction contains useful information.
   - 0.0: Completely irrelevant contexts retrieved.

Respond ONLY with a valid JSON object in this exact format:
```json
{{
  "score": 1.0,
  "reasoning": "Explanation of context relevance assessment"
}}
```"""
        try:
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = _extract_json(content)
            score = float(data.get("score", 0.5))
            score = min(max(score, 0.0), 1.0)
            return MetricScore(
                metric_name=self.name,
                score=score,
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"Error in Context Relevance evaluation: {e}")
            return MetricScore(metric_name=self.name, score=0.0, reasoning=str(e))


class SemanticSimilarityEvaluator(BaseMetric):
    """Evaluates semantic cosine similarity between the generated answer and ground truth answer."""

    def __init__(self, embeddings_model: Any):
        self.embeddings = embeddings_model

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> MetricScore:
        if not ground_truth or not ground_truth.strip():
            return MetricScore(
                metric_name=self.name,
                score=1.0,
                reasoning="No ground truth provided for comparison.",
            )

        try:
            vec_pred = np.array(self.embeddings.embed_query(answer))
            vec_gt = np.array(self.embeddings.embed_query(ground_truth))

            norm_pred = np.linalg.norm(vec_pred)
            norm_gt = np.linalg.norm(vec_gt)

            if norm_pred == 0 or norm_gt == 0:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(vec_pred, vec_gt) / (norm_pred * norm_gt))

            # Normalize cosine similarity to [0, 1] range
            score = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0 if cos_sim < 0 else cos_sim))
            score = round(score, 4)

            return MetricScore(
                metric_name=self.name,
                score=score,
                reasoning=f"Cosine similarity between embeddings: {cos_sim:.4f}",
            )
        except Exception as e:
            logger.error(f"Error in Semantic Similarity evaluation: {e}")
            return MetricScore(metric_name=self.name, score=0.0, reasoning=str(e))
