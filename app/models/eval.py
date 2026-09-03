from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """A test sample for evaluating RAG performance."""

    question: str = Field(..., description="The query to ask the RAG pipeline")
    ground_truth: Optional[str] = Field(
        None, description="Expected ground truth answer for accuracy comparison"
    )
    expected_source: Optional[str] = Field(
        None, description="Expected source document or context snippet"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Custom metadata for this test case"
    )


class MetricScore(BaseModel):
    """Score and reasoning for an individual evaluation metric."""

    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0.0 and 1.0")
    reasoning: Optional[str] = Field(
        None, description="Explanation/reasoning provided by the evaluator"
    )


class EvalResultItem(BaseModel):
    """Evaluation result for a single sample."""

    question: str
    generated_answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    scores: Dict[str, MetricScore] = Field(default_factory=dict)
    latency_seconds: float = 0.0


class EvalReport(BaseModel):
    """Aggregate evaluation report across all tested samples."""

    total_samples: int
    duration_seconds: float
    summary_scores: Dict[str, float] = Field(
        ..., description="Average scores across each metric dimension"
    )
    results: List[EvalResultItem] = Field(default_factory=list)
