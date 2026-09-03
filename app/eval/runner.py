import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from app.eval.metrics import (
    AnswerRelevanceEvaluator,
    BaseMetric,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    SemanticSimilarityEvaluator,
)
from app.models.eval import EvalReport, EvalResultItem, EvalSample
from app.rag.engine import RAGEngine

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Orchestrates comprehensive evaluation of a RAG pipeline."""

    def __init__(
        self,
        engine: RAGEngine,
        metrics: Optional[Sequence[BaseMetric]] = None,
    ):
        self.engine = engine
        if metrics is not None:
            self.metrics = list(metrics)
        else:
            self.metrics = [
                FaithfulnessEvaluator(self.engine.llm),
                AnswerRelevanceEvaluator(self.engine.llm),
                ContextRelevanceEvaluator(self.engine.llm),
                SemanticSimilarityEvaluator(self.engine.embeddings),
            ]

    def evaluate_sample(self, sample: EvalSample) -> EvalResultItem:
        """Run RAG query on a single sample and evaluate all metrics."""
        rag_output = self.engine.query_with_context(sample.question)
        answer = rag_output["answer"]
        contexts = rag_output["contexts"]
        latency = rag_output["latency_seconds"]

        scores = {}
        for metric in self.metrics:
            try:
                score_item = metric.evaluate(
                    question=sample.question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=sample.ground_truth,
                )
                scores[metric.name] = score_item
            except Exception as e:
                logger.error(f"Metric '{metric.name}' failed on question '{sample.question}': {e}")

        return EvalResultItem(
            question=sample.question,
            generated_answer=answer,
            contexts=contexts,
            ground_truth=sample.ground_truth,
            scores=scores,
            latency_seconds=latency,
        )

    def evaluate_dataset(
        self,
        samples: List[EvalSample],
        verbose: bool = True,
    ) -> EvalReport:
        """Evaluate a full list of samples and calculate aggregate metrics."""
        start_time = time.time()
        results: List[EvalResultItem] = []
        total = len(samples)

        if verbose:
            print(f"\n=======================================================")
            print(f"[EvalRunner] Starting RAG Evaluation across {total} test samples...")
            print(f"=======================================================\n")

        for idx, sample in enumerate(samples, start=1):
            if verbose:
                print(f"[{idx}/{total}] Evaluating: \"{sample.question}\"")

            result = self.evaluate_sample(sample)
            results.append(result)

            if verbose:
                scores_str = ", ".join(
                    [f"{name}: {item.score:.2f}" for name, item in result.scores.items()]
                )
                print(f"    -> Latency: {result.latency_seconds}s | Scores: {scores_str}")

        duration = round(time.time() - start_time, 2)

        # Calculate averages for each metric
        summary_scores: Dict[str, float] = {}
        for metric in self.metrics:
            scores = [
                res.scores[metric.name].score
                for res in results
                if metric.name in res.scores
            ]
            avg = sum(scores) / len(scores) if scores else 0.0
            summary_scores[f"avg_{metric.name}"] = round(avg, 4)

        report = EvalReport(
            total_samples=total,
            duration_seconds=duration,
            summary_scores=summary_scores,
            results=results,
        )

        if verbose:
            self.print_summary_card(report)

        return report

    def print_summary_card(self, report: EvalReport) -> None:
        """Print a formatted evaluation card to console."""
        print(f"\n=======================================================")
        print(f"[REPORT] RAG EVALUATION BENCHMARK SUMMARY")
        print(f"=======================================================")
        print(f"Total Test Samples : {report.total_samples}")
        print(f"Total Duration     : {report.duration_seconds}s")
        print(f"Average Latency    : {report.duration_seconds / max(report.total_samples, 1):.2f}s/query\n")
        print(f"{'Metric':<30} | {'Score':<10} | {'Status'}")
        print(f"{'-'*30}-|-{'-'*10}-|-{'-'*10}")

        for name, score in report.summary_scores.items():
            clean_name = name.replace("avg_", "").replace("_", " ").title()
            status = "EXCELLENT" if score >= 0.85 else ("GOOD" if score >= 0.70 else "NEEDS WORK")
            print(f"{clean_name:<30} | {score*100:>5.1f}%     | {status}")

        print(f"=======================================================\n")

    def format_markdown_report(self, report: EvalReport) -> str:
        """Generate a clean Markdown evaluation report."""
        lines = [
            "# RAG Evaluation Benchmark Report\n",
            f"- **Total Test Samples**: `{report.total_samples}`",
            f"- **Duration**: `{report.duration_seconds}s`",
            f"- **Average Query Latency**: `{report.duration_seconds / max(report.total_samples, 1):.2f}s`\n",
            "## Summary Metric Scores\n",
            "| Metric | Average Score | Performance Rating |",
            "| :--- | :---: | :--- |",
        ]

        for name, score in report.summary_scores.items():
            clean_name = name.replace("avg_", "").replace("_", " ").title()
            rating = "Excellent" if score >= 0.85 else ("Good" if score >= 0.70 else "Needs Attention")
            lines.append(f"| **{clean_name}** | `{score * 100:.1f}%` | {rating} |")

        lines.extend([
            "\n## Detailed Sample Breakdown\n",
        ])

        for i, res in enumerate(report.results, 1):
            lines.append(f"### Sample {i}: {res.question}\n")
            lines.append(f"- **Generated Answer**: {res.generated_answer}")
            if res.ground_truth:
                lines.append(f"- **Ground Truth**: {res.ground_truth}")
            lines.append(f"- **Latency**: `{res.latency_seconds}s`\n")
            lines.append("**Metric Scores & Reasoning:**")
            for m_name, m_score in res.scores.items():
                m_title = m_name.replace("_", " ").title()
                lines.append(f"- **{m_title}** (`{m_score.score * 100:.1f}%`): {m_score.reasoning}")
            lines.append("")

        return "\n".join(lines)

    def save_report(
        self, report: EvalReport, output_prefix: Union[str, Path] = "eval_results"
    ) -> None:
        """Save report in both JSON and Markdown formats."""
        path = Path(output_prefix)
        path.parent.mkdir(parents=True, exist_ok=True)

        json_path = path.with_suffix(".json")
        md_path = path.with_suffix(".md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.format_markdown_report(report))

        print(f"[RAGEvaluator] Reports saved to:\n  - JSON: {json_path}\n  - Markdown: {md_path}")
