import argparse
from pathlib import Path

from app.eval.dataset import DatasetManager
from app.eval.runner import RAGEvaluator
from app.rag.engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG Evaluation Benchmark on the current knowledge base."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_dataset.json",
        help="Path to evaluation dataset JSON file (default: data/eval_dataset.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples to run (useful for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results",
        help="Prefix/path for saving output JSON and Markdown reports (default: eval_results)",
    )

    args = parser.parse_args()

    print("[Eval] Initializing RAG Pipeline...")
    engine = RAGEngine()

    # Load dataset
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        print(f"[Eval] Loading test dataset from {dataset_path}...")
        samples = DatasetManager.load_from_json(dataset_path)
    else:
        print("[Eval] Specified dataset file not found. Using default golden test dataset...")
        samples = DatasetManager.get_default_dataset()

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
        print(f"[Eval] Limiting evaluation to first {len(samples)} sample(s).")

    evaluator = RAGEvaluator(engine=engine)
    report = evaluator.evaluate_dataset(samples=samples, verbose=True)

    # Save reports
    evaluator.save_report(report, output_prefix=args.output)
    print("[Eval] Evaluation complete!")


if __name__ == "__main__":
    main()
