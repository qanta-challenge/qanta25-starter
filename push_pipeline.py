"""
Usage examples for push_pipeline.py 🚀

For a Bonus Pipeline:
    python push_pipeline.py \
      pipelines.llama3_bonus.BonusPipeline \
      --task bonus \
      --model meta-llama/Llama-3.2-3B-Instruct \
      --repo-id my-llama--3-bonus-pipeline

For a Tossup Pipeline:
    python push_pipeline.py \
      pipelines.llama3_tossup.TossupPipeline \
      --task tossup \
      --model meta-llama/Llama-3.2-3B-Instruct \
      --repo-id my-llama--3-tossup-pipeline

Arguments:
    pipeline_class    Python import path to the pipeline class (positional argument, e.g. pipelines.llama3_bonus.BonusPipeline)
    --task            Task type: 'bonus' or 'tossup'
    --model, -m       Model repo to use (e.g. Qwen/Qwen3-0.6B) If not provided, will use the default model for the registered task
    --repo-id         Hugging Face repo ID to push to.
    --device-map      Device map for pipeline (default: auto)

After pushing, check your model here: ✨
    https://huggingface.co/models?search=your-hf-username
Replace 'your-hf-username' with your actual Hugging Face username!
"""

import argparse
import importlib

from transformers import pipeline


def dynamic_import(import_path):
    """Dynamically import a class from a string import path."""
    module_path, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(
        description="Push a custom pipeline to Hugging Face Hub."
    )
    parser.add_argument(
        "pipeline_class",
        help="Python import path to the pipeline class (e.g. hf_templates.bonus.simple_bonus_reasoner.QWenReasonerBonusPipeline)",
    )
    parser.add_argument(
        "--task",
        "--config",
        "--mode",
        required=True,
        choices=["bonus", "tossup"],
        help="Task name (e.g. bonus or tossup). This will be converted to 'quizbowl-bonus' or 'quizbowl-tossup'.",
    )
    parser.add_argument(
        "--model", "-m", default=None, help="Model repo to use (e.g. Qwen/Qwen3-0.6B)"
    )

    parser.add_argument(
        "--repo-id", required=True, help="Hugging Face repo ID to push to"
    )
    parser.add_argument(
        "--device-map", default="auto", help="Device map for pipeline (default: auto)"
    )
    args = parser.parse_args()

    # Dynamically import the pipeline class.
    # This will also register the pipeline class.
    pipeline_class = dynamic_import(args.pipeline_class)

    # Load the pipeline
    pipe = pipeline(
        f"quizbowl-{args.task}",
        model=args.model,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    # Push to hub
    print(f"Pushing pipeline to Hugging Face Hub repo: {args.repo_id}")
    pipe.push_to_hub(repo_id=args.repo_id)
    print("Pipeline pushed successfully!")


if __name__ == "__main__":
    main()
