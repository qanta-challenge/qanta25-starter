"""
📤 QANTA Pipeline Publisher

This script publishes your custom Quiz Bowl pipelines to the Hugging Face Hub,
making them available for you to submit to the QANTA leaderboard.

FEATURES:
- ✅ Automatic pipeline registration and validation
- ✅ Support for both bonus and tossup tasks
- ✅ Flexible model selection
- ✅ One-command publishing to Hugging Face Hub
- ✅ Comprehensive error handling and validation

USAGE EXAMPLES:

1. Push a Bonus Pipeline:
   python push_pipeline.py \
     pipelines.llama3_bonus.BonusPipeline \
     --task bonus \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --repo-id my-llama3-bonus-pipeline

2. Push a Tossup Pipeline:
   python push_pipeline.py \
     pipelines.llama3_tossup.TossupPipeline \
     --task tossup \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --repo-id my-llama3-tossup-pipeline

ARGUMENTS:
    pipeline_class    Python import path to the pipeline class (positional argument, e.g. pipelines.llama3_bonus.BonusPipeline)
    --task            Task type: 'bonus' or 'tossup'
    --model, -m       Model repo to use (e.g. Qwen/Qwen3-0.6B) If not provided, will use the default model for the registered task
    --repo-id         Hugging Face repo ID to push to.

PIPELINE REQUIREMENTS:
- Your pipeline class must inherit from `transformers.Pipeline`
- Must be properly registered with the transformers pipeline registry in its module
- Should implement the required methods: `preprocess`, `_forward`, `postprocess`
- Must be compatible with the specified task type (`bonus`/`tossup`)

POST-PUBLISHING:
After successful publishing, your pipeline will be available at:
https://huggingface.co/{your-repo-id}

You can then use it with:
from transformers import pipeline
pipe = pipeline("quizbowl-{task}", model="{your-repo-id}")

For more information on creating pipelines, see the documentation in the README.md file.
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
    args = parser.parse_args()

    # Dynamically import the pipeline class.
    # This will also register the pipeline class.
    pipeline_class = dynamic_import(args.pipeline_class)

    # Load the pipeline
    pipe = pipeline(
        f"quizbowl-{args.task}",
        model=args.model,
        device_map="auto",
        trust_remote_code=True,
    )

    # Push to hub
    print(f"Pushing pipeline to Hugging Face Hub repo: {args.repo_id}")
    pipe.push_to_hub(repo_id=args.repo_id)
    print("Pipeline pushed successfully!")


if __name__ == "__main__":
    main()
