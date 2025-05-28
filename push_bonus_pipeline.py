# %%
from transformers import pipeline

# Import your custom pipeline class here:
# Replace this import with your actual pipeline class
from hf_templates.bonus.simple_bonus_reasoner import QWenReasonerBonusPipeline

# Load the custom pipeline by specifying the task and underlying model
pipe = pipeline(
    "quizbowl-bonus", model="Qwen/Qwen3-0.6B", device_map="auto", trust_remote_code=True
)

sample_dataset = [
    {
        "leadin": "We are talking about a country in western Europe.",
        "part": "What is the capital of France?",
    },
    {
        "leadin": "We are talking about people who are known for their neutrality, chocolate, and watches.",
        "part": "Name this country in western Europe that shares the Alps with another country.",
    },
]

outs = pipe(sample_dataset)
for i, out in enumerate(outs):
    print(f"Sample {i + 1}:")
    print(f"  Lead-in: {sample_dataset[i]['leadin']}")
    print(f"  Part: {sample_dataset[i]['part']}")
    print(f"  Answer: {out['answer']}")
    print(f"  Confidence: {out['confidence']:.2f}")
    print()


# Set your pipeline name here:
PIPELINE_SUBMISSION_NAME = "simple-bonus-reasoner-2025a"

# Push it to your account on Hugging Face Hub
pipe.push_to_hub(repo_id=PIPELINE_SUBMISSION_NAME)
# %%
