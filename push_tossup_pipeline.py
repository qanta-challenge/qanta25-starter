# %%
from transformers import pipeline

# Import your custom pipeline class here:
# Replace this import with your actual pipeline class
from hf_templates.tossup.simple_tiny_tossup import QBTossupPipeline

# Load the custom pipeline by specifying the task and underlying model
pipe = pipeline("quizbowl-tossup", device_map="auto", trust_remote_code=True)

sample_dataset = [
    {
        "question_text": "Name this fashion capital that is home to the painting Mona Lisa.",
    },
    {
        "question_text": "In the fifth century, this country was home to the Visigoths and the Vandals.",
    },
]

outs = pipe(sample_dataset)
for i, out in enumerate(outs):
    print(f"Sample {i + 1}:")
    print(f"  Question   : {sample_dataset[i]['question_text']}")
    print(f"  Answer     : {out['answer']}")
    print(f"  Confidence : {out['confidence']:.2f}")
    print(f"  Buzz       : {out['buzz']}")


# Set your pipeline name here:
PIPELINE_SUBMISSION_NAME = "simple-tiny-llama-tossup-2025a"

# Push it to your account on Hugging Face Hub
pipe.push_to_hub(repo_id=PIPELINE_SUBMISSION_NAME)
# %%
