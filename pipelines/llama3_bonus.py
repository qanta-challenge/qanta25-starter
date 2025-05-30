# %%
# ----------------------------------------------------------
# Custom Hugging-Face pipeline for the “bonus” split that refers to the existing models
# Task id  :  quizbowl-bonus
# Expected input keys : leadin, part, previous_parts ('text' and 'guess')
# Must return        : answer, confidence, explanation
# ----------------------------------------------------------


import json_repair
import torch
from datasets import Dataset
from loguru import logger
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    Pipeline,
    TFAutoModelForCausalLM,
    pipeline,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY


def format_part(number: int, text: str, guess: str) -> str:
    return f"\t * Part {number}: {text}\n\t * Model Guess: {guess}"


system_prompt = """
You are a quizbowl player. Given the a leadin and your responses to the previous related parts, provide the answer, a brief (1-2 sentences) explanation to the provided question along with your confidence in the guess.
The answer should be a single word or short phrase, and the explanation should be concise and relevant to the question.
The answer should be formatted in the below JSON format:

{
    "answer": str,
    "explanation": str,
    "confidence": float (0-1 in the steps of 0.01)
    "justification": str (optional justification for the confidence score)
}
The confidence should be a float between 0 and 1, representing your confidence in the answer.
"""

user_prompt_template = """
"Leadin: {leadin}
Question: {part}"
What is being asked in the question? Provide a concise answer, a brief explanation, and your confidence in the guess along with justification."""


def prepare_conversation(leadin, part):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt_template.format(leadin=leadin, part=part),
        },
    ]
    return messages


def parse_output_text(output_text: str):
    try:
        start_index = output_text.find("{")
        if start_index == -1:
            raise ValueError("No JSON object found in the output text.")
        output_text = output_text[start_index:]
        json_data = json_repair.loads(output_text)
        if isinstance(json_data, list):
            json_data = json_data[0]
        answer = json_data.get("answer", "").strip()
        explanation = json_data.get("explanation", "").strip()
        confidence = json_data.get("confidence", 0.0)
    except Exception as e:
        logger.warning(
            f"Error parsing JSON: {e.__class__.__name__} - {e}. Got:\n{output_text}"
        )
        answer, explanation, confidence = "", "", 0.0

    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        logger.warning(f"Invalid confidence value: {confidence}. Defaulting to 0.0.")
        confidence = 0.0
    return {
        "answer": answer,
        "explanation": explanation,
        "confidence": confidence,
    }


def postprocess_response(output_text, scores=None):
    model_response = parse_output_text(output_text)

    # Compute a confidence score by averaging the max softmax probabilities over generated tokens.
    if scores is not None and len(scores) > 0:
        probs = [F.softmax(score, dim=-1).max().item() for score in scores]
        logit_confidence = float(sum(probs) / len(probs)) if probs else 0.0
        model_response["confidence"] = (
            model_response["confidence"] + logit_confidence
        ) / 2

    return model_response


class BonusPipeline(Pipeline):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _sanitize_parameters(self, **kwargs):
        # No additional parameters needed
        return {}, {}, {}

    def preprocess(self, inputs):
        batch_size = len(inputs["leadin"])
        conversations = []
        for i in range(batch_size):
            conversations.append(
                prepare_conversation(inputs["leadin"][i], inputs["part"][i])
            )

        model_inputs = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        )
        return model_inputs

    def _forward(self, model_inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Remove the input tokens from the output sequences
        # This is necessary because the model generates tokens based on the input context
        # and we only want the new tokens generated by the model.
        input_length = model_inputs["input_ids"].shape[1]
        outputs.sequences = outputs.sequences[:, input_length:]
        outputs.scores = torch.stack(outputs.scores, dim=1)
        return outputs

    def postprocess(self, model_outputs):
        output_texts = self.tokenizer.batch_decode(
            model_outputs.sequences, skip_special_tokens=True
        )
        records = []

        for output_text in output_texts:
            record = postprocess_response(output_text)
            records.append(record)
        return records


PIPELINE_REGISTRY.register_pipeline(
    "quizbowl-bonus",
    pipeline_class=BonusPipeline,
    pt_model=LlamaForCausalLM,
    default={
        "pt": ("meta-llama/Llama-3.2-3B-Instruct", "main"),
    },
    type="text",
)
# %%
if __name__ == "__main__":
    pipe = pipeline("quizbowl-bonus", device_map="auto", trust_remote_code=True)

    examples = [
        {
            "leadin": "This is a leadin.",
            "part": "What is the capital of France?",
        },
        {
            "leadin": "This is another leadin.",
            "part": "What is the largest planet in our solar system?",
            "previous_parts": [
                {"text": "What is the smallest planet?", "guess": "Mercury"},
                {"text": "What is the second smallest planet?", "guess": "Mars"},
            ],
        },
        {
            "leadin": "This is a leadin with no previous parts.",
            "part": "What is the chemical symbol for water?",
            "previous_parts": [],
        },
    ] * 5

    dataset = Dataset.from_list(examples)

    print("Dataset size:", len(dataset))
    outputs = []
    batch_size = 5
    for batch in tqdm(dataset.batch(batch_size), desc="Processing batches"):
        output = pipe(batch, batch_size=batch_size)
        outputs.extend(output)
