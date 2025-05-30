# %%
# ----------------------------------------------------------
# Custom Hugging-Face pipeline for the “tossup” split that only refers to the existing models
# Task id  :  quizbowl-tossup
# Expected input keys : question_text
# Must return        : answer, confidence, buzz
# ----------------------------------------------------------

import re

import json_repair
import torch
from datasets import Dataset
from loguru import logger
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import Pipeline, pipeline
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY


def format_part(number: int, text: str, guess: str) -> str:
    return f"\t * Part {number}: {text}\n\t * Model Guess: {guess}"


SYSTEM_PROMPT = """
You are a quizbowl player. Given a partially revealed Quiz question, provide your best guess for the answer to what is being asked in the question.
The answer should be a single word or short phrase. Also provide your confidence in the correctness of your guess, on a scale from 0 to 1, where 0 means no confidence and 1 means complete confidence.
If you are not sure, you can provide a guess with a low confidence score.
If you are completely unsure, you can return an empty string for the answer and a confidence score of 0.
You should also provide a justification for your assigned confidence score, if applicable.
The answer should be formatted in the below JSON format:

{
    "answer": str,
    "confidence": float (0-1 in the steps of 0.01)
    "justification": str (optional justification for the confidence score)
}
The confidence should be a float between 0 and 1, representing your confidence in the answer.
"""

PROMPT_TEMPLATE = """
Question revealed so far: "{question_text}"
What is being asked in the question? Provide a concise answer, your confidence in the guess."""


def prepare_conversation(question_text: str):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(question_text=question_text),
        },
    ]
    return messages


def fallback_parse_output_text(output_text: str):
    output_text = output_text.strip()
    answer = output_text.split("\n")[0].strip()
    # if answer starts with "Answer:" or "Guess:", or "A:", remove that part
    if answer.lower().startswith(("answer:", "guess:", "a:")):
        answer = answer.split(":", 1)[1].strip()

    # extract the line which has "confidence:" in it using regex
    match_conf = re.search(r"confidence\s*:\s*([0-9.]+)", output_text, re.IGNORECASE)
    confidence = float(match_conf.group(1)) if match_conf else 0.0

    # extract the line which has "justification:" in it using regex
    match_just = re.search(r"justification\s*:\s*(.*)", output_text, re.IGNORECASE)
    justification = match_just.group(1).strip() if match_just else ""

    return answer, confidence, justification


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
        confidence = json_data.get("confidence", 0.0)
        justification = json_data.get("justification", "")
    except Exception as e:
        logger.warning(
            f"Error parsing JSON: {e.__class__.__name__} - {e}. Got:\n{output_text}"
        )
        logger.info("Falling back to regex parsing.")
        answer, confidence, justification = fallback_parse_output_text(output_text)

    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        logger.warning(f"Invalid confidence value: {confidence}. Defaulting to 0.0.")
        confidence = 0.0
    return {
        "answer": answer,
        "justification": justification,
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
    model_response["buzz"] = model_response["confidence"] > 0.7
    return model_response


class TossupPipeline(Pipeline):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _sanitize_parameters(self, **kwargs):
        # No additional parameters needed
        return {}, {}, {}

    def preprocess(self, inputs):
        conversations = []
        for question in inputs["question_text"]:
            conversations.append(prepare_conversation(question))

        model_inputs = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
            pad_token_id=self.tokenizer.eos_token_id,
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
    "quizbowl-tossup",
    pipeline_class=TossupPipeline,
    pt_model=LlamaForCausalLM,
    default={
        "pt": ("meta-llama/Llama-3.2-3B-Instruct", "main"),
    },
    type="text",
)
# %%
if __name__ == "__main__":
    pipe = pipeline(
        "quizbowl-tossup",
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        trust_remote_code=True,
    )

    examples = [
        {
            "question_text": "What is the capital of France?",
        },
        {
            "question_text": "What is the largest planet in our solar system?",
        },
        {
            "question_text": "What is the chemical symbol for water?",
        },
    ] * 4

    dataset = Dataset.from_list(examples)

    print("Dataset size:", len(dataset))
    outputs = []
    batch_size = 4
    for batch in tqdm(dataset.batch(batch_size), desc="Processing batches"):
        output = pipe(batch, batch_size=batch_size)
        outputs.extend(output)
