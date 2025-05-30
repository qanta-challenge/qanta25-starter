# %%
# ----------------------------------------------------------
# Custom Hugging-Face pipeline for the “bonus” split that refers to the existing models
# Task id  :  quizbowl-bonus
# Expected input keys : leadin, part, previous_parts ('text' and 'guess')
# Must return        : answer, confidence, explanation
# ----------------------------------------------------------
import json
import sys

import json_repair
import torch
import torch.nn.functional as F
from datasets import Dataset
from loguru import logger
from rich import print as rprint
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Pipeline,
    TFAutoModelForCausalLM,
    pipeline,
)
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline

logger.remove()
logger.add(sys.stdout, diagnose=False, colorize=True, level="INFO", backtrace=True)


def format_part(number: int, text: str, guess: str) -> str:
    return f"\t * Part {number}: {text}\n\t * Model Guess: {guess}"


def format_parts(parts: list[dict]) -> str:
    text = "\n".join(
        [format_part(i + 1, p["text"], p["guess"]) for i, p in enumerate(parts)]
    )
    if text:
        return f"Previous Parts:\n{text}"
    return ""


system_prompt = """
You are a quizbowl player. Given the a leadin text and the question, provide the your best guess, a brief (1-2 sentences) explanation along with a well-calibrated confidence in the guess.
The answer should be a single word or short phrase, and the explanation should be concise and relevant to the question.
The confidence should be calibrated and very well-reasoned, and it should be a float value between 0 and 1, indicating your confidence in the answer.
You should also provide a justification for the confidence score, explaining why you chose that value.

Format your response in below JSON format:

{
    "answer": str,
    "explanation": str,
    "confidence": float (0-1 in the steps of 0.01)
    "justification": str (optional justification for the confidence score)
}
"""

user_prompt_template = """
Leadin: {leadin}
Question: {part}
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


def postprocess_response(output_text, scores=None):
    # Extract answer and explanation from the final content.
    answer, explanation = "", ""
    # If JSON parsing fails, attempt to repair the JSON string.
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
        logger.exception(
            f"Error parsing JSON: {e.__class__.__name__} - {e}. Got:\n{output_text}"
        )
        # If still fails, return empty answer and explanation.
        return {"answer": "", "confidence": 0.0, "explanation": ""}

    # Compute a confidence score by averaging the max softmax probabilities over generated tokens.
    if scores is not None and len(scores) > 0:
        probs = [F.softmax(score, dim=-1).max().item() for score in scores]
        logit_confidence = float(sum(probs) / len(probs)) if probs else 0.0
    else:
        logit_confidence = 0.0

    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
            if logit_confidence > 0.0:
                confidence = (confidence + logit_confidence) / 2
        except ValueError:
            confidence = logit_confidence

    return {
        "answer": answer,
        "confidence": confidence,
        "explanation": explanation,
    }


class QwenBonusPipeline(Pipeline):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.tokenizer.padding_side = "left"

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
            padding_side="left",
            padding=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        return model_inputs

    def _forward(self, model_inputs):
        with torch.inference_mode():
            # Refer to https://huggingface.co/Qwen/Qwen3-32B-AWQ#best-practices
            # for more details on the choice of generation parameters.
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                min_p=0.0,
            )
        input_length = model_inputs["input_ids"].shape[1]
        outputs.sequences = outputs.sequences[:, input_length:]
        outputs.scores = torch.stack(outputs.scores, dim=1)
        return outputs

    def postprocess(self, model_outputs):
        output_texts = self.tokenizer.batch_decode(
            model_outputs.sequences, skip_special_tokens=True
        )
        records = [postprocess_response(output_text) for output_text in output_texts]

        return records


PIPELINE_REGISTRY.register_pipeline(
    "quizbowl-bonus",
    pipeline_class=QwenBonusPipeline,
    pt_model=AutoModelForCausalLM,
    default={
        "pt": ("Qwen/Qwen3-0.6B", "main"),
    },
    type="text",
)
# %%
if __name__ == "__main__":
    pipe = pipeline(
        "quizbowl-bonus",
        "Qwen/Qwen3-32B-AWQ",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    examples = [
        {
            "leadin": "This is a leadin.",
            "part": "What is the capital of France?",
        },
        {
            "leadin": "This is another leadin.",
            "part": "What is the largest planet in our solar system?",
        },
        {
            "leadin": "This is a leadin with no previous parts.",
            "part": "What is the chemical symbol for water?",
        },
    ] * 4

    dataset = Dataset.from_list(examples)

    print("Dataset size:", len(dataset))
    outputs = []
    batch_size = 4
    for batch in tqdm(dataset.batch(batch_size), desc="Processing batches"):
        output = pipe(batch, batch_size=batch_size)
        outputs.extend(output)
# %%
# %%
