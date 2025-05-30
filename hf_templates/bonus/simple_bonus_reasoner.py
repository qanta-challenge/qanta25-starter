# %%
"""
File: simple_bonus_reasoner.py

This Python script demonstrates how to build a custom Hugging-Face pipeline that utilizes a reasoner model
for quizbowl bonus questions. It is inspired by simple_tiny_bonus.py and qwen_bonus_reasoner.py. The script provides
comprehensive documentation to serve as a tutorial for users who wish to integrate a reasoning step into the pipeline.

Overview:
----------
1. The pipeline takes a quizbowl question with a leadin, main question part, and previous parts.
2. It formats the input using a custom chat template that includes both system instructions and user prompt.
3. The pipeline uses a reasoner model, which is set up with thinking mode enabled. This allows the model to provide
   both "thinking" content (internal reasoning) and a final answer output.
4. The postprocessing step extracts the answer, explanation, and confidence score.
5. Detailed comments and docstrings are provided throughout this file, explaining the design and steps.

Usage:
----------
To use this pipeline, register the pipeline and call it with a dictionary having keys "leadin", "part",
and optionally "previous_parts" (a list of dicts containing "text" and "guess").
"""

import json_repair
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    TFAutoModelForCausalLM,
)
from transformers.pipelines import PIPELINE_REGISTRY


# Define helper functions for formatting parts of the quizbowl input.
def format_part(number: int, text: str, guess: str) -> str:
    """
    Format a single part of the previous questions.

    Args:
        number (int): The part number.
        text (str): The text of the part.
        guess (str): The model's guess for this part.

    Returns:
        str: The formatted string for this part.
    """
    return f"\t * Part {number}: {text}\n\t * Model Guess: {guess}"


def format_parts(parts: list[dict]) -> str:
    """
    Format a list of previous parts into a single string.

    Args:
        parts (list[dict]): List of dictionaries with keys "text" and "guess".

    Returns:
        str: The combined string of all formatted parts.
    """
    text = "\n".join(
        [format_part(i + 1, p["text"], p["guess"]) for i, p in enumerate(parts)]
    )
    if text:
        return f"Previous Parts:\n{text}"
    return ""


# Define system and user prompts for the pipeline.
system_prompt = """
You are a well-calibrated, rational and logical savvy Quizbowl player. Given a lead-in and a part question, you will provide a concise answer,
a brief (1-2 sentences) explanation, and your confidence in the guess.
The answer should be a single word or a short phrase, and the explanation should be concise and relevant.
Provide the output in the JSON format:

Don't think for more than 1000 words, and don't use more than 2000 tokens in total.

{
    "answer": <str>,
    "explanation": <str>,
    "confidence": <float between 0 and 1, in steps of 0.01>,
    "justification": <str, optional justification for the confidence score>
}
"""

user_prompt_template = """
Leadin: {leadin}
{previous_parts_text}
Question: {part}
What is being asked in the question? Provide a concise answer, a brief explanation, and your confidence in
the guess along with justification.
"""


class QWenReasonerBonusPipeline(Pipeline):
    """
    A custom pipeline for quizbowl bonus questions that includes reasoning steps using a reasoner model.

    This pipeline:
      1. Prepares input by combining system and user instructions.
      2. Utilizes the model's thinking capability by enabling 'enable_thinking' in the chat template.
      3. In the postprocess step, extracts both the thinking content and the final answer, along with a computed confidence score.

    Attributes:
        model (PreTrainedModel): The language model used for text generation.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
    """

    def __init__(self, model, tokenizer=None, **kwargs):
        """
        Initialize the pipeline with a model and tokenizer.

        Args:
            model (PreTrainedModel): The model to use for text generation.
            tokenizer (PreTrainedTokenizer, optional): The tokenizer for the model. If not provided,
                it will be automatically loaded from the model.
            **kwargs: Additional keyword arguments for the Pipeline class.
        """
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

        self.think_start_token_id = self.tokenizer.encode("<think>")[0]
        self.think_end_token_id = self.tokenizer.encode("</think>")[0]

    def _sanitize_parameters(self, **kwargs):
        # No additional parameters required.
        return {}, {}, {}

    def preprocess(self, inputs):
        """
        Prepare a batch of input texts by applying a chat template that includes a system directive
        and a user prompt with quizbowl details. Enables thinking mode for the reasoner.

        Args:
            inputs (dict): Dictionary where each value is a list of length batch size,
                           with keys "leadin", "part", and optionally "previous_parts".

        Returns:
            dict: Tokenized batch input for the model.
        """
        batch_size = len(inputs["leadin"])
        conversations = []
        for i in range(batch_size):
            leadin = inputs["leadin"][i]
            part = inputs["part"][i]
            previous_parts_text = ""
            if "previous_parts" in inputs:
                previous_parts = inputs["previous_parts"][i]
                previous_parts_text = format_parts(previous_parts[i])
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt_template.format(
                        leadin=leadin,
                        previous_parts_text=previous_parts_text,
                        part=part,
                    ),
                },
            ]
            conversations.append(messages)
        texts = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return self.tokenizer(texts, return_tensors="pt", padding=True)

    def _forward(self, model_inputs):
        """
        Perform the text generation using the model for a batch of inputs.

        Args:
            model_inputs (dict): Tokenized inputs for the model.

        Returns:
            list[dict]: A list of outputs for each item in the batch.
        """
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=32768,
                # do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            batch_size = model_inputs["input_ids"].shape[0]
            results = {
                "output_ids": [],
                "scores": [],
            }
            for i in range(batch_size):
                input_ids = model_inputs["input_ids"][i]
                output_ids = outputs.sequences[i][len(input_ids) :]
                # outputs.scores is a list of length max_new_tokens, each of shape (batch_size, vocab_size)
                # For each token, take the i-th batch element's score
                scores = (
                    [score[i] for score in outputs.scores] if outputs.scores else []
                )
                results["output_ids"].append(output_ids.tolist())
                results["scores"].append(scores)
            return results

    def _postprocess_single(self, output_ids: list[int], scores: list[float] = None):
        try:
            think_index = len(output_ids) - output_ids[::-1].index(
                self.think_end_token_id
            )
        except ValueError:
            print(
                "No thinking end token found in the output. Setting think_index to 0."
            )
            print(output_ids[-100:])
            think_index = 0
        # If thinking content is present, split the output accordingly.
        if think_index:
            thinking_content = self.tokenizer.decode(
                output_ids[:think_index], skip_special_tokens=True
            ).strip("\n")
            final_content = self.tokenizer.decode(
                output_ids[think_index:], skip_special_tokens=True
            ).strip("\n")
        else:
            thinking_content = ""
            final_content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Extract answer and explanation from the final content.
        answer, explanation = "", ""
        # If JSON parsing fails, attempt to repair the JSON string.
        try:
            json_data = json_repair.loads(final_content)
            answer = json_data.get("answer", "").strip()
            explanation = json_data.get("explanation", "").strip()
            confidence = json_data.get("confidence", 0.0)
        except Exception as e:
            print(f"Error parsing JSON: {e.__class__.__name__} - {e}")
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

        # For tutorial purposes, also print the internal thinking if needed.
        # In production, you might want to log this information instead of printing.
        # print("Internal Thinking:", thinking_content)

        return {
            "answer": answer,
            "confidence": confidence,
            "explanation": explanation,
            "thinking": thinking_content,
        }

    def postprocess(self, model_outputs):
        """
        Process the generated output from the model.

        The output is expected to contain both the "thinking" content (internal reasoning)
        and the final answer content. The method slices the output tokens based on a special
        token (e.g., </think>) if available. It extracts the answer and explanation and calculates
        a confidence score by averaging softmax probabilities over generated tokens.

        Args:
            model_outputs: The raw output from the model.

        Returns:
            dict: A dictionary with keys "answer", "confidence", and "explanation".
        """
        batch_output_ids = model_outputs["output_ids"]
        batch_scores = model_outputs.get("scores", [])
        keys = ["answer", "confidence", "explanation", "thinking"]
        # Initialize results dictionary with empty lists for each key.
        results = {key: [] for key in keys}
        for i, output_ids in enumerate(batch_output_ids):
            logprobs = batch_scores[i] if batch_scores else None
            result = self._postprocess_single(output_ids, logprobs)
            for key in keys:
                results[key].append(result.get(key, ""))
        return results

        # Attempt to parse the output that may include a thinking portion.
        # The special token id for ending thinking (assumed as an example) is 151668.
        # This part separates thinking process (internal) from the final response.


# Register the custom pipeline with Hugging Face's PIPELINE_REGISTRY.
PIPELINE_REGISTRY.register_pipeline(
    "quizbowl-bonus",
    pipeline_class=QWenReasonerBonusPipeline,
    pt_model=AutoModelForCausalLM,
    tf_model=TFAutoModelForCausalLM,
    default={
        # Specify the reasoner model; for example purposes, we use a placeholder model.
        "pt": ("Qwen/Qwen3-0.6B", "main"),
    },
    type="text",
)
# %%
if __name__ == "__main__":
    # Example usage of the pipeline.
    dataset_name = "qanta-challenge/qanta25-eval"
    dataset = load_dataset(dataset_name, "bonus", split="tiny_eval")

    examples = []
    if len(dataset) > 10:
        dataset = dataset.select(range(10))  # Limit to first 10 examples for testing.:
    for e in dataset:
        for part in e["parts"]:
            examples.append(
                {
                    "leadin": e["leadin"],
                    "part": part["question"],
                }
            )

    # Print the first example to verify the input format.
    print("Example input:", examples[0])

    # Load the model and tokenizer.
    model_name = "Qwen/Qwen3-0.6B"  # Replace with your actual model name.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # %%
    pipe = QWenReasonerBonusPipeline(model=model, tokenizer=tokenizer)

    part_dataset = Dataset.from_list(examples)

    outputs = pipe(part_dataset, batch_size=1)
    for e, o in zip(examples, outputs):
        print(f"Input: {e['part']}")
        print(
            f"Output: {o['answer']}, Confidence: {o['confidence']:.2f}, Explanation: {o['explanation']}"
        )
        print("-" * 80)

# %%
