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
You are a quizbowl player. Given a leadin and your responses to previous related parts,
provide the answer, a brief (1-2 sentences) explanation, and your confidence in the guess.
The answer should be a single word or a short phrase, and the explanation should be concise and relevant.
Provide the output in the JSON format:

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
        Prepare the input text by applying a chat template that includes a system directive
        and a user prompt with quizbowl details. Enables thinking mode for the reasoner.

        Args:
            inputs (dict): Dictionary containing "leadin", "part", and "previous_parts".

        Returns:
            str: The formatted text input for the model.
        """
        leadin = inputs.get("leadin", "")
        part = inputs.get("part", "")
        previous_parts_text = format_parts(inputs.get("previous_parts", []))

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
        # Apply chat template with enable_thinking=True to trigger reasoning output.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        return self.tokenizer(text, return_tensors="pt")

    def _forward(self, model_inputs):
        """
        Perform the text generation using the model.

        Args:
            model_inputs (dict): Tokenized inputs for the model.

        Returns:
            ModelOutput: The output from the model's generate function.
        """
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=32768,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = outputs.sequences[0]
            input_ids = model_inputs["input_ids"][0]
            output_ids = generated_ids[len(input_ids) :]
            return {
                "output_ids": output_ids,
                "scores": outputs.scores,
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
        output_ids = model_outputs["output_ids"].tolist()

        # Attempt to parse the output that may include a thinking portion.
        # The special token id for ending thinking (assumed as an example) is 151668.
        # This part separates thinking process (internal) from the final response.

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
        if "scores" in model_outputs:
            probs = [
                F.softmax(score, dim=-1).max().item()
                for score in model_outputs["scores"]
            ]
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
        print("Internal Thinking:", thinking_content)

        return {"answer": answer, "confidence": confidence, "explanation": explanation}


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
