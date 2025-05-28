# %%
# ----------------------------------------------------------
# Custom Hugging-Face pipeline for the “bonus” split that refers to the existing models
# Task id  :  quizbowl-bonus
# Expected input keys : leadin, part, previous_parts ('text' and 'guess')
# Must return        : answer, confidence, explanation
# ----------------------------------------------------------

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    Pipeline,
    TFAutoModelForCausalLM,
)
from transformers.pipelines import PIPELINE_REGISTRY


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
{previous_parts_text}
Question: {part}"
What is being asked in the question? Provide a concise answer, a brief explanation, and your confidence in the guess along with justification."""


class QBBonusPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        # No additional parameters needed
        return {}, {}, {}

    def preprocess(self, inputs):
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

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _forward(self, model_inputs):
        with torch.no_grad():
            return self.model.generate(
                **model_inputs,
                max_new_tokens=32,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

    def postprocess(self, model_outputs):
        generated_ids = model_outputs.sequences[0]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Remove the system prompt from the output
        output_text = output_text.replace(system_prompt, "").strip()

        # Try to extract answer and explanation from output_text
        # Assume format: "Answer: ... Explanation: ..."
        answer, explanation = "", ""
        if "Explanation:" in output_text:
            parts = output_text.split("Explanation:", 1)
            answer = parts[0].replace("Answer:", "").strip()
            explanation = parts[1].strip()
        else:
            # Fallback: first word as answer, rest as explanation
            tokens = output_text.split()
            answer = tokens[0] if tokens else ""
            explanation = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        # Confidence: use the mean max probability of generated tokens (approximation)
        if hasattr(model_outputs, "scores") and model_outputs.scores:
            probs = [
                F.softmax(score, dim=-1).max().item() for score in model_outputs.scores
            ]
            confidence = float(sum(probs) / len(probs)) if probs else 0.0
        else:
            confidence = 0.0
        return {"answer": answer, "confidence": confidence, "explanation": explanation}


PIPELINE_REGISTRY.register_pipeline(
    "quizbowl-bonus",
    pipeline_class=QBBonusPipeline,
    pt_model=AutoModelForCausalLM,
    tf_model=TFAutoModelForCausalLM,
    default={
        "pt": ("yujiepan/llama-3-tiny-random", "main"),
    },
    type="text",
)
