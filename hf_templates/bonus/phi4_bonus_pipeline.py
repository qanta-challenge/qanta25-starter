# %%
# ----------------------------------------------------------
# Custom Hugging-Face pipeline for the “bonus” split using Phi4 model.
# Task id  : quizbowl-phi4-bonus
# Expected input keys : leadin, part
# Must return        : answer, confidence, explanation
# ----------------------------------------------------------
import json
import sys

import json_repair
import torch
import torch.nn.functional as F
from datasets import Dataset
from loguru import logger
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY

logger.remove()
logger.add(sys.stdout, diagnose=False, colorize=True, level="INFO", backtrace=True)

# --------------------------------------------------------------------------
# Prompts and conversation preparation (from phi4.py)
# --------------------------------------------------------------------------
Phi4_SYSTEM_PROMPT_orig = (
    "You are Phi, a language model trained by Microsoft to help users. Your role as a Quizbowl assistant "
    "involves thoroughly exploring questions through a systematic thinking process before providing the final "
    "precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, "
    "exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and Solution using the specified format: "
    "<think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. "
    "Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new "
    "ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, "
    "based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution "
    "that you deem correct. Now, try to solve the following question through the above guidelines:"
)

Phi4_SYSTEM_PROMPT_revised = (
    "You are Phi, a language model trained by Microsoft to help users. As a Quizbowl assistant, your role is to explore questions "
    "thoroughly through a systematic thinking process, while being judicious with words and keeping your thoughts clear and concise, not overly verbose. "
    "Engage in a cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop a well-considered reasoning process. "
    "Please structure your response into two main sections: Thought and Solution using the specified format: "
    "<think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps, ensuring each step is succinct and focused. "
    "Each step should include considerations such as analyzing the question, summarizing relevant findings, brainstorming ideas, verifying accuracy, refining errors, and revisiting previous steps. "
    "Limit the Thought section to no more than {thinking_budget} tokens to maintain clarity and efficiency. But make sure to end the though section using </think> and then continue with the Solution section. "
    "In the Solution section, based on your reasoning, systematically present the final answer you deem correct. "
    "Now, try to solve the following question following these guidelines:"
)

task_instruction = """
Given a lead-in text and a quiz question, provide your best guess for the answer, a brief (1-2 sentences) explanation for the answer along with a well-calibrated confidence in the guess.
The answer should be a single word or short phrase, and the explanation should be concise and relevant to the question.
The confidence should be calibrated and very well-reasoned.
Also provide a justification for the confidence score, explaining why you chose that value.

Format your response in below JSON format:

{
    "answer": str,
    "explanation": str,
    "confidence": float (0-1 in the steps of 0.01)
    "justification": str (optional justification for the confidence score)
}"""
user_prompt_template = """
Leadin: {leadin}
Question: {part}"""

chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant<|im_sep|>' }}{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"""


def prepare_conversation(leadin, part):
    messages = [
        {
            "role": "system",
            "content": Phi4_SYSTEM_PROMPT_revised.replace(
                "{thinking_budget}", str(100)
            ),  # Adjust thinking budget as needed
        },
        {
            "role": "user",
            "content": task_instruction
            + user_prompt_template.format(leadin=leadin, part=part),
        },
    ]
    return messages


def postprocess_response(output_text, scores=None):
    # Extract answer, explanation, confidence, and justification from the generated JSON output.

    # EXTRACT THINKING AND FINAL CONTENT
    if "</think>" in output_text:
        thinking, final_content = output_text.rsplit("</think>", 1)
    else:
        final_content = output_text
        thinking = ""
    if "<think>" in thinking:
        thinking = thinking.split("<think>")[-1].strip()

    # EXTRACT JSON CONTENT
    try:
        start_index = final_content.find("{")
        if start_index == -1:
            raise ValueError("No JSON object found in the output text.")
        final_content = final_content[start_index:]
        json_data = json_repair.loads(final_content)
        if isinstance(json_data, list):
            json_data = json_data[0]
        answer = json_data.get("answer", "").strip()
        explanation = json_data.get("explanation", "").strip()
        confidence = json_data.get("confidence", 0.0)
        justification = json_data.get("justification", "").strip()
    except Exception as e:
        logger.exception(
            f"Error parsing JSON: {e.__class__.__name__} - {e}. Got:\n{final_content}\nFull output:\n{output_text}"
        )
        return {"answer": "", "confidence": 0.0, "explanation": "", "justification": ""}

    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        confidence = 0.0  # Default to 0.0 if conversion fails
    return {
        "thinking": thinking,
        "answer": answer,
        "confidence": confidence,
        "explanation": explanation,
        "justification": justification,
    }


# --------------------------------------------------------------------------
# Pipeline class for Phi4 Bonus
# --------------------------------------------------------------------------
class Phi4BonusPipeline(Pipeline):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.tokenizer.padding_side = "left"

    def _sanitize_parameters(self, **kwargs):
        # No additional parameters to sanitize
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
            padding_side="left",
            return_tensors="pt",
        )
        return model_inputs

    def _forward(self, model_inputs):
        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=32768,  # Adjusted for larger outputs
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_length = model_inputs["input_ids"].shape[1]
        # Remove the prompt tokens from the generated output sequences.
        outputs.sequences = outputs.sequences[:, input_length:]
        if outputs.scores:
            outputs.scores = torch.stack(outputs.scores, dim=1)
        return outputs

    def postprocess(self, model_outputs):
        output_texts = self.tokenizer.batch_decode(
            model_outputs.sequences, skip_special_tokens=True
        )
        records = [postprocess_response(output_text) for output_text in output_texts]
        return records


# --------------------------------------------------------------------------
# Pipeline registration
# --------------------------------------------------------------------------
PIPELINE_REGISTRY.register_pipeline(
    "quizbowl-bonus",
    pipeline_class=Phi4BonusPipeline,
    pt_model=AutoModelForCausalLM,
    default={
        "pt": ("microsoft/Phi-4-reasoning", "main"),
    },
    type="text",
)

# --------------------------------------------------------------------------
# Standalone testing (if run as a script)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    pipe = pipeline(
        "quizbowl-bonus",
        # "microsoft/Phi-4-reasoning",
        # "microsoft/Phi-4-mini-reasoning",
        "microsoft/phi-4",
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
            "leadin": "Another leadin here.",
            "part": "What is the largest planet in our solar system?",
        },
        {
            "leadin": "Leadin without history.",
            "part": "What is the chemical symbol for water?",
        },
    ] * 2

    dataset = Dataset.from_list(examples)
    print("Dataset size:", len(dataset))
    outputs = []
    batch_size = 2
    for batch in tqdm(dataset.batch(batch_size), desc="Processing batches"):
        output = pipe(batch, batch_size=batch_size)
        outputs.extend(output)
    print("Outputs:")
    print(outputs)
