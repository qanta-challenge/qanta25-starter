# %%
# ----------------------------------------------------------
# Custom Hugging-Face pipeline for the “tossup” split that only refers to the existing models
# Task id  :  quizbowl-tossup
# Expected input keys : question_text
# Must return        : answer, confidence, buzz
# ----------------------------------------------------------

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    Pipeline,
    TFAutoModelForCausalLM,
)
from transformers.pipelines import PIPELINE_REGISTRY


class QBTossupPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        # We don't need any additional parameters for this pipeline
        return {}, {}, {}

    def preprocess(self, inputs):
        prompt = (
            "Answer the quiz question revealed so far:\n"
            f"Question: {inputs['question_text']}\nAnswer:"
        )
        return self.tokenizer(prompt, return_tensors="pt", truncation=True)

    def _forward(self, model_inputs):
        with torch.no_grad():
            return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits  # shape: (1, seq_len, vocab)
        last_logits = logits[0, -1]  # take distribution for last token
        probs = F.softmax(last_logits, dim=-1)
        top_id = torch.argmax(probs)
        answer = self.tokenizer.decode(top_id, skip_special_tokens=True).strip()
        confidence = probs[top_id].item()
        buzz = confidence > 0.5
        return {"answer": answer, "confidence": confidence, "buzz": buzz}


PIPELINE_REGISTRY.register_pipeline(
    "quizbowl-tossup",
    pipeline_class=QBTossupPipeline,
    pt_model=AutoModelForCausalLM,
    tf_model=TFAutoModelForCausalLM,
    default={
        "pt": ("yujiepan/llama-3-tiny-random", "main"),
    },
    type="text",
)
