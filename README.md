# 🎯 QANTA 2025 Starter Kit

Welcome to the QANTA 2025 Challenge! This toolkit provides everything you need to build, run, evaluate, and deploy Quiz Bowl AI systems for both **bonus** and **tossup** question formats.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Setup environment (optional)
bash setup_env.sh

# Install dependencies
pip install -r requirements.txt

```

### 2. Run Your First Pipeline
```bash
# Run a demo pipeline on bonus questions
python run_pipeline.py pipelines.demo.bonus_pipeline \
  --model Qwen/Qwen2.5-3B-Instruct \
  --mode bonus \
  --debug

# Run a demo pipeline on tossup questions  
python run_pipeline.py pipelines.demo.tossup_pipeline \
  --model Qwen/Qwen2.5-3B-Instruct \
  --mode tossup \
  --debug
```

### 3. Evaluate Results
```bash
# Evaluate all bonus models
python evaluate.py --mode bonus

# Evaluate all tossup models
python evaluate.py --mode tossup
```

---
## 🛠️ Creating Custom Pipelines

### Pipeline Structure
Place your pipelines in the appropriate directory:
- **Bonus pipelines**: `pipelines/bonus/`
- **Tossup pipelines**: `pipelines/tossup/`
- Keeping everything flat directly in the `pipelines/` directory is also fine.

### Example: Bonus Pipeline
Create `pipelines/bonus/my_bonus_pipeline.py`:

```python
from transformers import Pipeline
from transformers.pipelines import PIPELINE_REGISTRY

class MyBonusPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        # Convert input dict to model input format
        # inputs: list of dicts with "leadin" and "part" keys
        return inputs

    def _forward(self, model_inputs):
        # Run your model inference
        outputs = self.model.generate(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        # Convert to required format
        return [{
            "answer": "Paris", 
            "confidence": 0.99, 
            "explanation": "Paris is the capital of France."
        }]

# Register the pipeline
PIPELINE_REGISTRY.register_pipeline(
    "quizbowl-bonus",
    MyBonusPipeline,
    pt_model=YourModelClass,
    default={"pt": ("Qwen/Qwen2.5-3B-Instruct", "main")},
    type="text",
)
```

### Example: Tossup Pipeline
Create `pipelines/tossup/my_tossup_pipeline.py`:

```python
from transformers import Pipeline
from transformers.pipelines import PIPELINE_REGISTRY

class MyTossupPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        # Convert input dict to model input format
        # inputs: list of dicts with "question_text" key
        return inputs

    def _forward(self, model_inputs):
        # Run your model inference
        outputs = self.model.generate(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        # Convert to required format
        return [{
            "answer": "Isaac Newton", 
            "confidence": 0.95, 
            "buzz": True
        }]

# Register the pipeline
PIPELINE_REGISTRY.register_pipeline(
    "quizbowl-tossup",
    MyTossupPipeline,
    pt_model=YourModelClass,
    default={"pt": ("Qwen/Qwen2.5-3B-Instruct", "main")},
    type="text",
)
```

---


## 📋 Main Entry Points

This toolkit provides three main scripts for different stages of the ML pipeline:

### 🔧 `run_pipeline.py` - Pipeline Runner
**Purpose**: Execute your custom pipelines on Quiz Bowl datasets with flexible configuration options.

**Key Features**:
- ✅ Automatic pipeline discovery and registration
- ✅ Batch processing with configurable batch sizes  
- ✅ Resume functionality (skip already processed examples)
- ✅ Packet filtering (process specific question packets)
- ✅ Debug mode for development

**Basic Usage**:
```bash
python run_pipeline.py <pipeline_module> --model <model_name> --mode <bonus|tossup>
```

**Examples**:
```bash
# Run bonus pipeline with packet filtering
python run_pipeline.py pipelines.bonus.my_pipeline \
  --model Qwen/Qwen2.5-3B-Instruct \
  --mode bonus \
  --packets "1-5,7,10" \
  --batch_size 8

# Debug mode (limits to 3 examples)
python run_pipeline.py pipelines.tossup.my_pipeline \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --mode tossup \
  --debug

# Resume from existing outputs
python run_pipeline.py pipelines.bonus.my_pipeline \
  --model Qwen/Qwen2.5-3B-Instruct \
  --mode bonus \
  --reprocess
```

### 📊 `evaluate.py` - Model Evaluation
**Purpose**: Evaluate model performance with comprehensive metrics and beautiful reporting.

**Key Features**:
- ✅ Automatic evaluation of all models in outputs directory
- ✅ Support for both bonus and tossup task evaluation
- ✅ Comprehensive metrics calculation and reporting
- ✅ Beautiful tabular output with rich formatting

**Basic Usage**:
```bash
python evaluate.py --mode <bonus|tossup> [--model <model_name>]
```

**Examples**:
```bash
# Evaluate all bonus models
python evaluate.py --mode bonus

# Evaluate specific model
python evaluate.py --mode tossup --model Qwen/Qwen2.5-3B-Instruct

# Use custom dataset
python evaluate.py --mode bonus --dataset qanta-challenge/custom-dataset
```

### 📤 `push_pipeline.py` - Pipeline Publisher  
**Purpose**: Publish your custom pipelines to Hugging Face Hub for sharing and deployment.

**Key Features**:
- ✅ Automatic pipeline registration and validation
- ✅ Support for both bonus and tossup tasks
- ✅ One-command publishing to Hugging Face Hub
- ✅ Comprehensive error handling

**Basic Usage**:
```bash
python push_pipeline.py <pipeline_class> --task <bonus|tossup> --repo-id <repo_name>
```

**Examples**:
```bash
# Push bonus pipeline
python push_pipeline.py \
  pipelines.llama3_bonus.BonusPipeline \
  --task bonus \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --repo-id my-username/my-llama3-bonus-pipeline

# Push tossup pipeline  
python push_pipeline.py \
  pipelines.qwen_tossup.TossupPipeline \
  --task tossup \
  --model Qwen/Qwen2.5-3B-Instruct \
  --repo-id my-username/my-qwen-tossup-pipeline
```

---


## 📈 Understanding Evaluation output Formats

### Tossup Task Output  
```json
{
    "qid": "tossup-question-id",
    "run_number": 3,
    "answer": "Isaac Newton",
    "confidence": 0.88,
    "buzz": true,
    "correct": true,
    "token_position": 42
}
```

### Bonus Task Output
```json
{
    "qid": "bonus-question-id",
    "part_number": 1,
    "answer": "Paris",
    "confidence": 0.95,
    "explanation": "Paris is the capital of France.",
    "correct": true
}
```


---

## 🎯 Tips for Success

### 1. **Start Simple**: Begin with the provided templates and gradually add complexity
### 2. **Use Debug Mode**: Always test with `--debug` flag first to catch issues early
### 3. **Monitor Performance**: Use `evaluate.py` frequently to track improvements
### 4. **Experiment with Models**: Try different base models for different performance characteristics
### 5. **Optimize Batch Size**: Balance speed vs memory usage with `--batch_size`

---

## 📁 Project Structure

```
qanta25-starter/
├── pipelines/              # Your custom pipeline implementations
│   ├── bonus/              # Bonus question pipelines
│   └── tossup/             # Tossup question pipelines
├── outputs/                # Generated model predictions
│   └── {dataset}/
│       ├── bonus/
|       |   └─{model_name}  # Bonus results (.jsonl files)
│       └── tossup/         
|           └─{model_name}  # Tossup results (.jsonl files)
├── metrics/                # Evaluation metrics and utilities
├── utils/                  # Shared utilities and helpers
├── run_pipeline.py         # 🔧 Main pipeline runner
├── evaluate.py             # 📊 Model evaluation suite
├── push_pipeline.py        # 📤 Pipeline publisher
└── README.md               # This file!
```

---

## 🆘 Troubleshooting

### Common Issues:

**"Pipeline not found"**: Make sure your pipeline is properly registered and the module path is correct.

**"CUDA out of memory"**: Reduce `--batch_size` or use `--device-map cpu`.

**"No outputs found"**: Check that your pipeline ran successfully and outputs are in the correct directory.

**"Import error"**: Ensure all dependencies are installed with `pip install -r requirements.txt`.

---

## 🎉 Ready to Compete!

You're now ready to build amazing Quiz Bowl AI systems! Remember:

1. **Build** your pipeline using the templates
2. **Run** it with `run_pipeline.py`
3. **Evaluate** performance with `evaluate.py` 
4. **Share** your best models with `push_pipeline.py`

Good luck in the QANTA 2025 Challenge! 🏆

---

**Need more help?** Reach out to us at [qanta@googlegroups.com](mailto:qanta@googlegroups.com) with your questions. We're happy to help!