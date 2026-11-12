# NimbusPay Support Ticket Triage

Fine-tuned **Qwen2.5-1.5B-Instruct** with QLoRA to classify NimbusPay support tickets into structured 8-field JSON records.

## Results

| Metric | Score |
|--------|-------|
| Exact Match (dev) | **88.5%** |
| Mean Field Accuracy (dev) | **97.8%** |
| Trainable Parameters | ~12.8M |
| Peak GPU Memory | ~7.2 GB |

## Approach

- **Quantization**: 4-bit NF4 with double quantization (QLoRA)
- **LoRA**: r=16, alpha=32, applied to all attention + MLP projections
- **Training**: 5 epochs, cosine LR schedule, batch=2 x grad_accum=8
- **Data**: Cleaned 1,445 -> 1,411 rows (fixed schema mismatches, dropped garbage)

## Repo Structure

```
├── README.md              # This file
├── report.md              # 3-page technical report
├── notebook.ipynb         # Colab notebook with all outputs
├── cleaning_log.csv       # Data audit log (Part A)
├── predictions_test.jsonl # 400 test predictions
├── adapter/               # PEFT LoRA adapter (best model)
└── score.py               # Provided scoring script
```

## Reproduce

```bash
pip install transformers peft bitsandbytes torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "./adapter")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```