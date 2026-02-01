# SQL Specialist - LoRA Fine-Tuning

This project implements a lightweight pipeline to fine-tune an LLM (using Unsloth) to convert Natural Language Questions into SQL Queries.

## 📂 Project Structure

- `train.py`: Fine-tunes the model using LoRA adapters on the `b-mc2/sql-create-context` dataset.
- `inference.py`: Loads the fine-tuned model and runs inference on new natural language queries.
- `config.yaml`: Configuration file for model parameters and training hyperparameters.

## 🚀 Setup

Prerequisites:
- Python 3.10+
- NVIDIA GPU with CUDA support
- `uv` (recommended for dependency management)

### Installation

1. **Install Dependencies**
   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   uv pip install transformers datasets trl
   ```

## 🧠 Usage

### 1. Training
To start fine-tuning the model:
```bash
uv run train.py
```
This will save the adapters to `models/sql_specialist_lora`.

### 2. Inference
To test the model with natural language queries:
```bash
uv run inference.py
```

## 📊 Example Output
**Question:** "Show me the top 5 most expensive products that are currently in stock."
**SQL Generated:** 
```sql
SELECT name FROM products WHERE stock_quantity > 0 ORDER BY price DESC LIMIT 5
```
