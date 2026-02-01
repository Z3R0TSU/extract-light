# SQL Specialist - LoRA Fine-Tuning

This project implements a lightweight pipeline to fine-tune an LLM (using Unsloth) to convert Natural Language Questions into SQL Queries.

## Project Structure

- `train.py`: Fine-tunes the model using LoRA adapters on the `b-mc2/sql-create-context` dataset.
- `inference.py`: Loads the fine-tuned model and runs inference on new natural language queries.
- `config.yaml`: Configuration file for model parameters and training hyperparameters.


### Installation

**Install Dependencies**
   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   uv pip install transformers datasets trl
   ```

