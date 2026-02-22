# extract-light — Lightweight SQL Specialist via LoRA Fine-Tuning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: LLaMA 3.2 1B](https://img.shields.io/badge/model-LLaMA%203.2%201B-orange.svg)](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)

A lightweight, schema-aware natural language to SQL system built by fine-tuning LLaMA 3.2-1B-Instruct using LoRA adapters and 4-bit quantization. Designed for efficient, edge-friendly deployment — not maximum compute.

---

## The Problem with Naive Text-to-SQL

Most text-to-SQL demos fail silently: they generate syntactically valid SQL that references tables and columns that don't exist. A model that doesn't know your schema will hallucinate column names with confidence.

extract-light addresses this by taking a `CREATE TABLE` statement as explicit context alongside every query. The model generates SQL grounded in the actual schema — the same approach used by production tools like GitHub Copilot and Cursor's SQL features.

---

## Design Decisions

**Why LLaMA 3.2-1B instead of a larger model?**
Larger models generate better SQL out of the box, but require significant infrastructure to deploy and fine-tune. The goal here was to demonstrate that a 1B parameter model, properly fine-tuned on domain-specific data, can produce high-quality SQL — and run on a single consumer GPU or CPU with quantization.

**Why LoRA instead of full fine-tuning?**
LoRA (Low-Rank Adaptation) freezes the base model weights and trains a small set of adapter parameters (~1-2% of total parameters). This reduces memory requirements dramatically while preserving base model knowledge. The resulting adapter is ~45MB — deployable anywhere the base model runs.

**Why 4-bit quantization?**
4-bit quantization via bitsandbytes reduces the model's memory footprint by ~75% with minimal quality loss on structured generation tasks like SQL. Combined with LoRA, the full fine-tuning pipeline runs on a single 16GB GPU.

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Llama-3.2-1B-Instruct` |
| Dataset | `b-mc2/sql-create-context` (78,577 NL→SQL pairs) |
| Training samples | 1,000 |
| Training steps | 60 |
| LoRA rank | 16 |
| Quantization | 4-bit (bitsandbytes) |
| Adapter size | ~45MB |

The dataset provides natural language questions paired with `CREATE TABLE` context and target SQL — ensuring the model learns schema-grounded generation rather than schema-free hallucination.

---

## Input Format

Every inference call follows this structure:

```
Question: <natural language query>
Schema:   <CREATE TABLE statement>
Answer:   <generated SQL>
```

**Example 1 — Join query:**
```
Question: Show the themes of competitions hosted in cities with population over 1000.
Schema:   CREATE TABLE city (City_ID VARCHAR, Population INTEGER);
          CREATE TABLE farm_competition (Theme VARCHAR, Host_city_ID VARCHAR)
Answer:   SELECT T2.Theme FROM city AS T1
          JOIN farm_competition AS T2 ON T1.City_ID = T2.Host_city_ID
          WHERE T1.Population > 1000
```

**Example 2 — Aggregation:**
```
Question: Show the different city statuses and average population for each.
Schema:   CREATE TABLE city (Status VARCHAR, Population INTEGER)
Answer:   SELECT Status, AVG(Population) FROM city GROUP BY Status
```

---

## Sample Inference Queries

Test queries used to evaluate the fine-tuned adapter post-training:

```python
queries = [
    {
        "question": "List students enrolled in 'Computer Science' with GPA > 3.5, ordered by GPA descending.",
        "schema": "CREATE TABLE students (name VARCHAR, gpa FLOAT, major VARCHAR)"
    },
    {
        "question": "Calculate the average salary of Engineering employees hired before 2020.",
        "schema": "CREATE TABLE employees (salary FLOAT, department VARCHAR, hire_date DATE)"
    },
    {
        "question": "Find the total number of orders and sum of amounts for customer 'John Doe'.",
        "schema": "CREATE TABLE orders (id INT, customer_name VARCHAR, total_amount FLOAT)"
    },
    {
        "question": "Show the top 5 most expensive products currently in stock.",
        "schema": "CREATE TABLE products (name VARCHAR, price FLOAT, stock_quantity INT)"
    },
    {
        "question": "Which cities have more than 10 customers?",
        "schema": "CREATE TABLE customers (id INT, city VARCHAR)"
    }
]
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/Z3R0TSU/extract-light.git
cd extract-light
uv sync
```

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. `uv sync` installs everything, including the pinned CUDA 12.1 wheels for PyTorch.

### Dependencies (manual install)

If installing manually without `uv sync`:

```bash
uv pip install torch==2.3.1 torchvision==0.18.1 xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121
uv pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
uv pip install transformers datasets trl pyyaml
```

### Fine-tune the model

```bash
python train.py
```

Adapter weights are saved to `models/sql_specialist_lora/`.

### Run inference

```bash
python inference.py
```

Loads the saved LoRA adapter and runs the test queries defined in the script.

---

## Project Structure

```
extract-light/
├── train.py          # LoRA fine-tuning pipeline
├── inference.py      # Load adapter and run sample queries
├── config.yaml       # Training hyperparameters
├── pyproject.toml    # Project dependencies
└── models/
    └── sql_specialist_lora/   # Saved LoRA adapter weights (~45MB)
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base Model | LLaMA 3.2-1B-Instruct (Unsloth) |
| Fine-tuning | LoRA via PEFT |
| Quantization | 4-bit (bitsandbytes) |
| Training Framework | TRL (SFTTrainer) |
| Dataset | b-mc2/sql-create-context (HuggingFace) |

---

## Limitations and Future Work

- Trained on 1,000 samples for 60 steps — extended training on the full 78K dataset would improve accuracy on complex multi-join queries
- No beam search or constrained decoding — adding grammar-constrained generation would eliminate any remaining syntax errors
- Evaluation against Spider or BIRD benchmark would provide a formal accuracy baseline
