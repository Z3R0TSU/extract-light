# extract-light

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Model: LLaMA 3.2 1B](https://img.shields.io/badge/model-LLaMA%203.2%201B-orange.svg)](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)

Lightweight LoRA fine-tuning pipeline for natural language to SQL generation. Fine-tunes LLaMA 3.2-1B-Instruct with 4-bit quantization to keep training and inference runnable on a single consumer GPU.

---

## The Problem

Most text-to-SQL approaches hallucinate column and table names because the model has no knowledge of the actual schema. The output is syntactically valid but wrong.

This project fixes that by passing a `CREATE TABLE` statement as context with every query. The model generates SQL grounded in the real schema instead of guessing.

---

## Design Decisions

**Why LLaMA 3.2-1B and not a larger model?**
The goal was to show that a small model, fine-tuned on domain-specific data, can produce accurate SQL without needing large infrastructure. It runs on a single GPU with quantization.

**Why LoRA?**
LoRA freezes the base model and only trains a small set of adapter weights (roughly 1-2% of total parameters). This cuts memory requirements significantly and produces a ~45MB adapter file that can be swapped onto the base model anywhere.

**Why 4-bit quantization?**
Reduces the model's memory footprint by ~75% with minimal impact on output quality for structured tasks like SQL generation. Combined with LoRA, the full pipeline fits on a 16GB GPU.

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Llama-3.2-1B-Instruct` |
| Dataset | `b-mc2/sql-create-context` (78,577 NL to SQL pairs) |
| Training samples | 1,000 |
| Training steps | 60 |
| LoRA rank | 16 |
| Quantization | 4-bit (bitsandbytes) |
| Adapter size | ~45MB |

The dataset pairs natural language questions with `CREATE TABLE` context and correct SQL answers. This teaches the model schema-grounded generation rather than free-form guessing.

---

## Input Format

Every inference call uses this structure:

```
Question: <natural language query>
Schema:   <CREATE TABLE statement>
Answer:   <generated SQL>
```

**Example 1 - Join query:**
```
Question: Show the themes of competitions hosted in cities with population over 1000.
Schema:   CREATE TABLE city (City_ID VARCHAR, Population INTEGER);
          CREATE TABLE farm_competition (Theme VARCHAR, Host_city_ID VARCHAR)
Answer:   SELECT T2.Theme FROM city AS T1
          JOIN farm_competition AS T2 ON T1.City_ID = T2.Host_city_ID
          WHERE T1.Population > 1000
```

**Example 2 - Aggregation:**
```
Question: Show the different city statuses and average population for each.
Schema:   CREATE TABLE city (Status VARCHAR, Population INTEGER)
Answer:   SELECT Status, AVG(Population) FROM city GROUP BY Status
```

---

## Sample Inference Queries

Test queries run after training to check the adapter:

```python
tests = [
    {
        "q": "List the names of students who enrolled in 'Computer Science' and have a GPA higher than 3.5, ordered by GPA descending.",
        "c": "CREATE TABLE students (name VARCHAR, gpa FLOAT, major VARCHAR)"
    },
    {
        "q": "Calculate the average salary of employees in the 'Engineering' department who were hired before 2020.",
        "c": "CREATE TABLE employees (salary INT, department VARCHAR, hire_date DATE)"
    },
    {
        "q": "Find the total number of orders and the sum of total amounts for customer 'John Doe'.",
        "c": "CREATE TABLE orders (id INT, customer_name VARCHAR, total_amount FLOAT)"
    },
    {
        "q": "Show me the top 5 most expensive products that are currently in stock.",
        "c": "CREATE TABLE products (name VARCHAR, price DECIMAL, stock_quantity INT)"
    },
    {
        "q": "Which cities have more than 10 customers? Return the city and customer count.",
        "c": "CREATE TABLE customers (id INT, city VARCHAR)"
    }
]
```

---

## Quick Start

### Requirements

- Python 3.10
- A CUDA-capable GPU
- Git

### Setup

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/Z3R0TSU/extract-light.git
cd extract-light
python3.10 -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

Install PyTorch first. Use whichever CUDA version matches your driver:

```bash
# CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets trl pyyaml
```

### Train

```bash
python train.py
```

Downloads the dataset, runs 60 training steps, and saves the adapter to `models/sql_specialist_lora/`. Took about 10 minutes on a single GPU.

### Inference

```bash
python inference.py
```

Loads the saved adapter and runs the five test queries. SQL output streams to the terminal for each one.

---

## Project Structure

```
extract-light/
├── train.py          # fine-tuning pipeline
├── inference.py      # load adapter and run queries
├── config.yaml       # model and training config
├── pyproject.toml    # dependencies
└── models/
    └── sql_specialist_lora/   # saved LoRA adapter (~45MB)
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base model | LLaMA 3.2-1B-Instruct via Unsloth |
| Fine-tuning | LoRA via PEFT |
| Quantization | 4-bit via bitsandbytes |
| Training | TRL SFTTrainer |
| Dataset | b-mc2/sql-create-context on HuggingFace |

---

## Limitations

- Trained on 1,000 of 78,577 available samples. More training steps on the full dataset would improve accuracy on complex multi-join queries.
- No constrained decoding. Adding grammar-constrained generation would catch any remaining syntax errors.
- No formal benchmark evaluation yet. Testing against Spider or BIRD would give a real accuracy number.
