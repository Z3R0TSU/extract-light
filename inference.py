import os
from unsloth import FastLanguageModel
from transformers import TextStreamer


checkpoint_path = "models/sql_specialist_lora"

if not os.path.exists(checkpoint_path):
    print(f"Error: Model not found at {checkpoint_path}. Run train.py first.")
    exit(1)


print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def run_query(question, context):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Convert the natural language query into a SQL statement.",
                f"Question: {question}\nContext: {context}",
                "", 
            )
        ], return_tensors = "pt").to("cuda")

    print(f"\nQ: {question}")
    print(f"C: {context}")
    print("SQL: ", end="")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# test queries
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

print("\n--- Running Challenging Queries ---")
for t in tests:
    run_query(t["q"], t["c"])