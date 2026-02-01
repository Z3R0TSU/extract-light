import yaml
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_prompts(examples):
    
    inputs       = examples["question"]
    contexts     = examples["context"]
    outputs      = examples["answer"]
    
    texts = []
    for input, context, output in zip(inputs, contexts, outputs):
        full_input = f"Question: {input}\nContext: {context}"
        
        text = alpaca_prompt.format(
            "Convert the natural language query into a SQL statement.",
            full_input,
            output
        ) + "<|end_of_text|>"
        texts.append(text)
    return { "text" : texts, }

def train():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model"]["name"],
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = config["model"]["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = config["lora"]["r"],
        target_modules = config["lora"]["target_modules"],
        lora_alpha = config["lora"]["lora_alpha"],
        lora_dropout = config["lora"]["lora_dropout"],
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
    )

    
    dataset = load_dataset("b-mc2/sql-create-context", split="train[:1000]")
    dataset = dataset.map(format_prompts, batched = True)
    

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = config["training"]["batch_size"],
            gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"],
            warmup_steps = 5,
            max_steps = config["training"]["max_steps"],
            learning_rate = float(config["training"]["learning_rate"]),
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "models/checkpoints",
        ),
    )

    trainer_stats = trainer.train()

    model.save_pretrained("models/sql_specialist_lora")
    tokenizer.save_pretrained("models/sql_specialist_lora")

if __name__ == "__main__":
    train()