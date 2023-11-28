import os
import re
import json
from tqdm.notebook import tqdm
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch

from datasets import Dataset, DatasetDict, load_from_disk, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
from peft import (
    PeftModel,
    PeftConfig,
    LoraConfig,
    get_peft_model,
)
from accelerate import Accelerator
import wandb

import warnings
warnings.filterwarnings("ignore")

# parameters
model_name_or_path = "meta-llama/Llama-2-7b-hf"
SEED = 42

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed_data_50tweets")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--train_all_linear_layers", type=str2bool, default=False)
    ap.add_argument("--output_dir", type=str, default="lora_result/")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--save_name", type=str, default="final_model")
    ap.add_argument("--dimension_id", type=int, default=None)
    ap.add_argument("--wandb_project", type=str, default=None)
    args = ap.parse_args()
    return (
        args.data_dir,
        args.lora_r,
        args.train_all_linear_layers,
        args.output_dir,
        args.per_device_train_batch_size,
        args.per_device_eval_batch_size,
        args.gradient_accumulation_steps,
        args.num_train_epochs,
        args.learning_rate,
        args.save_name,
        args.dimension_id,
        args.wandb_project,
    )


def load_data(data_dir: str, dimension_id: int, tokenizer):
    # Load preprocessed dataset
    dataset = load_from_disk(data_dir).shuffle(seed=SEED)

    # Case of binary classification
    if dimension_id is not None:
        assert dimension_id in range(0, 4)
        
        # Extract Nth dimension from 4-dimension labels
        def extract_nth_dimension(example, n):
            new_label = example['label'][n]
            return {'label': new_label}
        
        dataset = dataset.map(lambda example: extract_nth_dimension(example, dimension_id))
    
    # Dictionary to switch labels and IDs
    label2id = {l: id for id, l in enumerate(sorted(set(dataset['train']['label'])))}
    id2label = {id: l for l, id in label2id.items()}

    def get_tokenized(dataset, tokenizer):
        def tokenize_add_label(batch):
            batch["input_ids"] = tokenizer(batch["text"]).input_ids
            batch["labels"] = label2id[batch['label']]
            return batch

        dataset = dataset.map(tokenize_add_label, remove_columns=dataset.column_names["train"], num_proc=4)
        return dataset

    tokenized_dataset = get_tokenized(dataset, tokenizer) 
    train_data = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=SEED) # dataset for fine-tuning; train -> train data, test -> eval data

    return train_data, label2id, id2label


def main():
    (
        data_dir,
        lora_r,
        train_all_linear_layers,
        output_dir,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        gradient_accumulation_steps,
        num_train_epochs,
        learning_rate,
        save_name,
        dimension_id,
        wandb_project,
    ) = parse_args()
    
    # wandb setting
    #if local_rank == 0:
    wandb.init(
        project = wandb_project,
    )

    """
    Load datasets
    """    
    # Tokenize text data
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare dataset for fine-tuning
    train_data, label2id, id2label = load_data(data_dir, dimension_id, tokenizer)

    
    # Build a model
    #torch.cuda.set_device(local_rank)
    #device = torch.device(f"cuda:{local_rank}")
    device = "auto"

    """
    Load Llama 2 7B and set config for LoRA tuning
    """    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        device_map=device,
        use_flash_attention_2=True,
    )
    if getattr(model.config, "pad_token_id") is None:
        model.config.pad_token_id = model.config.eos_token_id
        
    # Set target_modules for LoRA
    if train_all_linear_layers:
        model_modules = str(model.modules)
        pattern = r'\((\w+)\): Linear'
        linear_layer_names = re.findall(pattern, model_modules)

        names = []
        # Print the names of the Linear layers
        for name in linear_layer_names:
            names.append(name)
        target_modules = list(set(names))
    else:
        target_modules = ['q_proj', 'v_proj']

    print(f"target modules for LoRA: {target_modules}")

    # Set configuration for LoRA
    config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=lora_r,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    """
    Set Trainer
    """
    # Implement LoRA tuning
    model.config.use_cache = False # to avoid error when setting "gradient_checkpointing=True"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",     
        bf16=True,
        warmup_steps=100,
        logging_steps=200,
        save_steps=200,
        save_total_limit=3,
        #max_steps=12,                    # only for testing purposes
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        label_names=["labels"],     
        group_by_length=True,
        report_to="wandb",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        #deepspeed='ds_config.json',
    )
    
    data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

    acc = evaluate.load('accuracy')
    f1 = evaluate.load('f1', average='macro')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        results = {}
        results.update(acc.compute(predictions=predictions, references=labels))
        results.update(f1.compute(predictions=predictions, references=labels, average="macro"))
        return results

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_data["train"],
        eval_dataset=train_data["test"], #.train_test_split(test_size=0.05)['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[],
    )

    """
    Implement training
    """
    # Implement training
    trainer.train()
    
    """
    Save the best model
    """
    # Save the weight at the last state
    save_dir = os.path.join(output_dir, save_name)
    trainer.save_model(save_dir)

if __name__=="__main__":
    main()