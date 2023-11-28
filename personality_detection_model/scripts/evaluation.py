import os 
import json
from tqdm import tqdm
import argparse

import torch

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import (
    PeftModel,
    PeftConfig,
)

SEED = 42


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed_data_50tweets")
    ap.add_argument("--checkpoint_dir", type=str, default="")
    ap.add_argument("--dimension_id", type=int, default=None)
    args = ap.parse_args()
    return (
        args.data_dir,
        args.checkpoint_dir,
        args.dimension_id,
    )


def load_data(data_dir: str, dimension_id: int):
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
    
    return dataset, label2id, id2label
    

def predict(test_data, tokenizer, model, save_dir):
    pred_list = []
    
    for dp in tqdm(test_data):
        inputs = tokenizer(dp['text'], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1).item()
            predictions = model.config.id2label[predictions]

        pred_list.append(predictions)
        
    return pred_list


def save_results(references, predictions, save_dir):
    # Save results
    results = {
        'references': references,
        'predictions': pred_list
    }
    
    with open(save_dir, "w") as f:
        json.dump(results, f)

        
def print_accuracy(refs, preds):
    cor = 0
    for ref, pred in tqdm(zip(refs, preds)):
        if ref == pred:
            cor += 1

    accuracy = round(cor / len(refs), 4)

    print(f"Accuracy: {accuracy}")



def main():
    (
        data_dir,
        checkpoint_dir,
        dimension_id,
    ) = parse_args()
    
    # Load dataset
    dataset, label2id, id2label = load_data(data_dir, dimension_id)
    
    # Directory of LoRA model
    # Load the configuration for LoRA model
    model_name = os.path.join(checkpoint_dir, "final_model")
    
    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        device_map="auto",
        use_flash_attention_2=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, model_name)

    # Merge the adapter weights with the base model which allows the model's inference to speed up
    merged_model = model.merge_and_unload()
    
    # Predict resutls
    save_dir = os.path.join(checkpoint_dir, "test_results.json")
    test_data = dataset['test']
    preds = predict(test_data, tokenizer, merged_model, save_dir)
        
    # Print accuracy
    refs = test_data['label']
    print_accuracy(refs, preds)
    
    # Save results into json file
    save_results(refs, preds, save_dir)
    
if __name__=="__main__":
    main()
    
