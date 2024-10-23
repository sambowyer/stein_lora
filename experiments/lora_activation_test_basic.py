import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.decomposition import PCA
import os
import copy
import pickle

# import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("glue", "mrpc")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=8)


# Evaluate on test set
test_dataset = tokenized_datasets["test"]
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, output_hidden_states=True).to(device)

model.eval()
base_activations = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        base_output = model(input_ids, attention_mask=attention_mask)
        
        base_activations.append(base_output.hidden_states[-1].cpu())

base_activations = torch.cat(base_activations, dim=0)

for seed in range(100):
    print("Seed:", seed)
    for r in [2, 4, 8, 16, 32]:
        torch.manual_seed(seed)
        # Apply LoRA adapters
        lora_config = LoraConfig(r=r, lora_alpha=32, lora_dropout=0.1)
        model_with_lora = get_peft_model(copy.deepcopy(model), lora_config).to(device)

        # Optimizer
        optimizer = AdamW(model_with_lora.parameters(), lr=2e-5)

        # Training loop
        num_epochs = 1
        for epoch in range(num_epochs):
            model_with_lora.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model_with_lora(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            # Evaluation
            model_with_lora.eval()
            eval_loss = 0
            for batch in eval_dataloader:
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    outputs = model_with_lora(input_ids, attention_mask=attention_mask, labels=labels)
                    eval_loss += outputs.loss.item()
            
            eval_loss /= len(eval_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Evaluation Loss: {eval_loss}")


        # Get activations
        model_with_lora.eval()

        lora_activations = []

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                lora_output = model_with_lora(input_ids, attention_mask=attention_mask)
                
                lora_activations.append(lora_output.hidden_states[-1].cpu())

        lora_activations = torch.cat(lora_activations, dim=0)

        # Compute difference
        activation_difference = lora_activations - base_activations

        # Save tensor
        os.makedirs("lora_activations_rank", exist_ok=True)
        torch.save(activation_difference, f"lora_activations_rank/activation_difference_r{r}_seed{seed}.pt")

        # PCA
        activation_difference_np = activation_difference.numpy().reshape((-1, activation_difference.shape[-1]))
        pca = PCA()

        pca.fit(activation_difference_np)

        # save the pca fit with pickle
        with open(f"lora_activations_rank/pca_r{r}_seed{seed}.pkl", 'wb') as f:
            pickle.dump(pca, f)
            
        print(f"DONE: r={r}, seed={seed}")

    print()

print("DONE")