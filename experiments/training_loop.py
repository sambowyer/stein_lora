import torch as t
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
import evaluate
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
import peft

from stein_lora import MultiLoraConfig, MultiLoraModel

# peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING['MultiLORA'] = MultiLoraModel

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print(f"Device: {device}")


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

K = 5
r = 4

# lora_config = LoraConfig(r=r,)
lora_config = MultiLoraConfig(r=r, K=K)
peft_model = get_peft_model(model, lora_config)

# breakpoint()

peft_model.print_trainable_parameters()



optimizer = AdamW(peft_model.parameters(), lr=1e-3)

peft_model.to(device)

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)



breakpoint() 

metrics = [evaluate.load("glue", "mrpc") for _ in range(K)]

def run_eval(model, eval_dataloader, metrics):

    for batch in eval_dataloader:
        batch = {k: t.cat(K*[v]).to(device) for k, v in batch.items()}

        with t.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        logits = logits.reshape(K, -1, *logits.shape[1:])
        # breakpoint()
        predictions = t.argmax(logits, dim=-1)

        for i, m in enumerate(metrics):
            m.add_batch(predictions=predictions[i], references=batch["labels"].reshape(K, -1)[i])

    for i, m in enumerate(metrics):
        print(f"LORA_{i}: {m.compute()}")


for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    progress_bar = tqdm(range(len(train_dataloader)))

    for batch in train_dataloader:
        batch = {k: t.cat(K*[v]).to(device) for k, v in batch.items()}

        # breakpoint()
        outputs = peft_model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    progress_bar.refresh()
    progress_bar.close()

    run_eval(peft_model, eval_dataloader, metrics)