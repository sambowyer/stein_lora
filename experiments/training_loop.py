import torch as t
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
import evaluate
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
import peft
from accelerate import Accelerator

from stein_lora import MultiLoraConfig, MultiLoraModel, RBF_kernel, SVGD

# peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING['MultiLORA'] = MultiLoraModel

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print(f"Device: {device}")

accelerator = Accelerator()

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

truncate_train = 500
truncate_val   = 200



train_dataloader = DataLoader(
    tokenized_datasets["train"].select(range(truncate_train)), shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"].select(range(truncate_val)), batch_size=8, collate_fn=data_collator
)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

K = 3
r = 4

# lora_config = LoraConfig(r=r,)
lora_config = MultiLoraConfig(r=r, K=K)
peft_model = get_peft_model(model, lora_config)

# breakpoint()

peft_model.print_trainable_parameters()

# loraA = []
# loraB = []
# for name, param in peft_model.named_parameters():
#     if 'lora_A' in name:
#         loraA.append(param)
#     elif 'lora_B' in name:
#         loraB.append(param)

# optimizer = AdamW(peft_model.parameters(), lr=1e-3)

optimizer = SVGD(peft_model, lr=1e1, kernel= RBF_kernel(sigma=1e-2), gamma=3e1)

peft_model.to(device)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


peft_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(peft_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

# breakpoint() 

metrics = [evaluate.load("glue", "mrpc") for _ in range(K)]
metrics_avg = evaluate.load("glue", "mrpc")

def run_eval(model, eval_dataloader, metrics):

    for batch in eval_dataloader:
        batch = {k: t.cat(K*[v]).to(device) for k, v in batch.items()}

        with t.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        logits = logits.reshape(K, -1, *logits.shape[1:])

        logits_avg = t.logsumexp(logits, dim=0) #- t.log(t.tensor(K, dtype=t.float32))

        predictions = t.argmax(logits, dim=-1)
        predictions_avg = t.argmax(logits_avg, dim=-1)

        metrics_avg.add_batch(predictions=predictions_avg, references=batch["labels"].reshape(K, -1)[0])
        
        for i, m in enumerate(metrics):
            m.add_batch(predictions=predictions[i], references=batch["labels"].reshape(K, -1)[i])

    print(f"Total:  {metrics_avg.compute()}")
    for i, m in enumerate(metrics):
        print(f"LORA_{i}: {m.compute()}")



run_eval(peft_model, eval_dataloader, metrics)

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
        optimizer.zero_grad()
        
        lr_scheduler.step()
        progress_bar.update(1)

    progress_bar.refresh()
    progress_bar.close()

    run_eval(peft_model, eval_dataloader, metrics)