import torch as t
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler, AutoModelForCausalLM
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
from peft import PeftType
from accelerate import Accelerator
import argparse
import time
import pickle
import cProfile
import os

from stein_lora import MultiLoraConfig, MultiLoraModel, SVGD
from utils import get_dataloader

startasctime = time.asctime()
start_time = time.time()

AUTO_BOOL = lambda x: x.lower() in ['true', '1', 't', 'y', 'yes']

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="bert-base-uncased")
argparser.add_argument("--dataset_path", type=str, default="glue")
argparser.add_argument("--dataset_name", type=str, default="mrpc")
argparser.add_argument("--truncate_train", type=int, default=-1)
argparser.add_argument("--truncate_test", type=int, default=-1)
argparser.add_argument("--optimizer", type=str, default="adamw")
argparser.add_argument("--lr", type=float, default=1e-3)
argparser.add_argument("--lr_decay", type=AUTO_BOOL, default=True)
argparser.add_argument("--num_epochs", type=int, default=10)
argparser.add_argument("--batch_size", type=int, default=4)
argparser.add_argument("--train_batch_size", type=int, default=-1)
argparser.add_argument("--test_batch_size", type=int, default=-1)
argparser.add_argument("--r", type=int, default=4)
argparser.add_argument("--K", type=int, default=10)
argparser.add_argument("--gamma", type=float, default=1e-2)
argparser.add_argument("--sigma", type=str, default="auto")
argparser.add_argument("--damping_lambda", type=float, default=1)
argparser.add_argument("--accelerate", type=AUTO_BOOL, default=False)
argparser.add_argument("--grad_checkpointing", type=AUTO_BOOL, default=False)
argparser.add_argument("--progress_bar", type=AUTO_BOOL, default=False)
argparser.add_argument("--save_results", type=AUTO_BOOL, default=True)
argparser.add_argument("--save_model_every", type=int, default=0)
argparser.add_argument("--write_log", type=AUTO_BOOL, default=False)
argparser.add_argument("--seed", type=int, default=42)
args = argparser.parse_args()

if args.train_batch_size == -1:
    args.train_batch_size = args.batch_size
if args.test_batch_size == -1:
    args.test_batch_size = args.batch_size

if args.sigma != "auto":
    args.sigma = float(args.sigma)

if args.model == "bert-base-uncased":
    short_model_name = "bert"
elif 'Llama' in args.model:
    short_model_name = "llama"

config_str = f"{short_model_name}_{args.dataset_name}_{args.optimizer}_r{args.r}_K{args.K}_gamma{args.gamma}_lr{args.lr}_seed{args.seed}"

logs_dir    = f"logs/{args.dataset_name}"
results_dir = f"results/{args.dataset_name}"
models_dir  = f"models/{args.dataset_name}"

for d in [logs_dir, results_dir, models_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

def write_log(log, print_log=True):
    if print_log:
        print(log)
    if args.write_log:
        with open(f"{logs_dir}/{config_str}.log", "a") as f:
            f.write(f"{log}\n")

print(args)

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

t.manual_seed(args.seed)
if device.type == 'cuda': 
    t.cuda.manual_seed(args.seed)

write_log(f"Start at: {startasctime}\n\n{args}\nDevice: {device}\n")

if args.accelerate:
    accelerator = Accelerator()


# Special case for MRPC
if args.dataset_name == "mrpc" and args.dataset_path == "glue":
    raw_datasets = load_dataset(args.dataset_path, args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "Llama" in args.model:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(example):

        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if args.truncate_train > 0:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(args.truncate_train))
    if args.truncate_test > 0:
        # tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(args.truncate_test))
        tokenized_datasets["test"] = tokenized_datasets["test"].select(range(args.truncate_test))

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=args.train_batch_size, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        # tokenized_datasets["validation"], batch_size=args.test_batch_size, collate_fn=data_collator
        tokenized_datasets["test"], batch_size=args.test_batch_size, collate_fn=data_collator
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

else:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_cache=True)#,  torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if "Llama" in args.model:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, use_cache=True)#,  torch_dtype=torch.float16, low_cpu_mem_usage=True)

    train_dataloader, test_dataloader, class_ids, N = get_dataloader(tokenizer, args.dataset_name, args.train_batch_size, args.test_batch_size)
    
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(class_ids))


if "Llama" in args.model:
    model.config.pad_token_id = model.config.eos_token_id

if args.grad_checkpointing:
    model.gradient_checkpointing_enable()

K = args.K
r = args.r

# lora_config = LoraConfig(r=r,)
lora_config = MultiLoraConfig(r=r, K=K)#, init_lora_weights='pissa')
peft_model = get_peft_model(model, lora_config)


peft_model.print_trainable_parameters()

if args.optimizer == "adamw":
    optimizer = AdamW(peft_model.parameters(), lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = SGD(peft_model.parameters(), lr=args.lr)
elif args.optimizer == "svgd":
    optimizer = SVGD(peft_model, lr=args.lr, sigma=args.sigma, gamma=args.gamma, damping_lambda=args.damping_lambda, base_optimizer=AdamW)
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

peft_model.to(device)

num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)

if args.lr_decay:
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer.base_optimizer if args.optimizer == "svgd" else optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

if args.accelerate:
    # peft_model, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(peft_model, train_dataloader, test_dataloader, lr_scheduler)

    if isinstance(optimizer, SVGD):
        peft_model, train_dataloader, test_dataloader, base_optimizer, lr_scheduler = accelerator.prepare(peft_model, train_dataloader, test_dataloader, optimizer.base_optimizer, lr_scheduler)
        optimizer.base_optimizer = base_optimizer
    else:
        peft_model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(peft_model, train_dataloader, test_dataloader, optimizer, lr_scheduler)


num_batches = len(train_dataloader)
stats = {
    "train_loss": t.zeros((num_epochs, num_batches)),
    "train_acc": t.zeros((num_epochs, num_batches)),
    "train_acc_per_particle": t.zeros((num_epochs, num_batches, K)),
    "train_disagreements": t.zeros((num_epochs, num_batches, K)),
    "train_kl_v_ens": t.zeros((num_epochs, num_batches, K)),
    "train_kl_particles": t.zeros((num_epochs, num_batches, K, K)),

    "epoch_times": t.zeros(num_epochs),

    "val_loss": t.zeros(num_epochs+1),
    "val_acc": t.zeros(num_epochs+1),
    "val_acc_per_particle": t.zeros((num_epochs+1, K)),
    "val_disagreements": t.zeros((num_epochs+1, K)),
    "val_kl_v_ens": t.zeros((num_epochs+1, K)),
    "val_kl_particles": t.zeros((num_epochs+1, K, K)),
}

@t.no_grad()
def update_stats(loss, logits, labels, train=True, epoch=None, batch=None):
    '''
    logits: (K*batch_size, num_classes)
    labels: (K*batch_size,) (K copies of the same labels)
    '''
    
    logits = logits.reshape(K, -1, *logits.shape[1:]).cpu()
    labels = labels.reshape(K, -1).cpu()

    # take average of logits over particles
    logits_avg = t.logsumexp(logits, dim=0) #- t.log(t.tensor(K, dtype=t.float32))

    predictions = t.argmax(logits, dim=-1)          # per-particle predictions
    predictions_avg = t.argmax(logits_avg, dim=-1)  # particle-averaged predictions

    # KL divergence between particles and ensemble
    kl_v_ens     = (logits.softmax(-1) * (logits.log_softmax(-1) - logits_avg.log_softmax(-1)         )).sum((-1,-2))
    
    # KL divergence between particles
    kl_particles = (logits.softmax(-1) * (logits.log_softmax(-1) - logits.unsqueeze(1).log_softmax(-1))).sum((-1,-2))

    # disagreements between particles and ensemble
    disagreements = t.sum(predictions != predictions_avg, dim=-1)

    # accuracy per particle
    acc_per_particle = t.sum(predictions == labels, dim=-1)
    
    # ensemble accuracy
    ensemble_acc = t.sum(predictions_avg == labels[0])

    
    prepend = "train" if train else "val"
    tensor_index = (epoch, batch) if train else epoch
    
    stats[f"{prepend}_loss"][tensor_index] += loss
    stats[f"{prepend}_acc"][tensor_index] += ensemble_acc
    stats[f"{prepend}_acc_per_particle"][tensor_index] += acc_per_particle
    stats[f"{prepend}_disagreements"][tensor_index] += disagreements
    stats[f"{prepend}_kl_v_ens"][tensor_index] += kl_v_ens
    stats[f"{prepend}_kl_particles"][tensor_index] += kl_particles

def get_stats_str(train, epoch):
    prepend = "train" if train else "val"

    loss = stats[f"{prepend}_loss"][epoch]
    acc_per_particle = stats[f"{prepend}_acc_per_particle"][epoch]
    ensemble_acc = stats[f"{prepend}_acc"][epoch]
    disagreements = stats[f"{prepend}_disagreements"][epoch]
    kl_v_ens = stats[f"{prepend}_kl_v_ens"][epoch]
    kl_particles = stats[f"{prepend}_kl_particles"][epoch]

    # take mean over first dimension if train
    if train:
        loss = loss.mean(0)
        acc_per_particle = acc_per_particle.mean(0)
        ensemble_acc = ensemble_acc.mean(0)
        disagreements = disagreements.mean(0)
        kl_v_ens = kl_v_ens.mean(0)
        kl_particles = kl_particles.mean(0)

    return f"""Disagreements per particle: {disagreements} (total examples: {len(train_dataloader.dataset) if train else len(test_dataloader.dataset)})
KL(particle || ensemble): {kl_v_ens}
KL(particle || particle): \n{kl_particles}
{"Train" if train else "Test"} Loss: {loss}
Acc per particle: {acc_per_particle}
Ensemble acc: {ensemble_acc}\n"""


def train(model, train_dataloader, epoch=0):
    epoch_start_time = time.time()
    model.train()

    write_log(f"Epoch {epoch}")

    if args.progress_bar:
        progress_bar = tqdm(range(len(train_dataloader)))

    batch_count = -1
    
    for batch in train_dataloader:
        batch_count += 1

        if args.dataset_name == "mrpc":
            batch = {k: t.cat(K*[v]).to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]

            labels = batch["labels"]

            outputs = model(**batch)
            loss = outputs.loss

            logits = outputs.logits 
        else:
            input_ids = t.cat(K*[batch["input_ids"]]).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id)

            labels = t.cat(K*[batch["answer_idx"]]).to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            logits = t.log_softmax(logits[..., -1, class_ids], -1)

            loss = -Categorical(logits=logits).log_prob(labels).mean() 

        update_stats(loss.item() * (input_ids.shape[0] / K), logits, labels, train=True, epoch=epoch, batch=batch_count)

        if args.accelerate:
            accelerator.backward(loss)
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        if args.lr_decay:
            lr_scheduler.step()
        if args.progress_bar:
            progress_bar.update(1)

    if args.progress_bar:
        progress_bar.refresh()
        progress_bar.close()

    for key, val in stats.items():
        if "train" in key:
            stats[key] /= len(train_dataloader) * 1.0 # not len(train_dataloader.dataset) * 1.0 because we want to average per-batch

    stats_str = get_stats_str(train=True, epoch=epoch)
    write_log(stats_str)

    epoch_time = time.time() - epoch_start_time
    write_log(f"Epoch time: {epoch_time}")
    stats["epoch_times"][epoch] = epoch_time

def test(model, test_dataloader, epoch=0):
    val_start_time = time.time()    
    model.eval()
    
    for batch in test_dataloader:
        # breakpoint()
        if args.dataset_name == "mrpc":
            batch = {k: t.cat(K*[v]).to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]

            labels = batch["labels"]

            with t.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            logits = outputs.logits

        else:
            input_ids = t.cat(K*[batch["input_ids"]]).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id)

            labels = t.cat(K*[batch["answer_idx"]]).to(device)

            with t.no_grad():
                logits = model(input_ids, attention_mask=attention_mask).logits
            logits = t.log_softmax(logits[..., -1, class_ids], -1)

            loss = -Categorical(logits=logits).log_prob(labels).mean()

        
        update_stats(loss.item() * (input_ids.shape[0] / K), logits, labels, train=False, epoch=epoch)

    for key, val in stats.items():
        if "val" in key:
            stats[key] /= len(test_dataloader.dataset) * 1.0

    stats_str = get_stats_str(train=False, epoch=epoch)
    write_log(stats_str)
    
    val_end_time = time.time()
    write_log(f"Test time: {val_end_time - val_start_time}")


def run():
    write_log("################################")
    write_log("Initial test run")
    test(peft_model, test_dataloader, epoch=0)

    for epoch in range(num_epochs):
        write_log("\n################################")
        train(peft_model, train_dataloader, epoch=epoch)

        test(peft_model, test_dataloader, epoch=epoch+1)

        if args.save_model_every > 0 and (epoch+1) % args.save_model_every == 0:
            peft_model.save_pretrained(f"{models_dir}/{config_str}_epoch{epoch}")

        # breakpoint()

run()

write_log(f"\nTotal time: {time.time() - start_time}")

if args.save_results:
    with open(f"{results_dir}/{config_str}.pkl", "wb") as f:
        pickle.dump(stats, f)

if device.type == 'cuda': 
    cuda_mem_summary = f"CUDA memory - Card size: {t.cuda.get_device_properties(device).total_memory/(1024**3):.2f}GB, Max allocated: {t.cuda.max_memory_allocated(device)/(1024**3):.2f}GB, Max reserved: {t.cuda.max_memory_reserved(device)/(1024**3):.2f}GB"
    write_log(cuda_mem_summary)

endasctime = time.asctime()
write_log(f"\nFinished at: {endasctime}")