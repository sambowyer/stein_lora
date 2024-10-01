'''
EXPERIMENT: if we set gamma to 0 and sigma to 1e-18, is SVGD equivalent to Adam?
HYPOTHESIS: yes
TRUTH: no
    TL;DR: the SVGD driving force term averages the log-likelihood gradients over K particles,
           whereas Adam treats each particle's log-likelihood gradient separately.

    EXPLANATION: with gamma=0 we only have the driving force term in the SVGD update, which is 
                 an average of the loss (log-likelihood) gradients over our K particles.
                 We _expected_ that with sigma=1e-18, the kernel matrix would be the identity,
                 but actually at initialisation it's all 1s (because the LoRA weights are all initially 0), 
                 so the driving force term is just the average of the log-likelihood gradients,
                 NOT the log-likelihood gradients per-particle (divided by K to obtain the 'kernel-weighted mean').

                NOTE: this means that IF GAMMA=0, then THE INITIAL UPDATE IS IDENTICAL FOR EVERY PARTICLE
                      (I thought this was a bug at first but it's actually a feature)


    ... but if we wanted to make it equivalent to Adam, we could change this einsum in svgd driving force calculation:
        update_A = torch.einsum('ij,jkl->ikl', kernel_matrix, log_lik_grad_A + A_log_prior_grad_) / K
        update_B = torch.einsum('ij,jkl->ikl', kernel_matrix, log_lik_grad_B + B_log_prior_grad_) / K
                                    -----^-----
    and turn it into:
        update_A = torch.einsum('ij,jkl->jkl', kernel_matrix, log_lik_grad_A + A_log_prior_grad_) / K
        update_B = torch.einsum('ij,jkl->jkl', kernel_matrix, log_lik_grad_B + B_log_prior_grad_) / K
                                    -----^-----
    BUT then they're only the same for the first iteration (i.e. the first update step)
    
    This is because at initialisation all LoRA weights are delta_W = AB = 0,
    so the kernel matrix (with our tiny bandwidth) is all 1s. This, along with the corrected
    driving force term (above), means that the first update is
        K * (gradient of log likelihood) / K = gradient of log likelihood
    where the left K multiplication comes from a redundant sum over the einsum's i dimension,
    which gets cancelled out by the standard SVGD division by K.
    
    Then after the first update, suddenly the kernel matrix becomes the identity, so
    the driving force term is just 
        gradient of the log likelihood / K.

OTHER DETAILS:
    - I had to set all conceivable seeds at the start of each batch to get consistent results
      (even though the weights all start the same, a forward pass still contains randomness (I forgot this))
'''


import torch as t
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
import evaluate
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
import peft
from accelerate import Accelerator
import argparse
import time
import pickle
import cProfile
import copy
import os
import numpy as np

from stein_lora import MultiLoraConfig, MultiLoraModel, SVGD, MultiLoraLayer

startasctime = time.asctime()
print(f"Start at: {startasctime}\n")
start_time = time.time()

AUTO_BOOL = lambda x: x.lower() in ['true', '1', 't', 'y', 'yes']

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="bert-base-uncased")
argparser.add_argument("--dataset_path", type=str, default="glue")
argparser.add_argument("--dataset_name", type=str, default="mrpc")
argparser.add_argument("--truncate_train", type=int, default=-1)
argparser.add_argument("--truncate_val", type=int, default=-1)
# argparser.add_argument("--optimizer", type=str, default="adamw")
argparser.add_argument("--lr", type=float, default=1e-3)
argparser.add_argument("--lr_decay", type=AUTO_BOOL, default=True)
argparser.add_argument("--num_epochs", type=int, default=5)
argparser.add_argument("--batch_size", type=int, default=4)
argparser.add_argument("--r", type=int, default=4)
argparser.add_argument("--K", type=int, default=10)
argparser.add_argument("--gamma", type=float, default=0)    # set gamma to 0 and sigma to 1e-18
argparser.add_argument("--sigma", type=str, default=1e-18)  # so that SVGD is equivalent to Adam (we hope)
argparser.add_argument("--progress_bar", type=AUTO_BOOL, default=False)
argparser.add_argument("--save_results", type=AUTO_BOOL, default=False)
argparser.add_argument("--write_job_logs", type=AUTO_BOOL, default=False)
argparser.add_argument("--seed", type=int, default=42)
args = argparser.parse_args()

if args.sigma != "auto":
    args.sigma = float(args.sigma)

def write_log(log):
    if args.write_job_logs:
        with open(f"logs/{args.model.replace("/", "--")}_{args.dataset_name}_{args.optimizer}_r{args.r}_K{args.K}_gamma{args.gamma}_lr{args.lr}.log", "a") as f:
            f.write(log)

print(args)

t.manual_seed(args.seed)

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print(f"Device: {device}\n")

write_log(f"Start at: {startasctime}\n\n{args}\nDevice: {device}\n\n")

# accelerator = Accelerator()

raw_datasets = load_dataset(args.dataset_path, args.dataset_name)
tokenizer = AutoTokenizer.from_pretrained(args.model)
if "Llama" in args.model:
    tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Note: we're only set up for MRPC right now
assert args.dataset_name == "mrpc" and args.dataset_path == "glue"

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

if args.truncate_train > 0:
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(args.truncate_train))
if args.truncate_val > 0:
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(args.truncate_val))

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=args.batch_size, collate_fn=data_collator
)

# breakpoint()
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
if "Llama" in args.model:
    # model.config.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

K = args.K
r = args.r

# lora_config = LoraConfig(r=r,)
lora_config = MultiLoraConfig(r=r, K=K)#, init_lora_weights='pissa')
peft_model = get_peft_model(model, lora_config)

# breakpoint()

peft_model.print_trainable_parameters()

svgd_model = peft_model
adam_model = copy.deepcopy(peft_model)

svgd_optimizer = SVGD(svgd_model, lr=args.lr, sigma=args.sigma, gamma=args.gamma, base_optimizer=AdamW)
adam_optimizer = AdamW(adam_model.parameters(), lr=args.lr)

svgd_model.to(device)
adam_model.to(device)

# t.use_deterministic_algorithms(True)
def seed_everything(seed):
    # random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.backends.cudnn.deterministic = True

seed_everything(args.seed)

num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)

if args.lr_decay:
    svgd_lr_scheduler = get_scheduler(
        "linear",
        optimizer=svgd_optimizer.base_optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    adam_lr_scheduler = get_scheduler(
        "linear",
        optimizer=adam_optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )


def run_eval(model, eval_dataloader, metrics=None):
    acc_per_particle = t.zeros(K).to(device)
    ensemble_acc = t.zeros(1).to(device)

    disagreements = t.zeros(K).to(device)
    loss = t.zeros(1).to(device)

    kl_v_ens = t.zeros(K).to(device)
    kl_particles = t.zeros((K,K)).to(device)

    model.eval()
    for batch in eval_dataloader:
        batch = {k: t.cat(K*[v]).to(device) for k, v in batch.items()}

        with t.no_grad():
            outputs = model(**batch)

        loss += outputs.loss.item()

        logits = outputs.logits
        logits = logits.reshape(K, -1, *logits.shape[1:])

        logits_avg = t.logsumexp(logits, dim=0) #- t.log(t.tensor(K, dtype=t.float32))

        predictions = t.argmax(logits, dim=-1)
        predictions_avg = t.argmax(logits_avg, dim=-1)

        kl_v_ens     += (logits.softmax(-1) * (logits.log_softmax(-1) - logits_avg.log_softmax(-1)         )).sum((-1,-2))
        kl_particles += (logits.softmax(-1) * (logits.log_softmax(-1) - logits.unsqueeze(1).log_softmax(-1))).sum((-1,-2))

        disagreements += t.sum(predictions != predictions_avg, dim=-1)

        ensemble_acc += t.sum(predictions_avg == batch["labels"].reshape(K, -1)[0])
        acc_per_particle += t.sum(predictions == batch["labels"].reshape(K, -1), dim=-1)



        # breakpoint()

        # metrics_avg.add_batch(predictions=predictions_avg, references=batch["labels"].reshape(K, -1)[0])
        
        # for i, m in enumerate(metrics):
        #     m.add_batch(predictions=predictions[i], references=batch["labels"].reshape(K, -1)[i])

    # print(f"Total:  {metrics_avg.compute()}")
    # for i, m in enumerate(metrics):
    #     print(f"LORA_{i}: {m.compute()}")

    loss /= len(eval_dataloader.dataset) * 1.0
    acc_per_particle /= len(eval_dataloader.dataset) * 1.0
    ensemble_acc /= len(eval_dataloader.dataset) * 1.0
    kl_v_ens /= len(eval_dataloader.dataset) * 1.0
    kl_particles /= len(eval_dataloader.dataset) * 1.0

    print(f"Disagreements per particle: {disagreements} (total examples: {len(eval_dataloader.dataset)})")
    print(f"KL(particle || ensemble): {kl_v_ens}") 
    print(f"KL(particle || particle): {kl_particles}")
    print(f"Validation Loss: {loss}")
    print(f"Acc per particle: {acc_per_particle}")
    print(f"Ensemble acc: {ensemble_acc}")

    write_log(f"""Disagreements per particle: {disagreements} (total examples: {len(eval_dataloader.dataset)})
KL(particle || ensemble): {kl_v_ens}
KL(particle || particle): {kl_particles}
Validation Loss: {loss}
Acc per particle: {acc_per_particle}
Ensemble acc: {ensemble_acc}\n""")
        
    return loss, acc_per_particle.cpu(), ensemble_acc.cpu(), disagreements.cpu()

modelname2model = {'svgd': svgd_model, 'adam': adam_model}
modelname2lrscheduler = {'svgd': svgd_lr_scheduler, 'adam': adam_lr_scheduler}
modelname2optimizer = {'svgd': svgd_optimizer, 'adam': adam_optimizer}

stats = {model_name: {
    "loss": [],
    "epoch_times": [],
    "val_loss": [],
    "val_acc": [],
    "val_acc_per_particle": [],
    "val_disagreements": []
} for model_name in modelname2model.keys()}

def train(model_names=["svgd", "adam"]):
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for model_name in model_names:
            modelname2model[model_name].train()

        print(f"Epoch {epoch}")
        write_log(f"Epoch {epoch}\n")

        if args.progress_bar:
            progress_bar = tqdm(range(len(train_dataloader)))

        for batch in train_dataloader:
            batch = {k: t.cat(K*[v]).to(device) for k, v in batch.items()}

            for model_name in model_names:
                seed_everything(args.seed)

                model = modelname2model[model_name]
                optimizer = modelname2optimizer[model_name]
                lr_scheduler = modelname2lrscheduler[model_name]

                # breakpoint()
                outputs = model(**batch)
                loss = outputs.loss
                print(loss)
                            
                loss.backward()

                # optimizer.step()
                # optimizer.zero_grad()
                
                if args.lr_decay:
                    lr_scheduler.step()
                if args.progress_bar:
                    progress_bar.update(1/len(train_dataloader))

            ams = [(m.lora_A['default'].weight, m.lora_B['default'].weight) for m in adam_model.modules() if isinstance(m, MultiLoraLayer)]
            sms = [(m.lora_A['default'].weight, m.lora_B['default'].weight) for m in svgd_model.modules() if isinstance(m, MultiLoraLayer)]            

            ams_grad = [(m.lora_A['default'].weight.grad, m.lora_B['default'].weight.grad) for m in adam_model.modules() if isinstance(m, MultiLoraLayer)]
            sms_grad = [(m.lora_A['default'].weight.grad, m.lora_B['default'].weight.grad) for m in svgd_model.modules() if isinstance(m, MultiLoraLayer)]
            breakpoint()

            for _, opt in modelname2optimizer.items():
                opt.step()
                opt.zero_grad()

        if args.progress_bar:
            progress_bar.refresh()
            progress_bar.close()

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time}")
        write_log(f"Epoch time: {epoch_time}\n")

        for model_name in model_names:
            model = modelname2model[model_name]

            val_start_time = time.time()
            val_loss, acc_per_particle, ensemble_acc, disagreements = run_eval(model, eval_dataloader)#, metrics)
            val_end_time = time.time()
            print(f"{model_name.upper()} Validation time: {val_end_time - val_start_time}")
            write_log(f"{model_name.upper()} Validation time: {val_end_time - val_start_time}\n")

            stats[model_name]["loss"].append(loss.item())
            stats[model_name]["val_loss"].append(val_loss)
            stats[model_name]["val_acc"].append(ensemble_acc)
            stats[model_name]["val_acc_per_particle"].append(acc_per_particle)
            stats[model_name]["val_disagreements"].append(disagreements)

        # breakpoint()

        print()

print("SVGD EVAL")
run_eval(svgd_model, eval_dataloader)

print("ADAM EVAL")
run_eval(adam_model, eval_dataloader)

# cProfile.run("train()", "profiles/train_stats.prof")
train()

print(f"Total time: {time.time() - start_time}")
write_log(f"\nTotal time: {time.time() - start_time}\n")

if args.save_results:
    with open(f"results/{args.optimizer}_r{r}_K{K}_gamma{gamma}_sigma{sigma}.pkl", "wb") as f:
        pickle.dump(stats, f)

if device.type == 'cuda': 
    cuda_mem_summary = f"CUDA memory - Card size: {t.cuda.get_device_properties(device).total_memory/(1024**3):.2f}GB, Max allocated: {t.cuda.max_memory_allocated(device)/(1024**3):.2f}GB, Max reserved: {t.cuda.max_memory_reserved(device)/(1024**3):.2f}GB"
    print(cuda_mem_summary)
    write_log(cuda_mem_summary)

endtime = time.asctime()
print(f"Finished at: {endtime}")
write_log(f"\nFinished at: {endtime}\n")