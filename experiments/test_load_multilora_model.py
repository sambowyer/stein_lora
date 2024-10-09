from stein_lora import MultiLoraConfig, MultiLoraModel
import peft
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch as t

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

# Load a basic gpt2 model
base_model1 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
base_model2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# Apply regular LoRA
lora_config = LoraConfig(r=4)
lora_model = get_peft_model(base_model1, lora_config)

# Apply Multi-LoRA
multi_lora_config = MultiLoraConfig(r=4, K=5)
multi_lora_model = get_peft_model(base_model2, multi_lora_config)



# save the lora model
lora_model.save_pretrained("temp/lora_model")

# load the lora model
lora_model2 = AutoModelForCausalLM.from_pretrained("temp/lora_model").to(device) 


# breakpoint()

multi_lora_model.peft_config['default'].peft_type = peft.PeftType.LORA

# save the multi-lora model
multi_lora_model.save_pretrained("temp/multi_lora_model")

# load the multi-lora model
multi_lora_model2 = AutoModelForCausalLM.from_pretrained("temp/multi_lora_model").to(device)
