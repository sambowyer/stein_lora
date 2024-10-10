from stein_lora import MultiLoraConfig, MultiLoraModel, save_multilora_weights, apply_saved_multilora_weights
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

# check if the two models are the same
assert lora_model.config == lora_model2.config
# assert lora_model.state_dict() == lora_model2.state_dict()

assert all(t.allclose(x,y) for x,y in zip([lora_model.state_dict()[x] for x in lora_model.state_dict()],
                                 [lora_model2.state_dict()[x] for x in lora_model2.state_dict()]))

# save the multi-lora model
save_multilora_weights(multi_lora_model, "temp/multi_lora_model")
# multi_lora_model.save_pretrained("temp/multi_lora_model")

# load the multi-lora model
# multi_lora_model2 = AutoModelForCausalLM.from_pretrained("temp/multi_lora_model").to(device)
base_model_temp = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
multi_lora_model2 = apply_saved_multilora_weights(base_model_temp, "temp/multi_lora_model")


# compare the two models
assert multi_lora_model.config == multi_lora_model2.config
# assert multi_lora_model.state_dict() == multi_lora_model2.state_dict()

assert all(t.allclose(x,y) for x,y in zip([multi_lora_model.state_dict()[x]  for x in multi_lora_model.state_dict()],
                                          [multi_lora_model2.state_dict()[x] for x in multi_lora_model2.state_dict()]))