from stein_lora import MultiLoraConfig, MultiLoraModel
from peft.utils.save_and_load import load_peft_weights
import peft

__all__ = ["save_multilora_weights", "apply_saved_multilora_weights"]

def save_multilora_weights(peft_model, save_directory: str, **kwargs):
    """
    Save the model to the specified directory.

    Args:
        peft_model ('PeftModel'): The model to save.
        save_directory (`str`): The directory to save the model to.
        kwargs: Additional keyword arguments passed to the `save_pretrained` method of the model.
    """
    # in order to use the save_pretrained method of the model, we need to temporarily
    # set self.peft_type to peft.PeftType.LORA
    # then change it back manually in the saved config file
    peft_model.peft_config['default'].peft_type = peft.PeftType.LORA

    peft_model.save_pretrained(save_directory, **kwargs)

    with open(f"{save_directory}/adapter_config.json", "r") as f:
        adapter_config = f.read()
    
    adapter_config = adapter_config.replace('"peft_type": "LORA"', f'"peft_type": "MultiLORA"')
    peft_model.peft_config['default'].peft_type = "MultiLORA"

    with open(f"{save_directory}/adapter_config.json", "w") as f:
        f.write(adapter_config)

def apply_saved_multilora_weights(base_model, save_directory, adapter_name="default"):
    """
    Applies the weights from a saved adapter to a model.

    Args:
        base_model ([`transformers.PreTrainedModel`]):
            The model to which the adapter should be applied.
        save_directory ([`str`]):
            The path to the folder containing saved adapter weights as 'adapter_model.safetensors'.

    Returns:
        MultiLoraModel: The model with the adapter applied.
    """

    # Load the adapter config and turn the base model into a MultiLoraModel
    saved_multilora_config = MultiLoraConfig.from_pretrained(save_directory)
    multi_lora_model = MultiLoraModel(base_model, saved_multilora_config, adapter_name=adapter_name)

    # Load the saved adapter weights and apply them to the model
    state_dict = load_peft_weights(save_directory, adapter_name=adapter_name)
    state_dict = {k: v for k, v in state_dict.items() if "lora" in k}

    for k, v in state_dict.items():
        lora_key = k.replace("base_model.", "").replace(".weight", ".default.weight")
        if lora_key in multi_lora_model.state_dict():
            multi_lora_model.state_dict()[lora_key].copy_(v)

    return multi_lora_model