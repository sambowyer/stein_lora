"""
Stein VI-trained LoRA
"""
__version__ = "0.0.1"

from .multi_lora import *
from .svgd import *
from .sadamw import SAdamW #*

# import peft
# peft_types.PeftType.MultiLora = "MultiLORA"
# peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING['MultiLORA'] = MultiLoraModel