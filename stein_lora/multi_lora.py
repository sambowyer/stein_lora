import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Any, Optional, Union, List
import re 
import math
from itertools import chain

from peft.utils import get_quantization_config
from peft.tuners.tuners_utils import BaseTunerLayer 
from peft.tuners.lora import LoraModel, LoraLayer, LoraConfig

__all__ = ["MultiLoraConfig", "MultiLoraModel", "MultiLoraLayer", "Linear", "nn_ParallelLinear"]


@dataclass
class MultiLoraConfig(LoraConfig):
    """
    An extension to the configuration class for LoRA models, used for configuration of [`MultiLoraModel`] class.
    This class adds a parameter `K` which represents the number of LoRA copies/`particles` to train in parallel.
    
    Args:
        K (`int`):
            The number of LoRA copies/`particles` to train in parallel.
        **kwargs:
            Additional keyword arguments passed along to the base [`LoraConfig`] class.
    """
    K: int = field(default=3, metadata={"help": "The number of LoRA copies/`particles` to train in parallel."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "MultiLORA"

        from peft import peft_model, mapping
        peft_model.PEFT_TYPE_TO_MODEL_MAPPING[self.peft_type] = MultiLoraModel
        mapping.PEFT_TYPE_TO_CONFIG_MAPPING[self.peft_type] = MultiLoraConfig


class MultiLoraModel(LoraModel):
    """
    Create a Multi-LoRA model by stacking multiple copies of each LoRA adapter in parallel.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`MultiLoraConfig`]): The configuration of the Multi-LoRA model.
        adapter_name (`str`): the name of the adapted, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Multi-LoRA model.

    Example:
        ```
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([]`MultiLoraConfig`]): The configuration of the Multi-LoRA model.
    """

    # prefix: str = "lora_"  # this is already inherited from LoraModel (we don't rename lora params from lora_A and lora_B to multi_lora_A and multi_lora_B)

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        K = lora_config.rank_pattern.get(target_name_key, lora_config.K)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "K": K,
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        # from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, MultiLoraLayer): # and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                K,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # TODO: implement multi-lora with different backends by creating different dispatchers. 
        # E.g. regular LoRA has a dispatcher for BnB, AQLM, GPTQ, etc. -- see peft.lora.model._create_new_module

        new_module = dispatch_multi_lora(target, adapter_name, lora_config=lora_config, **kwargs)

        if new_module is None:
            # no module could be matched (our of all (one) dispatchers tried)
            
            raise ValueError(
                f"Target module {target} is not supported.`."
            )

        return new_module
    

class MultiLoraLayer(LoraLayer):
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer, **kwargs)
        
        self.K = {}

    def update_layer(
        self, adapter_name, K, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types

        if K <= 0:
            raise ValueError(f"`K` should be a positive integer value but the value passed is {K}")
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.K[adapter_name] = K
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn_ParallelLinear(self.in_features, r, K=K, bias=False)
        self.lora_B[adapter_name] = nn_ParallelLinear(r, self.out_features, K=K, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            self.pissa_init(adapter_name, init_lora_weights)
        elif init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            # nn.init.normal_(self.lora_B[adapter_name].weight, std=1 / self.r[adapter_name])
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            # nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

class Linear(nn.Module, MultiLoraLayer):
    # Multi-Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        K: int = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MultiLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name,
            K,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights
        """
        pass

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        pass

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "multi-lora." + rep


class nn_ParallelLinear(nn.Module):
    __constants__ = ['in_features', 'out_features', 'K']
    in_features: int
    out_features: int
    K: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, K: int = 1, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = nn.Parameter(torch.empty((K, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(K, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # assume at start we have:
        # input.shape == [K * batch, seq_length, in_features]
        # weight.shape == [K, in_features, out_features]
        #
        # before matmul, we want to change input s.t.:
        # input.shape == [K, batch, seq_length, in_features]
        # 
        # then matmul of input @ weight will give us:
        # output.shape == [K, batch, seq_length, out_features]
        # 
        # but before we return, we want to reshape the output to:
        # output.shape == [batch * K, seq_length, out_features]

        batch_size = input.shape[0] // self.K
        seq_length = input.shape[1]

        out = input.reshape(self.K, batch_size, *input.shape[1:]) @ self.weight[:, None, ...].transpose(-1,-2)

        return out.reshape(self.K * batch_size, seq_length, self.out_features)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, K={}, bias={}'.format(
            self.in_features, self.out_features, self.K, self.bias is not None
        )

def dispatch_multi_lora(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # if isinstance(target_base_layer, torch.nn.Embedding):
    #     embedding_kwargs = kwargs.copy()
    #     embedding_kwargs.pop("fan_in_fan_out", None)
    #     embedding_kwargs.update(lora_config.loftq_config)
    #     new_module = Embedding(target, adapter_name, **embedding_kwargs)
    # elif isinstance(target_base_layer, torch.nn.Conv2d):
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Conv2d(target, adapter_name, **kwargs)
    # elif isinstance(target_base_layer, torch.nn.Linear):
    #     if kwargs["fan_in_fan_out"]:
    #         warnings.warn(
    #             "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
    #             "Setting fan_in_fan_out to False."
    #         )
    #         kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Linear(target, adapter_name, **kwargs)
    # elif isinstance(target_base_layer, Conv1D):
    #     if not kwargs["fan_in_fan_out"]:
    #         warnings.warn(
    #             "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
    #         )
    #         kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    new_module = Linear(target, adapter_name, **kwargs)

    return new_module



if __name__ == "__main__":
    model = torch.nn.Transformer()
    config = MultiLoraConfig(r=4, K=3)
    print(config)
    breakpoint()
    multi_lora = MultiLora(config, model)
    print(multi_lora)
