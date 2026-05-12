# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mem_linoss import MemLinOSSConfig
from .modeling_mem_linoss import MemLinOSSForCausalLM, MemLinOSSModel

AutoConfig.register(MemLinOSSConfig.model_type, MemLinOSSConfig, exist_ok=True)
AutoModel.register(MemLinOSSConfig, MemLinOSSModel, exist_ok=True)
AutoModelForCausalLM.register(MemLinOSSConfig, MemLinOSSForCausalLM, exist_ok=True)

__all__ = ['MemLinOSSConfig', 'MemLinOSSForCausalLM', 'MemLinOSSModel']
