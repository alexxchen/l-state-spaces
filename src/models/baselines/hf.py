import importlib
from contextlib import nullcontext

import torch

from src.models.sequence.base import SequenceModule


class HFModelAdapter(SequenceModule):
    """Wrap a Hugging Face model in the repository's sequence-model interface.

    The adapter accepts either token ids ``(B, L)`` or input embeddings
    ``(B, L, D)`` and returns ``(output, state)`` so it can be used by the
    existing training loop without changing the caller contract.
    """

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        auto_class="AutoModel",
        imports=None,
        init_strategy="from_pretrained",
        config_name=None,
        config_class=None,
        output="auto",
        torch_dtype=None,
        autocast_dtype=None,
        output_dtype=None,
        pad_token_id=None,
        use_cache=False,
        freeze=False,
        gradient_checkpointing=False,
        trust_remote_code=False,
        config_kwargs=None,
        model_kwargs=None,
        **kwargs,
    ):
        super().__init__()

        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "HFModelAdapter requires the `transformers` package to be installed."
            ) from exc

        if config_kwargs is None:
            config_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if imports is None:
            imports = []

        for module_name in imports:
            importlib.import_module(module_name)

        auto_model_cls = getattr(transformers, auto_class, None)
        if auto_model_cls is None:
            raise ValueError(
                f"Unknown transformers auto class `{auto_class}`."
            )

        self.auto_class = auto_class
        self.init_strategy = init_strategy
        self.output = self._resolve_output_mode(output, auto_class)
        self.model_dtype = self._resolve_torch_dtype(torch_dtype)
        self.autocast_dtype = self._resolve_torch_dtype(autocast_dtype)
        self.output_dtype = self._resolve_torch_dtype(output_dtype)
        self.use_cache = use_cache
        self.model = self._build_model(
            transformers=transformers,
            auto_model_cls=auto_model_cls,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config_name=config_name,
            config_class=config_class,
            config_kwargs=config_kwargs,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.model_dtype,
        )

        if self.model_dtype is not None:
            self.model = self.model.to(dtype=self.model_dtype)

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.config = self.model.config
        self.pad_token_id = (
            self.config.pad_token_id if pad_token_id is None else pad_token_id
        )
        self.d_model = self._infer_hidden_size(self.config)
        self.d_output = self._infer_output_size(self.config, self.output)

        # Keep a few top-level flags configurable through Hydra without
        # needing to thread them into model_kwargs.
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def forward(self, x, attention_mask=None, state=None, *args, **kwargs):
        inputs = self._prepare_inputs(x, attention_mask)

        model_kwargs = {
            **inputs,
            "return_dict": True,
        }
        if self.use_cache:
            model_kwargs["use_cache"] = True
        if state is not None and self.use_cache:
            model_kwargs["past_key_values"] = state
        if self.output == "hidden_states_last":
            model_kwargs["output_hidden_states"] = True

        with self._autocast_context(model_kwargs):
            outputs = self.model(**model_kwargs)
            y = self._extract_output(outputs)

        if (
            self.output_dtype is not None
            and torch.is_tensor(y)
            and torch.is_floating_point(y)
        ):
            y = y.to(dtype=self.output_dtype)

        next_state = getattr(outputs, "past_key_values", None) if self.use_cache else None
        return y, next_state

    def step(self, x, state=None, *args, **kwargs):
        raise NotImplementedError(
            "HFModelAdapter does not implement recurrent stepping. "
            "Disable step benchmarking for Hugging Face backbones."
        )

    @staticmethod
    def _resolve_output_mode(output, auto_class):
        if output != "auto":
            return output

        if auto_class == "AutoModel":
            return "last_hidden_state"
        if "Classification" in auto_class:
            return "logits"
        if "LM" in auto_class:
            return "logits"
        return "last_hidden_state"

    def _build_model(
        self,
        transformers,
        auto_model_cls,
        pretrained_model_name_or_path,
        config_name,
        config_class,
        config_kwargs,
        model_kwargs,
        trust_remote_code,
        torch_dtype,
    ):
        if self.init_strategy == "from_pretrained":
            if pretrained_model_name_or_path is None:
                raise ValueError(
                    "`pretrained_model_name_or_path` must be set when "
                    "`init_strategy=from_pretrained`."
                )
            return auto_model_cls.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )

        if self.init_strategy == "from_config":
            config = self._build_config(
                transformers=transformers,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config_name=config_name,
                config_class=config_class,
                config_kwargs=config_kwargs,
                trust_remote_code=trust_remote_code,
            )
            return auto_model_cls.from_config(config)

        raise ValueError(
            f"Unsupported HF adapter init strategy `{self.init_strategy}`."
        )

    @staticmethod
    def _build_config(
        transformers,
        pretrained_model_name_or_path,
        config_name,
        config_class,
        config_kwargs,
        trust_remote_code,
    ):
        if config_class is not None:
            config_cls = HFModelAdapter._resolve_class(transformers, config_class)
            return config_cls(**config_kwargs)

        config_source = config_name or pretrained_model_name_or_path
        if config_source is None:
            raise ValueError(
                "Set `config_class` or `config_name` when `init_strategy=from_config`."
            )

        return transformers.AutoConfig.from_pretrained(
            config_source,
            trust_remote_code=trust_remote_code,
            **config_kwargs,
        )

    @staticmethod
    def _resolve_class(transformers, class_path):
        config_cls = getattr(transformers, class_path, None)
        if config_cls is not None:
            return config_cls

        if "." not in class_path:
            raise ValueError(f"Unknown config class `{class_path}`.")

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        config_cls = getattr(module, class_name, None)
        if config_cls is None:
            raise ValueError(f"Unknown config class `{class_path}`.")
        return config_cls

    @staticmethod
    def _resolve_torch_dtype(dtype):
        if dtype is None:
            return None

        if isinstance(dtype, torch.dtype):
            return dtype

        if not isinstance(dtype, str):
            raise ValueError(f"Unsupported dtype specification `{dtype}`.")

        normalized = dtype.lower()
        aliases = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        resolved = aliases.get(normalized)
        if resolved is None:
            raise ValueError(f"Unsupported dtype specification `{dtype}`.")
        return resolved

    def _autocast_context(self, model_kwargs):
        if self.autocast_dtype is None:
            return nullcontext()

        for value in model_kwargs.values():
            if torch.is_tensor(value):
                device_type = value.device.type
                if device_type in {"cuda", "cpu"}:
                    return torch.autocast(device_type=device_type, dtype=self.autocast_dtype)
                break
        return nullcontext()

    @staticmethod
    def _infer_hidden_size(config):
        for attr in ("hidden_size", "d_model", "n_embd", "dim"):
            value = getattr(config, attr, None)
            if value is not None:
                return value
        raise ValueError(
            "Unable to infer hidden size from the Hugging Face config. "
            "Extend `_infer_hidden_size` for this architecture."
        )

    @staticmethod
    def _infer_output_size(config, output):
        if output in {"last_hidden_state", "hidden_states_last", "pooled_output"}:
            return HFModelAdapter._infer_hidden_size(config)
        if output == "logits":
            for attr in ("num_labels", "vocab_size"):
                value = getattr(config, attr, None)
                if value is not None:
                    return value
            return HFModelAdapter._infer_hidden_size(config)
        raise ValueError(f"Unsupported HF adapter output mode `{output}`.")

    def _prepare_inputs(self, x, attention_mask):
        attention_mask = self._coerce_attention_mask(x, attention_mask)
        if self._looks_like_token_ids(x):
            x = x.squeeze(-1) if x.ndim == 3 and x.size(-1) == 1 else x
            inputs = {"input_ids": x.long()}
        else:
            inputs = {"inputs_embeds": x}

        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        return inputs

    def _coerce_attention_mask(self, x, attention_mask):
        if attention_mask is None:
            if self.pad_token_id is None or not self._looks_like_token_ids(x):
                return None
            ids = x.squeeze(-1) if x.ndim == 3 and x.size(-1) == 1 else x
            return ids.ne(self.pad_token_id).long()

        if not torch.is_tensor(attention_mask):
            return attention_mask

        if attention_mask.ndim == 3 and attention_mask.size(-1) == 1:
            attention_mask = attention_mask.squeeze(-1)

        if attention_mask.ndim == 1:
            seq_len = self._sequence_length(x)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            attention_mask = positions < attention_mask.unsqueeze(1)

        return attention_mask.long()

    @staticmethod
    def _looks_like_token_ids(x):
        return torch.is_tensor(x) and not torch.is_floating_point(x)

    @staticmethod
    def _sequence_length(x):
        if x.ndim == 0:
            raise ValueError("Expected batched sequence inputs for HFModelAdapter.")
        if x.ndim == 1:
            return x.size(0)
        if x.ndim == 2:
            return x.size(1)
        return x.size(-2)

    def _extract_output(self, outputs):
        if self.output == "last_hidden_state":
            value = getattr(outputs, "last_hidden_state", None)
            if value is not None:
                return value
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None:
                return hidden_states[-1]
            raise ValueError("Hugging Face model output does not expose last hidden states.")

        if self.output == "hidden_states_last":
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise ValueError(
                    "Requested `hidden_states_last`, but the model did not return hidden states."
                )
            return hidden_states[-1]

        if self.output == "pooled_output":
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is not None:
                return last_hidden_state[:, 0]
            raise ValueError("Hugging Face model output does not expose a pooled representation.")

        if self.output == "logits":
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise ValueError("Requested `logits`, but the Hugging Face model did not return logits.")
            return logits

        raise ValueError(f"Unsupported HF adapter output mode `{self.output}`.")
