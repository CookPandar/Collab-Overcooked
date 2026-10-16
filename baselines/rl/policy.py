"""The trained model: a frozen base LM with one LoRA adapter per role, plus frozen copies of the initial
(SFT) adapters used as the KL reference."""

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

ROLE_ADAPTERS = {0: "Chef", 1: "Assistant"}
REFERENCE_ADAPTERS = {0: "kl_ref_0", 1: "kl_ref_1"}  # PEFT selects adapter keys by substring: names must not overlap


def load_adapter_exact(model, path, name):
    """Load LoRA weights in fp32. Loading into bf16 modules first would round the checkpoint."""
    weights = load_peft_weights(str(path), device="cpu")
    for param_name, param in model.named_parameters():
        if "lora_" in param_name and f".{name}." in param_name:
            param.data = param.data.float()
    set_peft_model_state_dict(model, weights, adapter_name=name)
    loaded = get_peft_model_state_dict(model, adapter_name=name)
    if set(loaded) != set(weights) or any(not torch.equal(loaded[k].cpu().float(), weights[k].float()) for k in weights):
        raise ValueError(f"adapter {name} was not restored exactly from {path}")


class LoraPolicy:
    def __init__(self, model_path, adapter_paths, reference_paths=None, device="cuda",
                 gradient_checkpointing=False, attn_implementation="sdpa", dtype=torch.bfloat16):
        """`adapter_paths` / `reference_paths`: {agent_index: LoRA directory}."""
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        torch.backends.cuda.matmul.allow_tf32 = True
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype,
                                                    attn_implementation=attn_implementation,
                                                    device_map={"": str(self.device)})
        adapters = [(ROLE_ADAPTERS[i], p, True) for i, p in sorted(adapter_paths.items())]
        adapters += [(REFERENCE_ADAPTERS[i], p, False) for i, p in sorted((reference_paths or {}).items())]
        name, path, trainable = adapters[0]
        self.model = PeftModel.from_pretrained(base, str(path), adapter_name=name, is_trainable=trainable)
        for name, path, trainable in adapters[1:]:
            self.model.load_adapter(str(path), adapter_name=name, is_trainable=trainable)
        for name, path, _ in adapters:
            load_adapter_exact(self.model, path, name)
        self.trainable = {i: ROLE_ADAPTERS[i] for i in adapter_paths}
        self.references = {i: REFERENCE_ADAPTERS[i] for i in (reference_paths or {})}
        self._set_requires_grad()
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        self.model.eval()  # no LoRA dropout: sampling and scoring use the same function

    def _set_requires_grad(self):
        trainable = set(self.trainable.values())
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name and any(f".{a}." in name for a in trainable)

    def parameters(self, agent_index=None):
        names = self.trainable.values() if agent_index is None else [self.trainable[agent_index]]
        return [p for n, p in self.model.named_parameters()
                if "lora_" in n and any(f".{a}." in n for a in names)]

    def _use(self, adapter):
        self.model.set_adapter(adapter)
        self._set_requires_grad()  # set_adapter freezes the inactive adapters

    def token_logprobs(self, adapter, prompts, responses, temperature):
        """Log-probabilities of each response token, one fp32 tensor per sample.

        The prompt prefix shared by all samples is encoded once and its key/value cache is reused by every
        sample (gradients flow through it as well). The remaining tokens are right-padded without an attention
        mask: with causal attention the real tokens never see the padding."""
        self._use(adapter)
        sequences = [torch.cat([p, r]) for p, r in zip(prompts, responses)]
        shared = min(len(p) for p in prompts) - 1
        for p in prompts[1:]:
            mismatch = (p[:shared] != prompts[0][:shared]).nonzero()
            if len(mismatch):
                shared = int(mismatch[0])
        base = self.model.get_base_model()
        cache = None
        if len(prompts) > 1 and shared >= 64:
            cache = base.model(input_ids=prompts[0][None, :shared].to(self.device), use_cache=True).past_key_values
            cache.batch_repeat_interleave(len(prompts))
        else:
            shared = 0
        suffixes = pad_sequence([s[shared:] for s in sequences], batch_first=True,
                                padding_value=self.tokenizer.pad_token_id).to(self.device)
        position_ids = torch.arange(shared, shared + suffixes.shape[1], device=self.device)[None].expand_as(suffixes)
        hidden = base.model(input_ids=suffixes, position_ids=position_ids, past_key_values=cache,
                            use_cache=cache is not None).last_hidden_state
        positions = [(i, len(p) - 1 - shared, len(r)) for i, (p, r) in enumerate(zip(prompts, responses))]
        gathered = torch.cat([hidden[i, start:start + n] for i, start, n in positions])
        logits = base.lm_head(gathered).float() / temperature
        targets = torch.cat(list(responses)).to(self.device)
        logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        return list(torch.split(logprobs, [n for _, _, n in positions]))

    def save(self, agent_index, path):
        self.model.save_pretrained(str(path), selected_adapters=[self.trainable[agent_index]])
