"""One model with one set of current weights (the unedited weights plus the current edits), seen two ways:
vLLM for generation and Hugging Face transformers for teacher-forced reads (log-probs, hidden states).
Every change of the edits goes to both, in place, so that the model that writes the next answer IS the edited model."""
import contextlib
import os

import torch

from . import config as C


def _worker_model(worker):
    w = getattr(worker, "worker", worker)
    return w.model_runner.model


def _vllm_load(worker, items):
    """runs inside the vLLM worker: write the given (checkpoint name, tensor) pairs with the model's own loader."""
    model = _worker_model(worker)
    loaded = model.load_weights(weights=iter(items))
    return len(loaded) if loaded is not None else -1


def _vllm_param_sums(worker, suffixes):
    """runs inside the vLLM worker: float sum of every parameter whose name ends with one of the suffixes."""
    model = _worker_model(worker)
    out = {}
    for n, p in model.named_parameters():
        for s in suffixes:
            if n.endswith(s):
                out[s] = float(p.detach().float().sum().item())
    return out


class Engine:
    def __init__(self, model_dir=C.MODEL_DIR, with_hf=True):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from safetensors import safe_open
        from vllm import LLM
        self.model_dir = model_dir
        kw = dict(model=model_dir, dtype="bfloat16", gpu_memory_utilization=C.GPU_MEM, max_model_len=C.MAX_MODEL_LEN,
                  max_num_seqs=C.MAX_NUM_SEQS, seed=0, enable_prefix_caching=False)
        try:
            self.llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
        except TypeError:
            self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self._st = safe_open(C.weights_file(model_dir), framework="pt")
        self.names = sorted(n for n in self._st.keys() if n.startswith(C.LM_PREFIX))
        self.edits = {}
        self.hf = None
        if with_hf:
            self._load_hf()

    # ------------------------------------------------------------------ weights
    def original(self, name):
        return self._st.get_tensor(name)

    def current(self, name):
        return self.edits[name] if name in self.edits else self.original(name)

    def _load_hf(self):
        import transformers as T
        last = None
        for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
            for dkw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
                try:
                    self.hf = getattr(T, cls).from_pretrained(self.model_dir, **dkw)
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
            if self.hf is not None:
                break
        if self.hf is None:
            raise RuntimeError(f"could not load the model with transformers: {last!r}")
        self.hf = self.hf.to("cuda").eval()
        params = dict(self.hf.named_parameters())
        self._hf_name = {}
        for n in self.names:
            for cand in (n, n.replace("model.language_model.", "model."), n.replace("model.language_model.", "language_model.model.")):
                if cand in params:
                    self._hf_name[n] = cand
                    break
        missing = [n for n in self.names if n not in self._hf_name and n not in C.FROZEN]
        if missing:
            raise RuntimeError(f"{len(missing)} checkpoint tensors not found in the transformers model, e.g. {missing[:3]}")
        self._hf_params = params
        base = getattr(self.hf, "model", self.hf)
        self.hf_lm = getattr(base, "language_model", base)
        self.hf_embed = self.hf_lm.embed_tokens.weight

    def _push(self, named):
        items = [(n, t.contiguous()) for n, t in named.items()]
        self.llm.collective_rpc(_vllm_load, args=(items,))
        with contextlib.suppress(Exception):
            self.llm.reset_prefix_cache()
        if self.hf is not None:
            with torch.no_grad():
                for n, t in named.items():
                    p = self._hf_params[self._hf_name[n]]
                    p.copy_(t.to(p.device, p.dtype))

    def set_edits(self, new):
        """make the complete set of edits `new` (name -> tensor) current; returns the names whose values changed."""
        target = {}
        for n, t in new.items():
            o = self.original(n)
            t = t.detach().to("cpu").to(o.dtype).reshape(o.shape).contiguous()
            if not torch.equal(t, o):
                target[n] = t
        changed = {}
        for n in set(self.edits) | set(target):
            if n in target:
                if n not in self.edits or not torch.equal(self.edits[n], target[n]):
                    changed[n] = target[n]
            else:
                changed[n] = self.original(n)
        if changed:
            self._push(changed)
        self.edits = target
        return sorted(changed)

    @contextlib.contextmanager
    def using(self, edits):
        """temporarily make `edits` current (e.g. {} for the original model)."""
        saved = dict(self.edits)
        self.set_edits(edits)
        try:
            yield
        finally:
            self.set_edits(saved)

    def vllm_sums(self, names):
        suf = {n: n.split("language_model.")[-1] for n in names}
        got = self.llm.collective_rpc(_vllm_param_sums, args=(list(suf.values()),))
        got = got[0] if isinstance(got, list) else got
        return {n: got.get(s) for n, s in suf.items()}

    def changed_entries(self):
        return {n: int((t != self.original(n)).sum()) for n, t in self.edits.items()}

    # ------------------------------------------------------------------ generation
    def chat(self, messages, thinking):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=thinking)

    def generate(self, prompts, params):
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        res = []
        for o in outs:
            s = o.outputs[0]
            res.append({"text": s.text, "ids": list(s.token_ids), "prompt_ids": list(o.prompt_token_ids),
                        "finish": s.finish_reason, "tokens": len(s.token_ids)})
        return res

    # ------------------------------------------------------------------ teacher-forced reads (transformers)
    @torch.no_grad()
    def hidden_means(self, seqs, layers):
        """seqs: list of (token ids, start, end). Mean over positions [start, end) of hidden_states[l] for l in layers
        (0 = embeddings, l = after layer l counted from 1). Returns float32 tensor [len(seqs), len(layers), hidden]."""
        out = torch.zeros(len(seqs), len(layers), C.HIDDEN)
        for i, (ids, a, b) in enumerate(seqs):
            x = torch.tensor([ids[:b]], device="cuda")
            hs = self.hf_lm(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states
            for j, l in enumerate(layers):
                out[i, j] = hs[l][0, a:b].float().mean(0).cpu()
        return out

    @torch.no_grad()
    def token_logprobs(self, ids):
        """log-prob of each token given the ones before it (len(ids) - 1 values)."""
        x = torch.tensor([ids], device="cuda")
        h = self.hf_lm(input_ids=x, use_cache=False).last_hidden_state[0]
        y = x[0, 1:]
        n = len(ids) - 1
        lp = torch.empty(n)
        for s in range(0, n, 512):
            e = min(s + 512, n)
            lg = (h[s:e] @ self.hf_embed.T).float().log_softmax(-1)
            lp[s:e] = lg.gather(-1, y[s:e, None])[:, 0].cpu()
        return lp
