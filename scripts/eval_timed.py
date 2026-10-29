"""Run lerobot-eval unchanged, but time every action-chunk inference call.

Usage: python scripts/eval_timed.py <timing_out.json> <lerobot-eval args...>
Wraps the policy's chunk-producing method (SmolVLA._get_action_chunk,
PI0Fast.predict_action_chunk) with a CUDA-synchronised timer; the eval itself is not altered.
"""
import atexit, json, sys, time
import torch

torch.set_num_threads(4)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3)

out = sys.argv[1]
sys.argv = ["lerobot-eval"] + sys.argv[2:]
times = []


def timed(fn):
    def w(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        return r
    return w


# Workaround (transformers 5.5.4): AutoProcessor cannot rebuild the FAST action processor from its repo
# (bpe tokenizer sub-load raises). Build it by hand from the same pinned files (ckpt/fast_tokenizer).
import transformers
_orig_ap = transformers.AutoProcessor.from_pretrained.__func__


def _ap(cls, name, *a, **k):
    if str(name).endswith("fast-libero-tokenizer-mean-std") or "fast_tokenizer" in str(name):
        import importlib.util, json, os
        d = "ckpt/fast_tokenizer"
        spec = importlib.util.spec_from_file_location("processing_action_tokenizer", f"{d}/processing_action_tokenizer.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        cfg = json.load(open(f"{d}/processor_config.json"))
        bpe = transformers.PreTrainedTokenizerFast(tokenizer_file=f"{d}/tokenizer.json")
        return m.UniversalActionProcessor(bpe, scale=cfg["scale"], vocab_size=cfg["vocab_size"],
                                          min_token=cfg["min_token"], action_dim=cfg["action_dim"],
                                          time_horizon=cfg["time_horizon"])
    return _orig_ap(cls, name, *a, **k)


transformers.AutoProcessor.from_pretrained = classmethod(_ap)

# google/paligemma-3b-pt-224 is gated (401 without an HF login): redirect every tokenizer load of it
# to a local copy of the tokenizer files from the ungated mirror nnh-pbbb/paligemma-3b-pt-224.
_orig_at = transformers.AutoTokenizer.from_pretrained.__func__


def _at(cls, name, *a, **k):
    if str(name) == "google/paligemma-3b-pt-224":
        name = "ckpt/paligemma_tokenizer"
    return _orig_at(cls, name, *a, **k)


transformers.AutoTokenizer.from_pretrained = classmethod(_at)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
SmolVLAPolicy._get_action_chunk = timed(SmolVLAPolicy._get_action_chunk)
PI0FastPolicy.predict_action_chunk = timed(PI0FastPolicy.predict_action_chunk)


# Speed-up for the FAST policy, output-identical: lerobot always decodes max_decoding_steps=256 tokens, but
# detokenize_actions() drops everything after the first "|" (end-of-action marker). Stop decoding at "|".
# Disable with EARLY_STOP=0 to run the unmodified decoder.
import inspect, os, textwrap
import lerobot.policies.pi0_fast.modeling_pi0_fast as _m
if os.environ.get("EARLY_STOP", "1") == "1" and hasattr(_m, "PI0FastPytorch"):
    _src = textwrap.dedent(inspect.getsource(_m.PI0FastPytorch.sample_actions_fast_kv_cache))
    _needle = "generated_action_tokens[:, t] = next_token.squeeze(-1)\n"
    assert _needle in _src
    _src = _src.replace(_needle, _needle + "        if bsize == 1 and next_token.item() == self._pipe_id:\n            break\n")
    exec(_src, _m.__dict__)
    _fn = _m.__dict__["sample_actions_fast_kv_cache"]

    def _patched(self, *a, **k):
        if not hasattr(self, "_pipe_id"):
            self._pipe_id = self._paligemma_tokenizer.convert_tokens_to_ids("|")
        return _fn(self, *a, **k)
    _m.PI0FastPytorch.sample_actions_fast_kv_cache = _patched


def dump():
    import statistics as s
    t = times[1:] or times  # drop first call (CUDA warmup)
    json.dump({"n_calls": len(times), "mean_s": s.mean(t) if t else None,
               "median_s": s.median(t) if t else None,
               "p95_s": sorted(t)[int(0.95 * (len(t) - 1))] if t else None,
               "raw_s": times}, open(out, "w"))


atexit.register(dump)
from lerobot.scripts.lerobot_eval import main
main()
