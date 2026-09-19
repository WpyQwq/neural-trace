"""Lazy local model adapter for observable forward and backward traces.

The adapter is deliberately conservative: it summarizes tensors in the
runtime process instead of sending activations or weights to the browser.
Large checkpoints are loaded only after the first stream request and failures
fall back to the mock event contract in ``main.py``.
"""

from __future__ import annotations

import importlib.util
import base64
import json
import queue
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

DEFAULT_ARCHITECTURE = {
    "model_type": "qwen3_5",
    "num_layers": 32,
    "pattern": "8 × (3L + 1A)",
    "hidden_size": 2560,
    "vocab_size": 248320,
    "context_length": 262144,
    "intermediate_size": 9216,
    "delta_heads_v": 32,
    "delta_heads_qk": 16,
    "attention_heads": 16,
    "attention_kv_heads": 4,
    "attention_head_dim": 256,
    "rope_dim": 64,
    "vision_encoder": True,
    "sparse_moe": True,
    "mtp": True,
    "precision": "BF16 / INT4",
}


class ModelRuntime:
    def __init__(self, model_dir: Path, model_name: str) -> None:
        self.model_dir = model_dir
        self.model_name = model_name
        self.status = "idle"
        self.error: str | None = None
        self.model: Any = None
        self.processor: Any = None
        self._torch: Any = None
        self._hooks: list[Any] = []
        self._load_lock = threading.Lock()
        self._trace_context = threading.local()

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.model is not None

    @property
    def checkpoint_path(self) -> Path | None:
        if not self.model_dir.exists():
            return None
        candidates = [self.model_dir]
        candidates.extend(path for path in sorted(self.model_dir.iterdir()) if path.is_dir())
        for path in candidates:
            has_config = (path / "config.json").exists()
            has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.safetensors.index.json"))
            if has_config and has_weights:
                return path
        return None

    @property
    def checkpoint_ready(self) -> bool:
        return self.checkpoint_path is not None

    @property
    def architecture(self) -> dict[str, Any]:
        architecture = dict(DEFAULT_ARCHITECTURE)
        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            return architecture
        try:
            config = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return architecture
        text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
        field_map = {
            "num_hidden_layers": "num_layers",
            "hidden_size": "hidden_size",
            "vocab_size": "vocab_size",
            "max_position_embeddings": "context_length",
            "intermediate_size": "intermediate_size",
            "num_attention_heads": "attention_heads",
            "num_key_value_heads": "attention_kv_heads",
            "head_dim": "attention_head_dim",
            "rope_theta_dim": "rope_dim",
            "model_type": "model_type",
        }
        for source_key, target_key in field_map.items():
            value = text_config.get(source_key)
            if isinstance(value, (int, float, str)):
                architecture[target_key] = value
        if not architecture.get("attention_head_dim") and architecture.get("attention_heads"):
            architecture["attention_head_dim"] = architecture["hidden_size"] // architecture["attention_heads"]
        return architecture

    def ensure_loaded(self) -> bool:
        if self.ready:
            return True
        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            self.status = "downloading"
            return False

        with self._load_lock:
            if self.ready:
                return True
            self.status = "loading"
            self.error = None
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoProcessor

                self._torch = torch
                try:
                    self.processor = AutoProcessor.from_pretrained(checkpoint_path, local_files_only=True, trust_remote_code=True)
                except Exception:
                    from transformers import AutoTokenizer

                    self.processor = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True, trust_remote_code=True)

                model_class = AutoModelForCausalLM
                image_text_model = getattr(__import__("transformers", fromlist=["AutoModelForImageTextToText"]), "AutoModelForImageTextToText", None)
                if image_text_model is not None:
                    model_class = image_text_model

                load_kwargs = {
                    "local_files_only": True,
                    "trust_remote_code": True,
                    "torch_dtype": "auto",
                }
                if importlib.util.find_spec("accelerate") is not None:
                    load_kwargs.update(device_map="auto", low_cpu_mem_usage=True)

                try:
                    self.model = model_class.from_pretrained(
                        checkpoint_path,
                        **load_kwargs,
                    )
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_path,
                        **load_kwargs,
                    )

                self.model.eval()
                self._install_hooks()
                self.status = "ready"
                return True
            except Exception as exc:  # pragma: no cover - depends on local GPU/runtime
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                self.model = None
                return False

    def start_trace(self, prompt: str, mode: str, image_data: str | None = None) -> queue.Queue[dict[str, Any]] | None:
        if not self.ensure_loaded():
            return None
        events: queue.Queue[dict[str, Any]] = queue.Queue()
        worker = threading.Thread(target=self._trace_worker, args=(prompt, mode, image_data, events), daemon=True)
        worker.start()
        return events

    def _install_hooks(self) -> None:
        if self._hooks or self.model is None:
            return

        seen_layers: set[int] = set()
        for name, module in self.model.named_modules():
            match = re.search(r"(?:^|\.)(?:language_model\.)?(?:layers|(?:model|transformer|decoder)\.(?:layers|h))\.(\d+)$", name)
            if match is not None:
                layer_index = int(match.group(1))
                if layer_index not in seen_layers:
                    seen_layers.add(layer_index)
                    self._hooks.append(module.register_forward_hook(self._make_forward_hook(layer_index)))
                    self._hooks.append(module.register_full_backward_hook(self._make_backward_hook(layer_index)))
            elif re.search(r"(?:^|\.)\b(?:visual|vision_model|vision_tower)\b$", name):
                self._hooks.append(module.register_forward_hook(self._make_vision_hook()))
            elif re.search(r"(?:^|\.)\b(?:router|routing|gate|experts|expert_gate|mtp)\b$", name, re.IGNORECASE):
                kind = "router_weights" if re.search(r"router|routing|gate", name, re.IGNORECASE) else "expert_mixture"
                if re.search(r"mtp", name, re.IGNORECASE):
                    kind = "mtp_logits"
                self._hooks.append(module.register_forward_hook(self._make_aux_hook(name, kind)))
            elif re.search(r"(?:^|\.)lm_head$", name):
                self._hooks.append(module.register_forward_hook(self._make_logits_hook()))

    def _make_forward_hook(self, layer_index: int):
        def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
            context = getattr(self._trace_context, "queue", None)
            if context is None:
                return
            tensor = self._first_tensor(output)
            summary = self._summarize(tensor)
            full_attention = (layer_index + 1) % 4 == 0
            mode = getattr(self._trace_context, "mode", "forward")
            step = getattr(self._trace_context, "step", 0)
            self._trace_context.step = step + 1
            if mode == "attention":
                kind = "attention_output" if full_attention else "delta_context"
            elif mode == "state":
                kind = "delta_state"
            else:
                kind = "full_attention" if full_attention else "delta_state"
            context.put({
                "step": min(step, 511),
                "layer": f"L{layer_index + 1:02d}",
                "kind": kind,
                "mode": mode,
                "value": summary["rms"],
                "mean": summary["mean"],
                "shape": summary["shape"],
                "source": "forward_hook",
            })

        return forward_hook

    def _make_vision_hook(self):
        def vision_hook(_module: Any, _inputs: Any, output: Any) -> None:
            context = getattr(self._trace_context, "queue", None)
            if context is None:
                return
            tensor = self._first_tensor(output)
            summary = self._summarize(tensor)
            step = getattr(self._trace_context, "step", 0)
            self._trace_context.step = step + 1
            context.put({
                "step": min(step, 511),
                "kind": "vision_encoder",
                "mode": getattr(self._trace_context, "mode", "forward"),
                "value": summary["rms"],
                "mean": summary["mean"],
                "shape": summary["shape"],
                "source": "vision_hook",
            })

        return vision_hook

    def _make_aux_hook(self, module_name: str, kind: str):
        def aux_hook(_module: Any, _inputs: Any, output: Any) -> None:
            context = getattr(self._trace_context, "queue", None)
            if context is None:
                return
            tensor = self._first_tensor(output)
            summary = self._summarize(tensor)
            layer_match = re.search(r"layers\.(\d+)", module_name)
            layer = f"L{int(layer_match.group(1)) + 1:02d}" if layer_match else "AUX"
            step = getattr(self._trace_context, "step", 0)
            self._trace_context.step = step + 1
            context.put({
                "step": min(step, 511),
                "layer": layer,
                "kind": kind,
                "mode": getattr(self._trace_context, "mode", "forward"),
                "value": summary["rms"],
                "mean": summary["mean"],
                "shape": summary["shape"],
                "module": module_name,
                "source": "aux_hook",
            })

        return aux_hook

    def _make_backward_hook(self, layer_index: int):
        def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
            context = getattr(self._trace_context, "queue", None)
            if context is None or getattr(self._trace_context, "mode", "forward") != "backward":
                return
            tensor = self._first_tensor(grad_output)
            summary = self._summarize(tensor)
            step = getattr(self._trace_context, "step", 0)
            self._trace_context.step = step + 1
            context.put({
                "step": min(step, 511),
                "layer": f"L{layer_index + 1:02d}",
                "kind": "gradient",
                "value": summary["rms"],
                "mean": summary["mean"],
                "shape": summary["shape"],
                "source": "backward_hook",
            })

        return backward_hook

    def _make_logits_hook(self):
        def logits_hook(_module: Any, _inputs: Any, output: Any) -> None:
            context = getattr(self._trace_context, "queue", None)
            if context is None:
                return
            tensor = self._first_tensor(output)
            if tensor is not None and tensor.ndim >= 2:
                tensor = tensor[:, -1, :]
            summary = self._summarize(tensor)
            step = getattr(self._trace_context, "step", 0)
            self._trace_context.step = step + 1
            context.put({
                "step": min(step, 511),
                "layer": "OUT",
                "kind": "logits",
                "value": summary["rms"],
                "mean": summary["mean"],
                "shape": summary["shape"],
                "source": "logits_hook",
            })

        return logits_hook

    def _trace_worker(self, prompt: str, mode: str, image_data: str | None, events: queue.Queue[dict[str, Any]]) -> None:
        self._trace_context.queue = events
        self._trace_context.mode = mode
        self._trace_context.step = 0
        try:
            inputs = self._encode(prompt, image_data)
            if image_data and "pixel_values" in inputs:
                summary = self._summarize(inputs["pixel_values"])
                events.put({
                    "step": 0,
                    "kind": "vision_input",
                    "mode": mode,
                    "value": summary["rms"],
                    "mean": summary["mean"],
                    "shape": summary["shape"],
                    "source": "processor",
                })
            if mode == "backward":
                self._run_backward(inputs, events)
            elif mode == "attention":
                self._run_attention(inputs, events)
            else:
                self._run_generation(inputs, events)
        except Exception as exc:  # pragma: no cover - depends on checkpoint/runtime
            events.put({"kind": "runtime_error", "error": f"{type(exc).__name__}: {exc}", "source": "model_runtime"})
        finally:
            events.put({"kind": "done", "source": "model_runtime"})
            self._trace_context.queue = None

    def _run_attention(self, inputs: dict[str, Any], events: queue.Queue[dict[str, Any]]) -> None:
        torch = self._torch
        try:
            with torch.inference_mode():
                outputs = self.model(
                    **inputs,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
            attentions = getattr(outputs, "attentions", None)
            emitted = False
            if attentions:
                for layer_index, attention in enumerate(attentions):
                    if attention is None or getattr(attention, "ndim", 0) < 4:
                        continue
                    size = min(7, int(attention.shape[-1]))
                    matrix = attention[0, :, -size:, -size:].float().mean(dim=0)
                    values = matrix.detach().cpu().reshape(-1).tolist()
                    summary = self._summarize(matrix)
                    step = getattr(self._trace_context, "step", 0)
                    self._trace_context.step = step + 1
                    events.put({
                        "step": min(step, 511),
                        "layer": f"L{layer_index + 1:02d}",
                        "kind": "attention_weight",
                        "mode": "attention",
                        "value": summary["rms"],
                        "mean": summary["mean"],
                        "shape": list(attention.shape),
                        "attention": [round(float(value), 6) for value in values],
                        "attention_shape": [size, size],
                        "source": "attention_output",
                    })
                    emitted = True
            if not emitted:
                events.put({
                    "step": min(getattr(self._trace_context, "step", 0), 511),
                    "kind": "attention_unavailable",
                    "mode": "attention",
                    "source": "model_runtime",
                })
        except Exception:
            events.put({
                "step": min(getattr(self._trace_context, "step", 0), 511),
                "kind": "attention_unavailable",
                "mode": "attention",
                "source": "model_runtime",
            })
        self._run_generation(inputs, events)

    def _encode(self, prompt: str, image_data: str | None = None) -> dict[str, Any]:
        processor = self.processor
        rendered_prompt = prompt
        image = None
        if image_data:
            try:
                from PIL import Image

                encoded_image = image_data.split(",", 1)[-1]
                image = Image.open(BytesIO(base64.b64decode(encoded_image))).convert("RGB")
            except Exception as exc:
                raise RuntimeError(f"invalid image input: {type(exc).__name__}") from exc
        apply_chat_template = getattr(processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}] if image is not None else prompt
                rendered_prompt = apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered_prompt = prompt
        processor_kwargs: dict[str, Any] = {"text": rendered_prompt, "return_tensors": "pt", "truncation": True, "max_length": 256}
        if image is not None:
            processor_kwargs["images"] = image
        try:
            encoded = processor(**processor_kwargs)
        except Exception:
            processor_kwargs.pop("truncation", None)
            processor_kwargs.pop("max_length", None)
            encoded = processor(**processor_kwargs)
        device = self._input_device()
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}

    def _run_generation(self, inputs: dict[str, Any], events: queue.Queue[dict[str, Any]]) -> None:
        torch = self._torch
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        input_length = 0
        if "input_ids" in inputs and getattr(inputs["input_ids"], "ndim", 0) >= 2:
            input_length = int(inputs["input_ids"].shape[-1])
        if getattr(generated, "ndim", 0) >= 2 and input_length:
            generated = generated[:, input_length:]
        try:
            text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception:
            tokenizer = getattr(self.processor, "tokenizer", None)
            try:
                text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0] if tokenizer is not None else ""
            except Exception:
                text = ""
        events.put({
            "step": min(getattr(self._trace_context, "step", 0), 511),
            "kind": "generation",
            "token": text[-1:] if text else "",
            "text": text,
            "value": 1.0,
            "source": "model_runtime",
        })

    def _run_backward(self, inputs: dict[str, Any], events: queue.Queue[dict[str, Any]]) -> None:
        torch = self._torch
        if "input_ids" not in inputs:
            raise RuntimeError("processor did not return input_ids; backward trace requires text tokens")
        self.model.zero_grad(set_to_none=True)
        labels = inputs["input_ids"].clone()
        with torch.enable_grad():
            outputs = self.model(**inputs, labels=labels, use_cache=False)
            loss = outputs.loss
            loss.backward()
        events.put({
            "step": min(getattr(self._trace_context, "step", 0), 511),
            "kind": "loss",
            "loss": float(loss.detach().float().item()),
            "value": float(loss.detach().float().item()),
            "source": "autograd",
        })

    def _input_device(self):
        try:
            return self.model.device
        except Exception:
            for parameter in self.model.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
        return self._torch.device("cpu")

    def _first_tensor(self, value: Any):
        if self._torch is not None and self._torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for item in value.values():
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _summarize(self, tensor: Any) -> dict[str, Any]:
        if tensor is None:
            return {"mean": 0.0, "rms": 0.0, "shape": []}
        torch = self._torch
        sampled = tensor.detach().float().reshape(-1)
        if sampled.numel() > 4096:
            stride = max(1, sampled.numel() // 4096)
            sampled = sampled[::stride]
        mean = float(sampled.mean().item())
        rms = float(torch.sqrt(torch.mean(sampled * sampled)).item())
        return {"mean": round(mean, 6), "rms": round(rms, 6), "shape": list(tensor.shape)}
