#!/usr/bin/env python3

import argparse
import gc
import json
import os
from pathlib import Path

from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _blank_image(size=64):
    return Image.new("RGB", (size, size), color=(127, 127, 127))


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_dtype():
    import torch

    return torch.float16 if _device() == "cuda" else torch.float32


def _cleanup_torch():
    try:
        import torch
    except Exception:
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _accelerate_available() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except Exception:
        return False


def _move_batch(inputs, device: str, float_dtype=None):
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue

        if float_dtype is not None and getattr(value, "is_floating_point", lambda: False)():
            moved[key] = value.to(device=device, dtype=float_dtype)
        else:
            moved[key] = value.to(device)
    return moved


def _format_error(exc: Exception) -> str:
    message = str(exc)
    if isinstance(exc, ImportError) and "timm" in message:
        return (
            f"{message}\n"
            "Install the missing Florence dependency in the active environment with:\n"
            "  ./venv310/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ timm"
        )
    return message or repr(exc)


def smoke_qwen(model_path: Path):
    import torch

    model_type = ""
    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        model_type = json.loads(cfg_path.read_text()).get("model_type", "")

    try:
        from transformers.models.qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )

        model_cls = Qwen2_5_VLForConditionalGeneration
        processor_cls = Qwen2_5_VLProcessor
        backend = "qwen2_5_vl"
    except Exception:
        from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

        model_cls = Qwen2VLForConditionalGeneration
        processor_cls = Qwen2VLProcessor
        backend = "qwen2_vl"

    print(f"[qwen] config_model_type={model_type} api_backend={backend}")
    processor = processor_cls.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    use_device_map = _device() == "cuda" and _accelerate_available()
    model = model_cls.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=_torch_dtype(),
        device_map="auto" if use_device_map else None,
    )
    if not getattr(model, "hf_device_map", None):
        model.to(_device())
    model.eval()

    image = _blank_image()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the image in one short phrase."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = _move_batch(inputs, _device(), float_dtype=_torch_dtype() if _device() == "cuda" else None)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
    else:
        decoded = []
    print(f"[qwen] generation_ok={bool(decoded)} output_preview={decoded[0][:120] if decoded else ''}")


def smoke_grounding(model_path: Path):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(str(model_path), local_files_only=True).to(_device())
    model.eval()

    image = _blank_image()
    text = "gray square."
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = _move_batch(inputs, _device())
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=0.1,
        text_threshold=0.1,
        target_sizes=[image.size[::-1]],
    )
    count = len(results[0]["boxes"]) if results else 0
    print(f"[grounding] inference_ok=True boxes={count}")


def smoke_florence(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=_torch_dtype(),
    ).to(_device())
    model.eval()

    image = _blank_image()
    prompt = "<CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = _move_batch(inputs, _device(), float_dtype=_torch_dtype() if _device() == "cuda" else None)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    print(f"[florence] generation_ok={bool(text)} output_preview={text[:120]}")


def main():
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Local model smoke test for VLN model assets.")
    parser.add_argument("--qwen-path", default="")
    parser.add_argument("--grounding-path", default="")
    parser.add_argument("--florence-path", default=str(root_dir / "src" / "llm_model" / "models" / "microsoft--Florence-2-large"))
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-grounding", action="store_true")
    parser.add_argument("--skip-florence", action="store_true")
    args = parser.parse_args()

    print(f"[env] device={_device()}")

    failures = []

    checks = []
    if not args.skip_florence:
        florence_path = Path(args.florence_path)
        if not florence_path.exists():
            raise FileNotFoundError(f"Missing Florence path: {florence_path}")
        checks.append(("florence", florence_path, smoke_florence))

    if not args.skip_qwen:
        qwen_path = Path(args.qwen_path)
        if not qwen_path.exists():
            raise FileNotFoundError(f"Missing Qwen path: {qwen_path}")
        checks.append(("qwen", qwen_path, smoke_qwen))

    if not args.skip_grounding:
        grounding_path = Path(args.grounding_path)
        if not grounding_path.exists():
            raise FileNotFoundError(f"Missing GroundingDINO path: {grounding_path}")
        checks.append(("grounding", grounding_path, smoke_grounding))

    for label, path, fn in checks:
        print(f"[{label}] smoke_start path={path}")
        try:
            fn(path)
        except Exception as exc:
            failures.append((label, exc))
            print(f"[{label}] smoke_failed: {_format_error(exc)}")
        finally:
            _cleanup_torch()

    if failures:
        print("[summary] smoke test completed with failures:")
        for label, exc in failures:
            print(f"[summary] {label}: {_format_error(exc)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
