#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


QWEN_REPO = "Qwen/Qwen2-VL-2B-Instruct"
GROUNDING_REPO = "IDEA-Research/grounding-dino-base"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

QWEN_MANIFEST = [
    "LICENSE",
    "README.md",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

GROUNDING_MANIFEST = [
    "README.md",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def _missing_manifest_files(local_dir: Path, filenames):
    missing = []
    for name in filenames:
        candidate = local_dir / name
        if not candidate.exists() or candidate.stat().st_size <= 0:
            missing.append(name)
    return missing


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _print_command(cmd):
    return " ".join(shlex_quote(str(x)) for x in cmd)


def shlex_quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def _run(cmd):
    print(f"[download] exec: {_print_command(cmd)}")
    subprocess.run(cmd, check=True)


def _try_run(cmd):
    print(f"[download] exec: {_print_command(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def _hf_snapshot_download(repo_id: str, local_dir: Path, endpoint: str, cache_dir: str):
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if endpoint:
        kwargs["endpoint"] = endpoint
    snapshot_download(**kwargs)


def _hf_resolve_url(endpoint: str, repo_id: str, filename: str) -> str:
    endpoint = (endpoint or DEFAULT_HF_ENDPOINT).rstrip("/")
    repo_path = "/".join(quote(part) for part in repo_id.split("/"))
    file_path = "/".join(quote(part) for part in filename.split("/"))
    return f"{endpoint}/{repo_path}/resolve/main/{file_path}"


def _download_with_python(url: str, dest: Path):
    import requests

    _ensure_dir(dest.parent)
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _download_file(url: str, dest: Path):
    _ensure_dir(dest.parent)
    wget_cmd = [
        "wget",
        "--continue",
        "--tries=5",
        "--timeout=30",
        "--waitretry=3",
        "--retry-connrefused",
        "-O",
        str(dest),
        url,
    ]
    if _try_run(wget_cmd):
        return

    curl_cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "30",
        "-o",
        str(dest),
        url,
    ]
    if _try_run(curl_cmd):
        return

    print("[download] falling back to Python requests")
    _download_with_python(url, dest)


def _download_manifest(repo_id: str, local_dir: Path, filenames, endpoint: str):
    _ensure_dir(local_dir)
    for name in filenames:
        url = _hf_resolve_url(endpoint, repo_id, name)
        dest = local_dir / name
        print(f"[download] file={name}")
        _download_file(url, dest)


def _download_qwen_modelscope(local_dir: Path):
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as e:
        if isinstance(e, ModuleNotFoundError):
            missing = getattr(e, "name", None) or str(e)
            raise RuntimeError(
                "ModelScope import failed because the current environment is missing runtime dependencies. "
                f"First missing module: {missing}\n"
                "Install the base ModelScope runtime deps in the selected environment, for example:\n"
                "  python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ "
                "pandas pyarrow datasets addict simplejson oss2 python-dateutil attrs yapf scipy "
                "sortedcontainers gast"
            ) from e
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if sys.version_info < (3, 9):
            raise RuntimeError(
                "ModelScope import failed on this Python runtime. "
                f"Current Python is {py_ver}. Newer ModelScope releases require newer Python, "
                "so on Python 3.8 you should pin an older compatible release first, for example:\n"
                "  python3 -m pip uninstall -y modelscope\n"
                "  python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --no-deps 'modelscope==1.9.5'\n"
                f"Original import error: {e}"
            ) from e
        raise RuntimeError(
            "ModelScope backend requested but modelscope import failed. "
            "Try reinstalling it with:\n"
            "  python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --force-reinstall modelscope\n"
            f"Original import error: {e}"
        ) from e

    _ensure_dir(local_dir)
    downloaded_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct", cache_dir=str(local_dir.parent))
    downloaded_path = Path(downloaded_dir)
    if downloaded_path.resolve() != local_dir.resolve():
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.copytree(downloaded_path, local_dir)


def _download_target(label: str, repo_id: str, local_dir: Path, endpoint: str, cache_dir: str, provider: str):
    if label == "qwen":
        manifest = QWEN_MANIFEST
    else:
        manifest = GROUNDING_MANIFEST

    missing = _missing_manifest_files(local_dir, manifest)
    if not missing:
        print(f"[download] {label} already present locally, skipping network download")
        return "local"
    if local_dir.exists():
        print(f"[download] {label} missing files: {', '.join(missing)}")

    if provider == "modelscope":
        if label != "qwen":
            raise RuntimeError("ModelScope backend is only implemented for qwen in this helper.")
        _download_qwen_modelscope(local_dir)
        return "modelscope"

    if provider == "hf_api":
        _hf_snapshot_download(repo_id, local_dir, endpoint, cache_dir)
        return "hf_api"

    if provider == "hf_manifest":
        _download_manifest(repo_id, local_dir, manifest, endpoint)
        return "hf_manifest"

    if provider != "auto":
        raise RuntimeError(f"Unsupported provider: {provider}")

    try:
        _hf_snapshot_download(repo_id, local_dir, endpoint, cache_dir)
        return "hf_api"
    except Exception as api_err:
        print(f"[warn] hf_api download failed for {label}: {api_err}", file=sys.stderr)

    try:
        _download_manifest(repo_id, local_dir, manifest, endpoint)
        return "hf_manifest"
    except Exception as manifest_err:
        print(f"[warn] hf_manifest download failed for {label}: {manifest_err}", file=sys.stderr)

    if label == "qwen":
        try:
            _download_qwen_modelscope(local_dir)
            return "modelscope"
        except Exception as ms_err:
            print(f"[warn] modelscope download failed for {label}: {ms_err}", file=sys.stderr)
            raise RuntimeError(
                "All qwen download backends failed. "
                "Try browser/manual download or install modelscope and rerun with --qwen-provider modelscope."
            ) from ms_err

    raise RuntimeError(
        f"All download backends failed for {label}. "
        "Try rerunning with --grounding-provider hf_manifest and a stable mirror endpoint."
    )


def main():
    root_dir = Path(__file__).resolve().parents[1]
    default_qwen_dir = root_dir / "src" / "llm_model" / "models" / "Qwen--Qwen2-VL-2B-Instruct"
    default_grounding_dir = root_dir / "src" / "llm_model" / "models" / "GroundingDino" / "grounding-dino-base"

    parser = argparse.ArgumentParser(description="Download local VLN model assets with Hugging Face and mirror fallbacks.")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Optional Hugging Face compatible endpoint, for example https://hf-mirror.com",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", ""), help="Optional cache dir for huggingface_hub")
    parser.add_argument("--qwen-repo", default=QWEN_REPO)
    parser.add_argument("--qwen-dir", default=str(default_qwen_dir))
    parser.add_argument("--qwen-provider", choices=["auto", "hf_api", "hf_manifest", "modelscope"], default="auto")
    parser.add_argument("--grounding-repo", default=GROUNDING_REPO)
    parser.add_argument("--grounding-dir", default=str(default_grounding_dir))
    parser.add_argument("--grounding-provider", choices=["auto", "hf_api", "hf_manifest"], default="auto")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-grounding", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=5.0, help="Warn if target filesystem has less than this much free space before download")
    args = parser.parse_args()

    targets = []
    if not args.skip_qwen:
        targets.append(("qwen", args.qwen_repo, Path(args.qwen_dir), args.qwen_provider))
    if not args.skip_grounding:
        targets.append(("grounding", args.grounding_repo, Path(args.grounding_dir), args.grounding_provider))

    if not targets:
        print("Nothing to do: both downloads are skipped.", file=sys.stderr)
        return 1

    for label, repo_id, local_dir, provider in targets:
        probe_dir = local_dir.parent if local_dir.parent.exists() else root_dir
        free_gb = _disk_free_gb(probe_dir)
        print(f"[download] target={label} repo={repo_id}")
        print(f"[download] local_dir={local_dir}")
        print(f"[download] free_space_gb={free_gb:.2f}")
        if free_gb < args.min_free_gb:
            print(
                f"[warn] free space is below {args.min_free_gb:.2f} GiB. "
                f"Consider using a larger --{'qwen-dir' if label == 'qwen' else 'grounding-dir'} path.",
                file=sys.stderr,
            )

        used_provider = _download_target(label, repo_id, local_dir, args.endpoint, args.cache_dir, provider)
        print(f"[ok] downloaded {label} to {local_dir} via {used_provider}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
