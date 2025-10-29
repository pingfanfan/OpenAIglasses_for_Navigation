"""Quick environment sanity check for the desktop demo workflow.

This script focuses on the "single Windows PC + recorded video + local
microphone" setup so that users can spot the most common blockers before
launching ``python app_main.py``.  It does **not** modify the environment – it
only inspects files/modules and prints friendly guidance.

Usage (from the repository root)::

    python scripts/preflight_check.py

Exit status ``0`` means everything essential is ready.  A non-zero exit code
indicates that at least one required item is missing.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    """Represents the outcome of a single check."""

    label: str
    status: str  # "ok", "warn", or "error"
    detail: str
    hint: Optional[str] = None

    def is_error(self) -> bool:
        return self.status == "error"

    def is_warning(self) -> bool:
        return self.status == "warn"


def read_env_var(name: str) -> Optional[str]:
    """Return ``name`` from the real env or .env if present."""

    if value := os.getenv(name):
        return value

    env_path = ROOT / ".env"
    if not env_path.exists():
        return None

    try:
        with env_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, _, val = raw.partition("=")
                if key.strip() == name:
                    return val.strip().strip('"').strip("'")
    except OSError:
        return None

    return None


def check_dashscope_key() -> CheckResult:
    value = read_env_var("DASHSCOPE_API_KEY")
    if value and len(value) > 10:
        obscured = value[:6] + "…" + value[-4:]
        return CheckResult("DashScope API Key", "ok", f"已配置（{obscured}）")
    return CheckResult(
        "DashScope API Key",
        "error",
        "未在环境变量或 .env 中找到 DASHSCOPE_API_KEY。",
        "复制 .env.example 为 .env 并填入从 DashScope 控制台获取的密钥。",
    )


def check_model_files() -> Iterable[CheckResult]:
    model_dir = ROOT / "model"
    required = [
        ("盲道分割模型", "yolo-seg.pt", True),
        ("YOLO-E 障碍物模型", "yoloe-11l-seg.pt", False),
        ("寻物模型 (shoppingbest5)", "shoppingbest5.pt", False),
        ("手部检测 task", "hand_landmarker.task", False),
    ]

    if not model_dir.exists():
        yield CheckResult(
            "模型目录",
            "error",
            f"未找到 {model_dir}。",
            "在仓库根目录创建 model/ 并放入 README 列出的模型文件。",
        )
        return

    for label, filename, required_flag in required:
        path = model_dir / filename
        if path.exists():
            yield CheckResult(label, "ok", f"已找到 {filename}")
        else:
            status = "error" if required_flag else "warn"
            hint = "请下载后放到 model/ 目录。" if required_flag else "缺失将禁用对应功能，可稍后再补。"
            yield CheckResult(
                label,
                status,
                f"缺少 {filename}",
                hint,
            )


def check_voice_assets() -> Iterable[CheckResult]:
    voice_dir = Path(os.getenv("VOICE_DIR", ROOT / "voice"))
    map_file = voice_dir / "map.zh-CN.json"

    if not voice_dir.exists():
        yield CheckResult(
            "语音播报资源目录",
            "warn",
            f"VOICE_DIR={voice_dir} 不存在。",
            "如无需语音播报，可在 .env 中设置 ENABLE_TTS=false；否则请保持 voice/ 目录完整。",
        )
        return

    if map_file.exists():
        yield CheckResult("语音映射表", "ok", "已检测到 voice/map.zh-CN.json")
    else:
        yield CheckResult(
            "语音映射表",
            "warn",
            "缺少 voice/map.zh-CN.json。",
            "语音播报仍可运行，但将回退到旧版 music/ 音频映射。",
        )


def check_python_module(label: str, module: str, hint: str) -> CheckResult:
    try:
        importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - diagnostics only
        return CheckResult(label, "error", f"导入失败: {exc}", hint)
    return CheckResult(label, "ok", "模块可用")


def check_python_environment() -> Iterable[CheckResult]:
    modules = [
        ("PyTorch", "torch", "请参考 README 中的 Windows 安装指引单独安装 PyTorch。"),
        ("Ultralytics", "ultralytics", "执行 pip install ultralytics==8.3.200"),
        ("OpenCV", "cv2", "执行 pip install opencv-python==4.8.1.78"),
        ("WebSockets", "websockets", "执行 pip install websockets==12.0"),
        ("SoundDevice", "sounddevice", "执行 pip install sounddevice==0.4.6"),
    ]
    for label, module, hint in modules:
        yield check_python_module(label, module, hint)

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            yield CheckResult("CUDA", "ok", f"检测到 GPU: {device_name}")
        else:
            yield CheckResult(
                "CUDA",
                "warn",
                "未检测到可用的 CUDA 设备，将以 CPU 模式运行。",
                "若已安装 NVIDIA 驱动与 CUDA Toolkit，请确认 PyTorch 安装了对应的 +cu118 版本。",
            )
    except Exception as exc:  # pragma: no cover - optional diagnostics
        yield CheckResult("CUDA", "warn", f"无法检测 CUDA: {exc}")


def check_scripts_present() -> Iterable[CheckResult]:
    for label, rel_path in (
        ("视频回放脚本", "scripts/replay_video.py"),
        ("麦克风推流脚本", "scripts/mic_ws_client.py"),
    ):
        path = ROOT / rel_path
        if path.exists():
            yield CheckResult(label, "ok", f"已找到 {rel_path}")
        else:
            yield CheckResult(label, "error", f"缺少 {rel_path}", "请重新拉取仓库或复制示例脚本。")


def format_result(result: CheckResult) -> str:
    status_map = {
        "ok": "[ OK ]",
        "warn": "[WARN]",
        "error": "[FAIL]",
    }
    head = status_map.get(result.status, "[ ???? ]")
    lines = [f"{head} {result.label}: {result.detail}"]
    if result.hint:
        lines.append(f"       ↳ 建议: {result.hint}")
    return "\n".join(lines)


def run_checks() -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(check_dashscope_key())
    results.extend(check_model_files())
    results.extend(check_voice_assets())
    results.extend(check_python_environment())
    results.extend(check_scripts_present())
    return results


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="验证 Windows 单机体验所需的常见依赖是否就绪。"
    )
    parser.add_argument(
        "--no-exit-code",
        action="store_true",
        help="始终返回 0，方便在批处理脚本中仅查看结果。",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("== AI Glasses Preflight Check ==")
    print(f"仓库根目录: {ROOT}\n")

    results = run_checks()
    errors = sum(1 for item in results if item.is_error())
    warnings = sum(1 for item in results if item.is_warning())

    for item in results:
        print(format_result(item))

    print("\n总结: {} 个错误, {} 个警告".format(errors, warnings))

    if errors and not args.no_exit_code:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
