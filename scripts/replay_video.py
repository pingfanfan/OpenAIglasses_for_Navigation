"""Push frames from a local video file to the /ws/camera endpoint.

This helper lets Windows users test the navigation pipeline without any
ESP32-CAM hardware.  Run it alongside ``python app_main.py``.
"""

import argparse
import asyncio
from pathlib import Path

import cv2
import websockets


def _iter_frames(capture: cv2.VideoCapture, *, loop: bool):
    """Yield frames from ``capture`` and optionally loop forever."""

    while True:
        ok, frame = capture.read()
        if ok:
            yield frame
            continue

        if not loop:
            break

        # rewind to the beginning and try again
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)


def _frame_delay(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        return 1.0 / 25.0
    return 1.0 / float(fps)


async def replay_video(video_path: Path, uri: str, *, loop: bool, quality: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频文件: {video_path}")

    delay = _frame_delay(cap)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    print(f"[REPLAY] 连接 {uri}，即将推送 {video_path.name} 的画面 (质量={quality})")
    async with websockets.connect(uri, max_size=2**23) as ws:
        for frame in _iter_frames(cap, loop=loop):
            ok, buffer = cv2.imencode(".jpg", frame, encode_param)
            if not ok:
                continue
            await ws.send(buffer.tobytes())
            await asyncio.sleep(delay)

    cap.release()
    print("[REPLAY] 视频推流结束。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="本地视频文件路径")
    parser.add_argument(
        "--uri",
        default="ws://127.0.0.1:8081/ws/camera",
        help="FastAPI 摄像头 WebSocket 地址",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="播放完成后重新从头循环播放",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG 编码质量 (1-100)，数值越高画质越好，带宽越大",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"视频文件不存在: {video_path}")

    asyncio.run(
        replay_video(
            video_path,
            args.uri,
            loop=args.loop,
            quality=max(1, min(100, args.quality)),
        )
    )


if __name__ == "__main__":
    main()
