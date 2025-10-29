"""Stream local microphone audio into the /ws_audio endpoint.

This helper is meant for Windows (but also works on Linux/macOS) when you
want to drive the navigation system using the PC's own microphone instead of
an ESP32.  It captures 16 kHz mono PCM frames from the default input device
and streams them to the FastAPI backend over WebSocket using the same
protocol as the hardware clients.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import queue
import signal
import sys
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets


SAMPLE_RATE = 16_000
CHANNELS = 1
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # int16


class GracefulExit(SystemExit):
    """Raised when we receive Ctrl+C so the async loop can exit cleanly."""


def install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    def _handler(_sig, _frame):
        raise GracefulExit(0)

    with contextlib.suppress(NotImplementedError):
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture microphone audio and forward it to /ws_audio so the "
            "navigation system can be controlled without extra hardware."
        )
    )
    parser.add_argument(
        "--uri",
        default="ws://127.0.0.1:8081/ws_audio",
        help="WebSocket endpoint of the running FastAPI service.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional sounddevice input device name or index.  Use 'python -m "
            "sounddevice' to list devices."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional logging for troubleshooting.",
    )
    return parser


async def sender(uri: str, frames: "queue.Queue[bytes]", debug: bool) -> None:
    async for ws in websockets.connect(uri, max_size=2 ** 20):
        if debug:
            print(f"[mic-ws] connected to {uri}")
        try:
            await ws.send("START")
            ack = await ws.recv()
            if debug:
                print(f"[mic-ws] server replied: {ack}")
            if not isinstance(ack, str) or not ack.startswith("OK:STARTED"):
                raise RuntimeError(f"Unexpected server reply: {ack!r}")

            while True:
                try:
                    chunk = frames.get(timeout=1.0)
                except queue.Empty:
                    continue
                if not chunk:
                    continue
                await ws.send(chunk)
        except GracefulExit:
            if debug:
                print("[mic-ws] stopping on signal")
            with contextlib.suppress(Exception):
                await ws.send("STOP")
            return
        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"[mic-ws] connection error: {exc}")
        finally:
            with contextlib.suppress(Exception):
                await ws.close()
            if debug:
                print("[mic-ws] websocket closed, retrying in 2s…")
            await asyncio.sleep(2.0)


def open_input_stream(device: Optional[str], frames: "queue.Queue[bytes]", debug: bool):
    def callback(indata, _frames, _time, status):
        if status and debug:
            print(f"[mic-ws] input status: {status}")
        # Ensure we always push int16 bytes.
        if indata.dtype != np.int16:
            data = np.clip(indata * 32767.0, -32768, 32767).astype(np.int16)
        else:
            data = indata.copy()
        try:
            frames.put_nowait(data.tobytes())
        except queue.Full:
            if debug:
                print("[mic-ws] frame queue full, dropping audio")

    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLES_PER_FRAME,
        device=device,
        dtype=np.int16,
        channels=CHANNELS,
        callback=callback,
    )


async def main_async(args) -> None:
    frames: "queue.Queue[bytes]" = queue.Queue(maxsize=10)
    loop = asyncio.get_running_loop()
    install_signal_handlers(loop)

    stream = open_input_stream(args.device, frames, args.debug)
    stream.start()
    try:
        await sender(args.uri, frames, args.debug)
    finally:
        stream.stop()
        stream.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except GracefulExit:
        pass


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
