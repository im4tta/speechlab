"""Shared PCM <-> WAV helpers. Extracted from api/download.py so services
(dubbing, joining) can use them without importing from an api module."""

from __future__ import annotations

import struct


def pcm16_from_wav(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Strip the WAV container, returning (pcm16_bytes, sample_rate, sample_count).
    Walks chunks so non-standard headers still parse."""
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    sample_rate = 24000
    pcm = b""
    while pos + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[pos:pos + 4]
        chunk_size = int.from_bytes(wav_bytes[pos + 4:pos + 8], "little")
        if chunk_id == b"fmt " and pos + 16 <= len(wav_bytes):
            sample_rate = int.from_bytes(wav_bytes[pos + 12:pos + 16], "little")
        elif chunk_id == b"data":
            pcm = wav_bytes[pos + 8:pos + 8 + chunk_size]
            break
        pos += 8 + chunk_size
        if chunk_size % 2 == 1:
            pos += 1
    if not pcm:
        raise ValueError("WAV has no data chunk")
    return pcm, sample_rate, len(pcm) // 2


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    data_size = len(pcm)
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1, channels,
        sample_rate, byte_rate, block_align, bits, b"data", data_size,
    )
    return header + pcm
