"""OrcasoundHLSSegment — an immutable description of a contiguous audio clip within an HLS stream.

All fields are derived deterministically from S3 folder epoch + M3U8 playlist
metadata.  No I/O happens at construction time; call ``download_as_wav`` /
``download_as_flac`` to materialise audio on disk.
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import ffmpeg
from pytz import timezone as pytz_tz

FOLDER_TO_AUDIO_OFFSET = 2.0

HLS_DOWNLOAD_TIMEOUT_S = 30
HLS_DOWNLOAD_RETRIES = 3
MPEGTS_PACKET_SIZE = 188


@dataclass(frozen=True)
class OrcasoundHLSSegment:
    """Immutable metadata for a contiguous range of HLS .ts segments."""

    # --- location in S3 ---
    bucket: str
    hydrophone_id: str
    folder_epoch: int
    segment_urls: List[str] = field(repr=False)

    # --- position within the M3U8 playlist ---
    start_index: int
    end_index: int  # exclusive

    # --- cumulative durations from M3U8 playlist (relative to folder epoch) ---
    start_cum_dur_s: float
    end_cum_dur_s: float

    # --- audio offset (seconds after folder epoch before audio in M3U8 playlist actually starts) ---
    # approximate calibration constant that may vary between hydrophones or stream conditions
    folder_to_audio_offset_s: float = FOLDER_TO_AUDIO_OFFSET

    # --- convenience ---
    @property
    def duration_s(self) -> float:
        return self.end_cum_dur_s - self.start_cum_dur_s

    @property
    def start_unix(self) -> float:
        return self.folder_epoch + self.start_cum_dur_s + self.folder_to_audio_offset_s

    @property
    def end_unix(self) -> float:
        return self.folder_epoch + self.end_cum_dur_s + self.folder_to_audio_offset_s

    @property
    def start_utc(self) -> datetime:
        return datetime.fromtimestamp(self.start_unix, tz=timezone.utc)

    @property
    def end_utc(self) -> datetime:
        return datetime.fromtimestamp(self.end_unix, tz=timezone.utc)

    @property
    def start_iso(self) -> str:
        """ISO-8601 UTC timestamp, e.g. ``2020-09-01T22:13:02Z``."""
        return self.start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def name(self) -> str:
        """Human-readable segment name in Pacific time (matches legacy naming)."""
        pst = self.start_utc.astimezone(pytz_tz("US/Pacific"))
        slug = self.hydrophone_id.replace("_", "-")
        return slug + "_" + pst.strftime("%Y_%m_%d_%H_%M_%S_%Z")

    # --- I/O: download and convert ---

    def _download_ts(self, dest_dir: str) -> List[str]:
        """Download .ts segment files into *dest_dir*.  Returns local filenames."""
        filenames: List[str] = []
        for url in self.segment_urls:
            fname = os.path.basename(url)
            dest = os.path.join(dest_dir, fname)
            if not os.path.isfile(dest):
                part = dest + ".part"
                last_error: Exception | None = None
                for attempt in range(1, HLS_DOWNLOAD_RETRIES + 1):
                    try:
                        with urllib.request.urlopen(
                            url, timeout=HLS_DOWNLOAD_TIMEOUT_S
                        ) as resp, open(part, "wb") as out:
                            expected_size = resp.headers.get("Content-Length")
                            expected_size = (
                                int(expected_size) if expected_size else None
                            )
                            size = 0
                            while chunk := resp.read(1024 * 1024):
                                out.write(chunk)
                                size += len(chunk)
                            out.flush()

                        if expected_size is not None and size != expected_size:
                            raise IOError(
                                f"incomplete download ({size} of {expected_size} bytes)"
                            )
                        with open(part, "rb") as check:
                            header = check.read(MPEGTS_PACKET_SIZE * 3)
                        if (
                            not header
                            or header[0] != 0x47
                            or any(
                                header[offset] != 0x47
                                for offset in range(
                                    MPEGTS_PACKET_SIZE,
                                    len(header),
                                    MPEGTS_PACKET_SIZE,
                                )
                            )
                        ):
                            raise IOError("downloaded file is not valid MPEG-TS")
                        os.replace(part, dest)
                        break
                    except (OSError, urllib.error.URLError, ValueError) as exc:
                        last_error = exc
                        try:
                            os.remove(part)
                        except FileNotFoundError:
                            pass
                        if attempt < HLS_DOWNLOAD_RETRIES:
                            time.sleep(0.5 * attempt)
                else:
                    raise RuntimeError(
                        f"failed to download complete HLS segment {url}: {last_error}"
                    ) from last_error
            filenames.append(fname)
        return filenames

    def _concat_and_convert(
        self, ts_dir: str, filenames: List[str], out_path: str
    ) -> str:
        """Concatenate .ts files then convert with ffmpeg to *out_path*."""
        concat_path = os.path.join(ts_dir, self.name + ".ts")
        with open(concat_path, "wb") as out:
            for fname in filenames:
                with open(os.path.join(ts_dir, fname), "rb") as inp:
                    shutil.copyfileobj(inp, out)

        stream = ffmpeg.input(
            concat_path,
            f="mpegts",
            err_detect="ignore_err",
            fflags="+genpts",
        )
        stream = ffmpeg.output(stream, out_path)
        try:
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
        except ffmpeg.Error as exc:
            stderr = (exc.stderr or b"").decode(errors="replace").strip()
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"ffmpeg conversion failed{detail}") from exc
        if not os.path.isfile(out_path) or os.path.getsize(out_path) <= 44:
            raise RuntimeError("ffmpeg conversion produced an empty output file")
        return out_path

    def download_as_wav(self, dest_dir: str) -> str:
        """Download segments, convert to WAV, return the output path."""
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        wav_path = os.path.join(dest_dir, self.name + ".wav")
        with TemporaryDirectory() as tmp:
            filenames = self._download_ts(tmp)
            self._concat_and_convert(tmp, filenames, wav_path)
        return wav_path

    def download_as_flac(self, dest_dir: str) -> str:
        """Download segments, convert to FLAC, return the output path."""
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        flac_path = os.path.join(dest_dir, self.name + ".flac")
        with TemporaryDirectory() as tmp:
            filenames = self._download_ts(tmp)
            self._concat_and_convert(tmp, filenames, flac_path)
        return flac_path
