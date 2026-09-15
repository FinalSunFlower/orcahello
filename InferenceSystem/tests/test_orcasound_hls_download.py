# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for HLS segment download validation."""

from io import BytesIO
from unittest.mock import patch

import pytest

from orcasound_hls.types import OrcasoundHLSSegment


class _Response:
    def __init__(self, payload: bytes, content_length: int | None = None):
        self._stream = BytesIO(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stream.close()

    def read(self, size=-1):
        return self._stream.read(size)


def _segment():
    return OrcasoundHLSSegment(
        bucket="bucket",
        hydrophone_id="hydrophone",
        folder_epoch=1,
        segment_urls=["https://example.test/live000.ts"],
        start_index=0,
        end_index=1,
        start_cum_dur_s=0,
        end_cum_dur_s=10,
    )


def _valid_ts() -> bytes:
    packet = bytes([0x47]) + bytes(187)
    return packet * 3


def test_download_retries_truncated_response(tmp_path):
    payload = _valid_ts()
    responses = [
        _Response(payload[:100], len(payload)),
        _Response(payload, len(payload)),
    ]

    with patch(
        "orcasound_hls.types.urllib.request.urlopen",
        side_effect=responses,
    ) as urlopen:
        filenames = _segment()._download_ts(str(tmp_path))

    assert filenames == ["live000.ts"]
    assert (tmp_path / "live000.ts").read_bytes() == payload
    assert not (tmp_path / "live000.ts.part").exists()
    assert urlopen.call_count == 2


def test_download_rejects_invalid_mpegts_after_retries(tmp_path):
    payload = b"not an MPEG-TS file" * 40

    with patch(
        "orcasound_hls.types.urllib.request.urlopen",
        side_effect=lambda *args, **kwargs: _Response(payload, len(payload)),
    ) as urlopen, pytest.raises(RuntimeError, match="not valid MPEG-TS"):
        _segment()._download_ts(str(tmp_path))

    assert not (tmp_path / "live000.ts").exists()
    assert not (tmp_path / "live000.ts.part").exists()
    assert urlopen.call_count == 3
