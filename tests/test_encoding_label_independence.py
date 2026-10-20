"""Khoá chữ ký của encoder: hàm mã hoá không được nhận nhãn làm đầu vào."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from trafficllm.encoding.ustc import encode_trace_gpt2

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_PATH = REPO_ROOT / "data" / "processed" / "ustc_seed42" / "traces.npy"


def test_encoder_accepts_exactly_one_positional_parameter() -> None:
    signature = inspect.signature(encode_trace_gpt2)
    assert list(signature.parameters) == ["raw_trace"]
    parameter = signature.parameters["raw_trace"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty


def test_encoder_rejects_an_extra_argument() -> None:
    with pytest.raises(TypeError):
        encode_trace_gpt2("e742 421d", 3)  # type: ignore[call-arg]


def test_encoder_body_never_mentions_a_class_identifier() -> None:
    source_path = Path(inspect.getsourcefile(encode_trace_gpt2))
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "encode_trace_gpt2"
    )
    segment = ast.get_source_segment(source, function) or ""
    assert "label" not in segment.lower()


def test_encoding_matches_the_upstream_character_level_format() -> None:
    assert encode_trace_gpt2("e742 421d 1dc8") == "e 7 4 2 4 2 1 d 1 d c 8"


def test_encoding_ignores_extra_whitespace() -> None:
    assert encode_trace_gpt2("  e742   421d ") == encode_trace_gpt2("e742 421d")


def test_output_alphabet_is_hex_digits_and_spaces() -> None:
    encoded = encode_trace_gpt2("e742 421d 1dc8")
    assert set(encoded) <= set("0123456789abcdef ")


def test_encoder_is_deterministic() -> None:
    raw = "e742 421d 1dc8 c868"
    assert encode_trace_gpt2(raw) == encode_trace_gpt2(raw)


@pytest.mark.skipif(not TRACES_PATH.exists(), reason="chưa có traces.npy trong data/processed")
def test_real_trace_round_trip() -> None:
    traces = np.load(TRACES_PATH, allow_pickle=True)
    raw = str(traces[0])
    encoded = encode_trace_gpt2(raw)
    assert encoded.replace(" ", "") == raw.replace(" ", "")
    assert encoded.count(" ") == len(raw.replace(" ", "")) - 1
