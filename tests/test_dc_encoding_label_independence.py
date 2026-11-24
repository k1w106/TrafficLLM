"""Khoá encoder DC: bản corrected không nhận nhãn, bản reference thì có."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from trafficllm.encoding.dc import (
    encode_dc_trace_corrected,
    encode_dc_trace_reference,
)


def test_corrected_encoder_has_no_label_parameter() -> None:
    signature = inspect.signature(encode_dc_trace_corrected)
    assert list(signature.parameters) == ["values", "divisor", "pattern"]
    assert "label" not in signature.parameters


def test_corrected_encoder_body_never_mentions_label() -> None:
    source_path = Path(inspect.getsourcefile(encode_dc_trace_corrected))
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "encode_dc_trace_corrected"
    )
    segment = ast.get_source_segment(source, function) or ""
    assert "label" not in segment.lower()


def test_corrected_encoding_is_label_independent() -> None:
    values = [0, 100, 500, 736]
    assert encode_dc_trace_corrected(values) == encode_dc_trace_corrected(values)


def test_reference_encoding_depends_on_label() -> None:
    values = [500]
    first = encode_dc_trace_reference(values, label=1)
    second = encode_dc_trace_reference(values, label=5)
    assert first != second


def test_corrected_pattern_keeps_number_of_ones() -> None:
    encoded = encode_dc_trace_corrected([100])
    chunks = encoded.split()
    assert len(chunks) == 8
    bits = "".join(format(int(chunk, 16), "04b") for chunk in chunks)
    assert bits.count("1") == 100 // 50 + 1
    assert bits.startswith("1110")


def test_zero_is_a_sentinel_token() -> None:
    assert encode_dc_trace_corrected([0]) == "0"
    assert encode_dc_trace_reference([0], label=3) == "0"


def test_output_alphabet_is_hex_digits_and_spaces() -> None:
    encoded = encode_dc_trace_corrected([0, 1, 100, 736])
    assert set(encoded) <= set("0123456789abcdef ")


def test_divisor_changes_bit_budget() -> None:
    default = encode_dc_trace_corrected([500], divisor=50)
    paper = encode_dc_trace_corrected([500], divisor=40)
    assert default != paper


def test_reference_handles_partial_last_hex_chunk() -> None:
    """Upstream đổi cả chunk 2 bit cuối; encoder không được đòi đúng 4 bit."""
    # 15 digit -> pattern "10"*15 = 30 bit -> chunk cuối dài 2 bit.
    encoded = encode_dc_trace_reference([700], label=0)
    assert len(encoded.split()) == 8


def test_reference_pattern_is_not_truncated_at_32_bits() -> None:
    """Upstream chỉ pad thêm 0, không cắt pattern dài hơn 32 bit."""
    # 41 digit -> pattern "10"*41 = 82 bit -> 21 hex chunk.
    encoded = encode_dc_trace_reference([2000], label=0)
    assert len(encoded.split()) == 21


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        encode_dc_trace_corrected([10], divisor=0)
    with pytest.raises(ValueError):
        encode_dc_trace_corrected([10], pattern="unknown")
    with pytest.raises(ValueError):
        encode_dc_trace_corrected([-1])
