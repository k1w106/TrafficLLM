"""DC encoding cho audit label leakage (P5).

DC là chuỗi 500 giá trị packet-count. Paper/upstream đổi mỗi giá trị thành một
chuỗi 32 bit rồi 4 bit -> 1 hex, các hex cách nhau bởi dấu cách. Điểm mấu chốt:

- ``encode_dc_trace_reference`` **nhận nhãn** và xếp các bit 1/0 theo một pattern
  riêng cho từng class. Đây là bản tái hiện hành vi upstream, phục vụ audit;
  không dùng cho pipeline corrected.
- ``encode_dc_trace_corrected`` **không nhận nhãn**. Số bit 1 vẫn là
  ``floor(value / divisor) + 1`` (divisor cấu hình được, mặc định 50 giống code
  upstream; paper dùng 40), nhưng các bit 1 được xếp cố định về bên trái. Nhờ
  vậy loại bỏ leakage mà vẫn giữ cùng bit-budget, hex và tokenization.

Giá trị 0 được giữ thành token ``"0"`` như upstream (không tạo 32-bit pattern).
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_DIVISOR = 50
BIT_WIDTH = 32
HEX_DIGITS = "0123456789abcdef"


def _to_value(raw: float) -> int:
    value = int(round(float(raw)))
    if value < 0:
        raise ValueError(f"DC value phai >= 0, nhan {value}")
    return value


def _bits_to_hex(bits: str) -> str:
    """Đổi 1–4 bit thành 1 ký tự hex, không pad (giống ``bin_to_hex`` upstream)."""
    if not 1 <= len(bits) <= 4:
        raise ValueError(f"can 1-4 bit, nhan {len(bits)}")
    return HEX_DIGITS[int(bits, 2)]


def _reference_pattern(num_digits: int, label: int) -> str:
    """Tái hiện ``create_pattern`` của upstream: run ``1``/``0`` dài ``label+1``.

    Upstream **không cắt** pattern ở 32 bit: nếu pattern dài hơn 32 thì giữ nguyên
    và chỉ pad thêm 0 khi ngắn hơn. Bản ``corrected`` mới cố định 32 bit.
    """
    if label < 0:
        raise ValueError("label phai >= 0")
    pattern = ""
    remaining = int(num_digits)
    while remaining != 0:
        if remaining < label:
            pattern += "1" * remaining
            break
        pattern += "1" * (label + 1)
        pattern += "0" * (label + 1)
        remaining -= label + 1
    return pattern + "0" * max(0, BIT_WIDTH - len(pattern))


def _reference_bits(value: int, label: int, divisor: int) -> str:
    num_digits = value // divisor + 1
    return _reference_pattern(num_digits, label)


def _neutral_bits(value: int, divisor: int) -> str:
    """Pattern label-free: đúng số bit 1, dồn về bên trái."""
    num_ones = max(1, min(BIT_WIDTH, value // divisor + 1))
    return "1" * num_ones + "0" * (BIT_WIDTH - num_ones)


def _encode_values(values: Iterable[float], bit_fn) -> str:
    parts: list[str] = []
    for raw in values:
        value = _to_value(raw)
        if value == 0:
            parts.append("0")
            continue
        bits = bit_fn(value)
        for offset in range(0, len(bits), 4):
            parts.append(_bits_to_hex(bits[offset : offset + 4]))
    return " ".join(parts)


def encode_dc_trace_reference(
    values: Iterable[float], label: int, divisor: int = DEFAULT_DIVISOR
) -> str:
    """Encode DC **có nhãn** — chỉ dùng để tái hiện leakage của upstream.

    Không gọi hàm này trong pipeline corrected. ``label`` là nhãn gốc của trace
    và được dùng để chọn pattern bit, đúng như hành vi audit cần đo.
    """
    if divisor <= 0:
        raise ValueError("divisor phai > 0")
    return _encode_values(values, lambda value: _reference_bits(value, int(label), divisor))


def encode_dc_trace_corrected(
    values: Iterable[float],
    *,
    divisor: int = DEFAULT_DIVISOR,
    pattern: str = "left",
) -> str:
    """Encode DC **không nhận nhãn**; pattern cố định.

    ``pattern`` hiện chỉ hỗ trợ ``"left"``: các bit 1 dồn về bên trái, phần còn
    lại là 0. Số bit 1 vẫn theo ``floor(value / divisor) + 1``.
    """
    if divisor <= 0:
        raise ValueError("divisor phai > 0")
    if pattern != "left":
        raise ValueError(f"pattern khong ho tro: {pattern}")
    return _encode_values(values, lambda value: _neutral_bits(value, divisor))
