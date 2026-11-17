"""Validate cosine-search vectors at their persisted float32 precision."""

import math
import struct
from collections.abc import Sequence


def float32_vector(values: Sequence[float], dimensions: int) -> list[float]:
    """Return representable values, rejecting unusable vectors without their content.

    Float64 values that underflow together become a zero vector in pgvector and
    cannot support cosine distance. Individual underflow is acceptable if some
    representable component remains nonzero. Do not normalize or drop chunks.
    """
    try:
        if isinstance(values, (str, bytes, bytearray)) or len(values) != dimensions:
            raise ValueError
        converted = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError
            converted.append(struct.unpack("!f", struct.pack("!f", value))[0])
        if not all(math.isfinite(value) for value in converted) or not any(converted):
            raise ValueError
    except (TypeError, ValueError, OverflowError, struct.error):
        raise ValueError(
            "Vector must have finite nonzero float32 values and valid dimensions"
        ) from None
    return converted
