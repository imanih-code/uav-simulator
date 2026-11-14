"""Forward error correction: the classic K=7, rate-1/2 convolutional code
(generators 171/133 octal, the "NASA standard" used from Voyager to
802.11a) with a soft-decision Viterbi decoder.

Soft decisions matter: the decoder is fed the despread correlation values
(how *confident* each bit is), not just their signs, which is worth ~2 dB.
"""
from __future__ import annotations

import numpy as np

K = 7
N_STATES = 1 << (K - 1)          # 64
GENERATORS = (0o171, 0o133)
TAIL_BITS = K - 1


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


# reg = (u << 6) | state  (newest input in the MSB); next state = reg >> 1
_OUT = np.zeros((N_STATES, 2, 2), dtype=np.float64)   # [state, u] -> two ±1 symbols
for _s in range(N_STATES):
    for _u in (0, 1):
        _reg = (_u << (K - 1)) | _s
        _OUT[_s, _u] = [2.0 * _parity(_reg & g) - 1.0 for g in GENERATORS]

# For each next state: its two possible predecessors and the symbols on that branch.
_NEXT_U = np.arange(N_STATES) >> (K - 2)                       # input bit that leads here
_PRED = np.stack([((np.arange(N_STATES) & (N_STATES // 2 - 1)) << 1) | b for b in (0, 1)], axis=1)
_PRED_OUT = np.stack([_OUT[_PRED[:, b], _NEXT_U] for b in (0, 1)], axis=1)  # [ns, b, 2]


def encode(bits: np.ndarray) -> np.ndarray:
    """bits (0/1) -> coded bits (0/1), length 2 * (len(bits) + 6), zero-terminated."""
    state = 0
    out = np.empty(2 * (len(bits) + TAIL_BITS), dtype=np.uint8)
    for i, u in enumerate(np.concatenate([np.asarray(bits, dtype=np.uint8),
                                          np.zeros(TAIL_BITS, dtype=np.uint8)])):
        reg = (int(u) << (K - 1)) | state
        out[2 * i] = _parity(reg & GENERATORS[0])
        out[2 * i + 1] = _parity(reg & GENERATORS[1])
        state = reg >> 1
    return out


def decode(soft: np.ndarray) -> np.ndarray:
    """Soft values (positive = bit 1, magnitude = confidence), length 2*(n+6)
    -> the n most likely information bits (0/1)."""
    soft = np.asarray(soft, dtype=np.float64).reshape(-1, 2)
    steps = len(soft)
    # Branch metrics for every step at once: [t, next_state, predecessor b]
    branch = np.einsum("nbk,tk->tnb", _PRED_OUT, soft)
    branch0 = np.ascontiguousarray(branch[:, :, 0])
    branch1 = np.ascontiguousarray(branch[:, :, 1])
    pred0, pred1 = _PRED[:, 0], _PRED[:, 1]
    metric = np.full(N_STATES, -1e18)
    metric[0] = 0.0
    decisions = np.empty((steps, N_STATES), dtype=bool)
    for t in range(steps):
        c0 = metric[pred0] + branch0[t]
        c1 = metric[pred1] + branch1[t]
        take1 = c1 > c0
        decisions[t] = take1
        metric = np.where(take1, c1, c0)
    state = 0  # zero-terminated by the tail bits
    bits = np.empty(steps, dtype=np.uint8)
    for t in range(steps - 1, -1, -1):
        bits[t] = _NEXT_U[state]
        state = int(_PRED[state, int(decisions[t, state])])
    return bits[: steps - TAIL_BITS]
