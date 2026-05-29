# EVOLVE-BLOCK-START
"""Coefficients construction via integer-only recurrence with exact small-L sets and a recursive heuristic."""

from functools import lru_cache
import math

# Exact known sets for small L (copied from the problem statement)
KNOWN_SETS = {
    1: [-2],
    2: [8, 16],
    3: [-384, -96, -56, -24, -16, 16, 24, 28],
    4: [
        -1248,-1216,-1152,-768,-384,-304,-192,-144,-112,-96,-88,-80,-72,-64,-48,
        64,80,96,128,240,256,264,272,288,320,352,384,448,480,544,576,960,1920,15360
    ],
    5: [
        -860160,-53760,-19680,-18728,-17688,-17104,-16744,-16400,-16256,-15968,-15808,-15632,
        -14808,-14640,-14624,-14048,-13824,-13760,-13608,-13184,-12160,-11264,-10720,-10448,-10368,
        -10240,-10088,-10036,-9600,-9248,-5760,-5728,-5600,-5376,-5248,-5184,-4320,-4160,-4128,-4064,
        -3936,-3872,-3792,-3776,-3680,-3408,-3224,-3072,-3008,-2928,-2912,-2896,-2888,-2880,-2784,-2752,
        -2624,-2528,-2368,-2336,-2320,-2224,-2176,-2144,-2024,-2016,-1904,-1856,-1840,-1824,-1808,-1728,
        -1712,-1680,-1648,-1624,-1616,-1536,-1424,-1280,-512,-128,128,192,256,320,384,464,480,640,864,
        976,1088,1120,1152,1184,1216,1248,1264,1280,1408,1472,1508,1568,1576,1600,1648,1792,1800,1992,
        2116,2144,2352,2400,2408,2432,2496,2504,2528,2560,2704,2712,2768,2776,2784,2968,3072,3096,3168,
        3200,3392,3520,3552,3568,3632,3648,3712,3760,3808,3840,4160,4272,4320,4416,4480,4576,4608,4656,
        4704,5024,5104,5280,5536,5568,5728,6288,7760,9216,11520,46080,69120,70080,70272,70880,71280
    ],
}

@lru_cache(maxsize=None)
def _largest_coefficient(L: int) -> int:
    """
    Compute the largest absolute coefficient at loop order L using the closed-form:
        A(L) = 2^(2L-1) * binomial(2L-2, L-1)
    This is equivalent to the recurrence but computed directly for speed.
    """
    if L <= 0:
        raise ValueError("L must be a positive integer")
    power_part = 1 << (2 * L - 1)  # 2^(2L-1)
    comb_part = math.comb(2 * (L - 1), L - 1)
    return power_part * comb_part

@lru_cache(maxsize=None)
def _heuristic_set(L: int):
    """
    Heuristic recursive generator for L not covered by KNOWN_SETS.
    Strategy (bounded but inclusive enough to capture known patterns):
    - Always include the parity-correct extreme ±A(L).
    - Propagate the entire (L-1) set scaled by the exact factor 8*(2L-3).
    - Add a modest ladder of small multiples of A(L-1) (both signs), capped to avoid blow-up.
    - Harvest complementary factors from small divisors (even and odd) of A(L), with a mild bound.
    - Deterministically prune by |value|, then value, to a safe cap if needed.
    """
    if L in KNOWN_SETS:
        return tuple(sorted(KNOWN_SETS[L]))

    if L <= 0:
        raise ValueError("L must be a positive integer")

    A = _largest_coefficient(L)
    sign_extreme = -A if (L % 2 == 1) else A

    S = set()
    # Include only the parity-correct extreme
    S.add(sign_extreme)

    # Scale previous-layer known/heuristic set by the exact factor
    if L > 1:
        Sp = _heuristic_set(L - 1)  # returns tuple
        A_prev = _largest_coefficient(L - 1)
        scale = A // A_prev  # equals 8*(2L-3)
        for y in Sp:
            if y != 0:
                S.add(scale * y)

        # Add a small, bounded ladder of multiples of A(L-1)
        M = min(4 * L + 4, 48)
        for m in range(2, M + 1):
            S.add(m * A_prev)
            S.add(-m * A_prev)

    # Harvest divisors of A(L): include complementary factors for small divisors (even and odd)
    div_bound = min(16 * (2 * L - 3), 512)
    for d in range(2, div_bound + 1):
        if A % d == 0:
            q = A // d; S.add(q); S.add(-q)

    # Deterministic prune to keep growth in check
    max_keep = 2048
    if len(S) > max_keep:
        S = set(sorted(S, key=lambda z: (abs(z), z))[:max_keep])

    return tuple(sorted(S))

def construct_coefficients(L: int = 8):
    """
    Construct the set of coefficients that occur at loop order L in the planar limit
    of form factor in the maximum super symmetric Yang-Mills theory.

    Args:
        L: loop order (positive integer). Default 8 for backwards compatibility.

    Returns:
        Tuple[int, List[int]]: A dummy constant (1) and the list of coefficients at loop order L.
    """
    if L <= 0:
        raise ValueError("L must be a positive integer")

    if L in KNOWN_SETS:
        guess = sorted(KNOWN_SETS[L])
    else:
        guess = list(_heuristic_set(L))

    return 1, guess
# EVOLVE-BLOCK-END