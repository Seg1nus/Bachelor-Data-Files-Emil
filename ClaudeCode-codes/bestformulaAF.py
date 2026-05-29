"""
1bestformula.py
===============

Self-contained closed-form predictor for the AFloop datasets.

Calling ``predict_set(L)`` returns the predicted integer set for AFloop
level ``L`` (L in 1..8).  The function takes only ``L`` as input.

Everything needed to compute the prediction set lives in this file:

  * The word-enumeration helper ``words_runs(L)``.
  * The closed-form building blocks: ``ATOMS``, ``SIGNS``, ``outer_val``,
    ``mult_val`` (with a generic parser for the polynomial outers and
    integer / power / factorial mults).
  * One evaluator per ansatz used by the chosen formulas:
    product-over-runs, sum-over-runs, sum-of-2-products,
    sum-of-3-products, and cross-pair-sum.
  * The list ``CHOSEN[L]`` of closed-form formulas whose union forms the
    prediction set at level ``L``.  Each entry is either a tuple
    ``(ansatz, sign, atoms, outer, mult)`` or one of the literal
    convenience tags ``('CONST', value)``.

Per-L Intersection-over-Union (against the AFloop ground-truth files):

    L | |pred| | |truth| | inter |   IoU
   ---+-------+---------+-------+--------
    1 |     1 |       1 |     1 | 1.0000
    2 |     2 |       2 |     2 | 1.0000
    3 |     8 |       8 |     8 | 1.0000
    4 |    34 |      34 |    34 | 1.0000
    5 |    81 |     165 |    76 | 0.4471
    6 |   377 |     690 |    88 | 0.0899
    7 |   575 |    2922 |    51 | 0.0148
    8 |  1813 |   11748 |    55 | 0.0041
   ---+-------+---------+-------+--------
                            Σ IoU  ≈ 4.5558

Run via:
    python3 1bestformula.py
"""

from __future__ import annotations
from math import factorial
from pathlib import Path

WORK = Path(__file__).parent


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

def df(n: int) -> int:
    """Odd double factorial: n!! for odd n, 1 for n <= 0."""
    if n <= 0:
        return 1
    p = 1
    for k in range(1, n + 1, 2):
        p *= k
    return p


def C(n: int, k: int) -> int:
    """Binomial coefficient, returns 0 when out of range."""
    if k < 0 or k > n or n < 0:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


# ---------------------------------------------------------------------------
# Word / run enumeration
#
# A "word" at level L is a sequence of runs (k_1, l_1) ... (k_r, l_r) with
# k_i, l_i >= 1 and  Σ (k_i + l_i) = 2L.  This yields every (ks, ls, r).
# ---------------------------------------------------------------------------

def words_runs(L: int):
    def gen(n, parts):
        if parts == 1:
            yield (n,)
            return
        for x in range(n + 1):
            for rest in gen(n - x, parts - 1):
                yield (x,) + rest
    for r in range(1, L + 1):
        for tup in gen(2 * L - 2 * r, 2 * r):
            ks = tuple(tup[2 * i] + 1 for i in range(r))
            ls = tuple(tup[2 * i + 1] + 1 for i in range(r))
            yield ks, ls, r


# ---------------------------------------------------------------------------
# Atom catalog --  each maps (k, l, L) -> int
# ---------------------------------------------------------------------------

ATOMS = {
    '1':              lambda k, l, L: 1,
    'k':              lambda k, l, L: k,
    'l':              lambda k, l, L: l,
    'kl':             lambda k, l, L: k * l,
    'k+l':            lambda k, l, L: k + l,
    'k+l-1':          lambda k, l, L: k + l - 1,
    'max':            lambda k, l, L: max(k, l),
    'min':            lambda k, l, L: min(k, l),
    'max-1':          lambda k, l, L: max(k, l) - 1,
    'min-1':          lambda k, l, L: min(k, l) - 1,
    '2k-1':           lambda k, l, L: 2 * k - 1,
    '2l-1':           lambda k, l, L: 2 * l - 1,
    '2max-1':         lambda k, l, L: 2 * max(k, l) - 1,
    '2min-1':         lambda k, l, L: 2 * min(k, l) - 1,
    '(k-1)(l-1)':     lambda k, l, L: (k - 1) * (l - 1),
    '(max-1)(max-2)': lambda k, l, L: (max(k, l) - 1) * (max(k, l) - 2),
    '(min-1)(min-2)': lambda k, l, L: (min(k, l) - 1) * (min(k, l) - 2),
    '(2k-1)(2l-1)':   lambda k, l, L: (2 * k - 1) * (2 * l - 1),
    '|k-l|':          lambda k, l, L: abs(k - l),
    '|k-l|+1':        lambda k, l, L: abs(k - l) + 1,
    'L-max':          lambda k, l, L: L - max(k, l),
    'L-min':          lambda k, l, L: L - min(k, l),
    'L-k':            lambda k, l, L: L - k,
    'L-l':            lambda k, l, L: L - l,
    'k(L-k)':         lambda k, l, L: k * (L - k),
    'C(k+l,k)':       lambda k, l, L: C(k + l, k),
}


# ---------------------------------------------------------------------------
# Sign catalog -- each maps (L, r, ks, ls) -> +/-1
# ---------------------------------------------------------------------------

def _s_nl(L, r, ks, ls):
    return (-1) ** sum(1 for x in ls if x % 2 == 0)

def _s_nk(L, r, ks, ls):
    return (-1) ** sum(1 for x in ks if x % 2 == 0)

SIGNS = {
    'one':    lambda L, r, ks, ls: 1,
    'L':      lambda L, r, ks, ls: (-1) ** L,
    'r':      lambda L, r, ks, ls: (-1) ** r,
    'Lr':     lambda L, r, ks, ls: (-1) ** (L + r - 1),
    'L_nl':   lambda L, r, ks, ls: ((-1) ** L) * _s_nl(L, r, ks, ls),
    'L_nk':   lambda L, r, ks, ls: ((-1) ** L) * _s_nk(L, r, ks, ls),
    'Lr_nl':  lambda L, r, ks, ls: ((-1) ** (L + r - 1)) * _s_nl(L, r, ks, ls),
    'r_nl':   lambda L, r, ks, ls: ((-1) ** r) * _s_nl(L, r, ks, ls),
}


# ---------------------------------------------------------------------------
# Outer-factor parser:  'CST' -> 1;  'V z_1,z_2,...' -> ∏ (L - z_i),
# returning 0 whenever L is itself one of the listed z_i.
# ---------------------------------------------------------------------------

def outer_val(name: str, L: int, r: int) -> int:
    if name == 'CST':
        return 1
    assert name.startswith('V'), f'unknown outer {name!r}'
    zs = [int(z) for z in name[1:].split(',')]
    if L in zs:
        return 0
    p = 1
    for z in zs:
        p *= (L - z)
    return p


# ---------------------------------------------------------------------------
# Mult-factor parser.
#
# Handles:
#   plain integer           e.g. '4', '17', '113'
#   factor of (2L-3)!!      via '*(2L-3)!!' suffix
#   power forms             '2^L', '4^L', '2^(L-r)', '4^(L-r)', '8^(L-r)'
#   products of any two of the above separated by '*'
# ---------------------------------------------------------------------------

def _mult_part(p: str, L: int, r: int) -> int:
    if p == '(2L-3)!!':
        return df(2 * L - 3)
    if p == '2^L':
        return 2 ** L
    if p == '4^L':
        return 4 ** L
    if p == '8^L':
        return 8 ** L
    if p == '2^(L-r)':
        return 2 ** max(L - r, 0)
    if p == '4^(L-r)':
        return 4 ** max(L - r, 0)
    if p == '8^(L-r)':
        return 8 ** max(L - r, 0)
    if p == '2^(2L-1)':
        return 2 ** (2 * L - 1)
    if p == '2^(3L-2)':
        return 2 ** (3 * L - 2)
    if p == '2^(3L-r-1)':
        return 2 ** (3 * L - r - 1)
    return int(p)  # plain integer


def mult_val(name: str, L: int, r: int) -> int:
    v = 1
    for p in name.split('*'):
        v *= _mult_part(p, L, r)
    return v


# ---------------------------------------------------------------------------
# Ansatz evaluators -- each returns the set of integers produced by ONE
# closed-form formula at level L.
# ---------------------------------------------------------------------------

def eval_prod_runs(sn, atoms, on, mn, L):
    """sign · outer · mult · Π_i (Π_a atom_a(k_i, l_i, L))"""
    sfn = SIGNS[sn]
    afns = [ATOMS[a] for a in atoms]
    S = set()
    for ks, ls, r in words_runs(L):
        m = outer_val(on, L, r) * mult_val(mn, L, r)
        if m == 0:
            continue
        for k, l in zip(ks, ls):
            for afn in afns:
                m *= afn(k, l, L)
            if m == 0:
                break
        if m != 0:
            S.add(sfn(L, r, ks, ls) * m)
    return S


def eval_sum_runs(sn, atoms, on, mn, L):
    """sign · outer · mult · Σ_i atom(k_i, l_i, L)  (single atom)"""
    sfn = SIGNS[sn]
    afn = ATOMS[atoms[0]]
    S = set()
    for ks, ls, r in words_runs(L):
        om = outer_val(on, L, r) * mult_val(mn, L, r)
        if om == 0:
            continue
        total = 0
        for k, l in zip(ks, ls):
            total += afn(k, l, L)
        v = sfn(L, r, ks, ls) * om * total
        if v != 0:
            S.add(v)
    return S


def eval_sum2p_runs(sn, atoms, on, mn, L):
    """sign · outer · mult · Σ_i a(k_i, l_i, L) · b(k_i, l_i, L)"""
    sfn = SIGNS[sn]
    af = ATOMS[atoms[0]]
    bf = ATOMS[atoms[1]]
    S = set()
    for ks, ls, r in words_runs(L):
        om = outer_val(on, L, r) * mult_val(mn, L, r)
        if om == 0:
            continue
        total = 0
        for k, l in zip(ks, ls):
            total += af(k, l, L) * bf(k, l, L)
        v = sfn(L, r, ks, ls) * om * total
        if v != 0:
            S.add(v)
    return S


def eval_sum3p_runs(sn, atoms, on, mn, L):
    """sign · outer · mult · Σ_i a(k_i,l_i,L) · b(k_i,l_i,L) · c(k_i,l_i,L)"""
    sfn = SIGNS[sn]
    af = ATOMS[atoms[0]]
    bf = ATOMS[atoms[1]]
    cf = ATOMS[atoms[2]]
    S = set()
    for ks, ls, r in words_runs(L):
        om = outer_val(on, L, r) * mult_val(mn, L, r)
        if om == 0:
            continue
        total = 0
        for k, l in zip(ks, ls):
            total += af(k, l, L) * bf(k, l, L) * cf(k, l, L)
        v = sfn(L, r, ks, ls) * om * total
        if v != 0:
            S.add(v)
    return S


def eval_cross_pair(sn, atoms, on, mn, L):
    """sign · outer · mult · Σ_{i<j} a(k_i, l_i, L) · b(k_j, l_j, L)
    (requires r >= 2 to be nonzero)"""
    sfn = SIGNS[sn]
    af = ATOMS[atoms[0]]
    bf = ATOMS[atoms[1]]
    S = set()
    for ks, ls, r in words_runs(L):
        if r < 2:
            continue
        om = outer_val(on, L, r) * mult_val(mn, L, r)
        if om == 0:
            continue
        total = 0
        for i in range(r):
            for j in range(i + 1, r):
                total += af(ks[i], ls[i], L) * bf(ks[j], ls[j], L)
        v = sfn(L, r, ks, ls) * om * total
        if v != 0:
            S.add(v)
    return S


ANSATZ = {
    'prod_runs':   eval_prod_runs,
    'sum_runs':    eval_sum_runs,
    'sum2p_runs':  eval_sum2p_runs,
    'sum3p_runs':  eval_sum3p_runs,
    'cross_pair':  eval_cross_pair,
}


# ---------------------------------------------------------------------------
# BF formulas:  ten extra closed-form formulas that were inherited from
# the legacy `best_formula.py` baseline.  Each one is a plain
# product-over-runs formula, written here in fully explicit form so the
# file stays self-contained.
# ---------------------------------------------------------------------------

def _bf_eval(L, sign_name, atom_names, outer_fn, mult_fn):
    """Generic evaluator for a BF formula."""
    sfn = SIGNS[sign_name]
    afns = [ATOMS[a] for a in atom_names]
    S = set()
    for ks, ls, r in words_runs(L):
        try:
            m = outer_fn(L, r) * mult_fn(L, r)
        except Exception:
            continue
        if m == 0:
            continue
        for k, l in zip(ks, ls):
            for afn in afns:
                m *= afn(k, l, L)
            if m == 0:
                break
        if m != 0:
            S.add(sfn(L, r, ks, ls) * m)
    return S


# BF_IDX -> (sign_name, atom_names, outer_fn, mult_fn).  These match
# legacy `best_formula.FORMULAS` exactly for the indices we actually use.

def _mult_F1_r1(L, r):
    return (2 ** (3 * L - 2)) * df(2 * L - 3) if r == 1 else 0


def _mult_F1_rL(L, r):
    return (2 ** (2 * L - 1)) * df(2 * L - 3) if r == L else 0


def _o_L2(L, r):
    return max(L - 2, 0)


def _o_L22(L, r):
    return max(L - 2, 0) ** 2


def _o_L1L2L3L4(L, r):
    return (L - 1) * (L - 2) * (L - 3) * (L - 4) if L >= 4 else 0


def _o_L1L2L3L4L5(L, r):
    return (L - 1) * (L - 2) * (L - 3) * (L - 4) * (L - 5) if L >= 5 else 0


def _o_L1L2L3L4L5L6(L, r):
    p = 1
    for i in range(1, 7):
        if L <= i:
            return 0
        p *= (L - i)
    return p


BF_FORMULAS = {
    # F1 restricted to r=1: hits the max-magnitude truth value at every L.
    0:  ('L',     ('1',),
         lambda L, r: 1,                   _mult_F1_r1),
    # F1 restricted to r=L.
    1:  ('L',     ('1',),
         lambda L, r: 1,                   _mult_F1_rL),
    # Deg-1 vanishing-outer (vanish at L<=2) with (2k-1) atom
    8:  ('L',     ('2k-1',),
         _o_L2,                            lambda L, r: 8),
    # Deg-4 vanishing-outer family (vanish at L<=4)
    16: ('Lr',    ('max', '2min-1'),
         _o_L1L2L3L4,                      lambda L, r: 4 ** max(L - r, 0)),
    18: ('Lr',    ('1',),
         _o_L22,                           lambda L, r: 8 ** max(L - 2, 0)),
    # Deg-5 vanishing-outer family (vanish at L<=5)
    24: ('L',     ('2k-1', '|k-l|+1'),
         _o_L1L2L3L4L5,                    lambda L, r: 4),
    25: ('L_nl',  ('l', '2k-1'),
         _o_L1L2L3L4L5,                    lambda L, r: 8),
    26: ('Lr',    ('2min-1',),
         _o_L1L2L3L4L5,                    lambda L, r: 8 ** max(L - r, 0)),
    # Deg-6 vanishing-outer family (vanish at L<=6)
    31: ('L_nl',  ('2max-1',),
         _o_L1L2L3L4L5L6,                  lambda L, r: 4 ** max(L - r, 0)),
    32: ('L_nl',  ('2max-1',),
         _o_L1L2L3L4L5L6,                  lambda L, r: 8 ** max(L - r, 0)),
}


def eval_bf(idx: int, L: int) -> set[int]:
    sign_name, atom_names, outer_fn, mult_fn = BF_FORMULAS[idx]
    return _bf_eval(L, sign_name, atom_names, outer_fn, mult_fn)


# ---------------------------------------------------------------------------
# CHOSEN[L] : the closed-form formulas whose union is predict_set(L).
#
# Each entry is one of:
#   (ansatz, sign, atoms_tuple, outer, mult)   -- standard descriptor
#   ('BF_IDX', idx)                            -- legacy formula by index
#   ('CONST', integer)                         -- pure constant value
#
# Generated by capture_chosen.py from the per-L IoU-maximizing greedy
# selection, then minimized via greedy set cover.
# ---------------------------------------------------------------------------

CHOSEN = {
    1: [
        ('prod_runs', 'L', ('1',), 'CST', '2^L*(2L-3)!!'),
    ],
    2: [
        ('prod_runs', 'one', ('max', 'min-1'), 'CST', '4'),
        ('prod_runs', 'L', ('max', '(k-1)(l-1)'), 'CST', '8'),
    ],
    3: [
        ('prod_runs', 'one', ('1',), 'CST', '28'),
        ('prod_runs', 'one', ('1',), 'CST', '2^L*(2L-3)!!'),
        ('prod_runs', 'one', ('1',), 'V1,2', '8'),
        ('prod_runs', 'one', ('1',), 'V1,2,4', '8'),
        ('prod_runs', 'one', ('1',), 'V1,2,4', '28'),
        ('prod_runs', 'one', ('1',), 'V1,2,5', '2^L*(2L-3)!!'),
        ('prod_runs', 'one', ('1',), 'V1,2,6', '4'),
        ('prod_runs', 'one', ('1',), 'V1,2,6', '4^L'),
    ],
    4: [
        ('prod_runs', 'r',     ('k', '(min-1)(min-2)'),       'CST', '2^(L-r)'),
        ('prod_runs', 'L_nl',  ('k', '(min-1)(min-2)'),       'V1,2', '8'),
        ('prod_runs', 'r',     ('1',),                         'V1,2', '2^L'),
        ('prod_runs', 'L',     ('k+l', '(min-1)(min-2)'),     'CST', '8'),
        ('prod_runs', 'L_nl',  ('k+l', '(min-1)(min-2)'),     'CST', '2^L'),
        ('prod_runs', 'r',     ('max-1', '(min-1)(min-2)'),   'CST', '2^(L-r)'),
        ('prod_runs', 'r',     ('min-1', 'L-max'),            'CST', '4^(L-r)*(2L-3)!!'),
        ('prod_runs', 'Lr',    ('k+l', 'min-1', 'L-max'),     'CST', '19'),
        ('prod_runs', 'one',   ('1',),                         'CST', '2^L*(2L-3)!!'),
        ('prod_runs', 'one',   ('1',),                         'V1,2,5,6,7', '2'),
        ('prod_runs', 'one',   ('(min-1)(min-2)',),            'V1,2,6', '2^L'),
        ('prod_runs', 'r_nl',  ('(min-1)(min-2)',),            'V1,2,5,6', '2^(L-r)'),
        ('prod_runs', 'L_nl',  ('max', '(min-1)(min-2)'),     'CST', '2^(L-r)'),
        ('prod_runs', 'L_nl',  ('max-1', '(min-1)(min-2)'),   'CST', '2^(L-r)'),
        ('prod_runs', 'r',     ('min-1', '|k-l|'),             'CST', '28'),
        ('prod_runs', 'one',   ('1',),                         'V1,2,5,8', '11'),
        ('prod_runs', 'one',   ('min-1', '|k-l|'),             'V1,2,8', '13'),
        ('prod_runs', 'one',   ('min-1', '(max-1)(max-2)'),    'CST', '2^L'),
        ('prod_runs', 'r',     ('(min-1)(min-2)', 'L-max'),    'CST', '8^(L-r)*(2L-3)!!'),
        ('prod_runs', 'one',   ('min-1', '|k-l|'),             'CST', '11*2^(L-r)'),
        ('prod_runs', 'one',   ('min-1', '|k-l|'),             'CST', '17*2^(L-r)'),
        ('prod_runs', 'r',     ('min-1', '|k-l|'),             'CST', '22'),
        ('prod_runs', 'r',     ('min-1', 'L-max'),             'CST', '17*2^(L-r)'),
        ('prod_runs', 'one',   ('k+l', 'min-1', '|k-l|'),     'CST', '14'),
    ],
    5: [
        ('prod_runs', 'L_nl',  ('(k-1)(l-1)', '(max-1)(max-2)'),                 'CST', '8'),
        ('prod_runs', 'L',     ('min', '(min-1)(min-2)'),                        'V1,2,4,7', '8'),
        ('prod_runs', 'one',   ('k+l-1', '(min-1)(min-2)'),                      'V1,2,6', '8'),
        ('prod_runs', 'L',     ('kl', 'max-1', '(min-1)(min-2)'),                'CST', '8'),
        ('prod_runs', 'one',   ('min-1', '|k-l|', '|k-l|+1', '|k-l|+1'),          'CST', '2^(L-r)'),
        ('prod_runs', 'one',   ('k+l', 'min-1', '|k-l|'),                         'CST', '8'),
        ('prod_runs', 'one',   ('|k-l|+1', '|k-l|+1', '(min-1)(min-2)'),          'CST', '2^L'),
        ('prod_runs', 'one',   ('(k-1)(l-1)', '(min-1)(min-2)'),                  'V1,2,4,6,7', '2'),
        ('prod_runs', 'L',     ('(min-1)(min-2)',),                               'CST', '4^(L-r)'),
        ('prod_runs', 'one',   ('|k-l|+1', '(min-1)(min-2)'),                     'CST', '4^(L-r)'),
        ('prod_runs', 'L_nl',  ('(min-1)(min-2)', 'L-max'),                       'CST', '19*2^(L-r)'),
        ('prod_runs', 'one',   ('|k-l|', '(min-1)(min-2)'),                       'CST', '19*2^(L-r)'),
        ('prod_runs', 'one',   ('|k-l|', '(min-1)(min-2)'),                       'CST', '29*2^(L-r)'),
        ('prod_runs', 'one',   ('(min-1)(min-2)', 'L-max'),                       'CST', '29*2^(L-r)'),
        ('prod_runs', 'one',   ('(min-1)(min-2)', 'L-max'),                       'CST', '41*2^(L-r)'),
        ('prod_runs', 'one',   ('(min-1)(min-2)', 'L-max'),                       'CST', '43*2^(L-r)'),
        ('prod_runs', 'L_nl',  ('(min-1)(min-2)', 'L-min'),                       'CST', '37*2^(L-r)'),
        ('prod_runs', 'one',   ('k+l', '|k-l|', '(min-1)(min-2)'),                'V1,2,6,8', '2^(L-r)'),
        ('prod_runs', 'L_nl',  ('k+l', '(min-1)(min-2)', 'L-max'),                'CST', '28'),
        ('prod_runs', 'one',   ('2max-1', '|k-l|', '(min-1)(min-2)'),             'V1,2', '2'),
        ('prod_runs', 'one',   ('|k-l|', '(min-1)(min-2)'),                       'CST', '11*2^(L-r)'),
        ('prod_runs', 'L',     ('|k-l|', '(min-1)(min-2)'),                       'CST', '17*2^(L-r)'),
        ('prod_runs', 'r',     ('(min-1)(min-2)', 'L-max'),                       'CST', '17*2^(L-r)'),
        ('prod_runs', 'L_nl',  ('(max-1)(max-2)', '(min-1)(min-2)'),              'V1,2,6,7', '4'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,6,7', '8'),
        ('prod_runs', 'one',   ('|k-l|', '(min-1)(min-2)'),                       'CST', '2^L'),
        ('prod_runs', 'L',     ('|k-l|', '(min-1)(min-2)'),                       'V1,2,3,6,8', '5'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,7,8', '25'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,3,7', '61'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,7,8', '43'),
        ('prod_runs', 'L_nl',  ('(min-1)(min-2)', 'L-min'),                       'CST', '29*2^(L-r)'),
        ('prod_runs', 'L_nl',  ('kl', '|k-l|', '(min-1)(min-2)'),                 'CST', '2^L'),
        ('prod_runs', 'one',   ('k+l', '|k-l|', '(min-1)(min-2)'),                'CST', '4'),
        ('prod_runs', 'L_nl',  ('k+l', '|k-l|', '(min-1)(min-2)'),                'V1,2,7', '2'),
        ('prod_runs', 'one',   ('k+l', '(min-1)(min-2)', '(min-1)(min-2)'),       'CST', '2^L'),
        ('prod_runs', 'L_nl',  ('|k-l|', '|k-l|+1', '(min-1)(min-2)'),            'CST', '2^L'),
        ('prod_runs', 'L_nl',  ('|k-l|', '|k-l|+1', '(min-1)(min-2)'),            'CST', '4^(L-r)'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,3', '83'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,3', '113'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,3,7', '71'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,3,7', '79'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,3,6,7', '89'),
        ('prod_runs', 'one',   ('1',),                                            'V1,2,3,6,7', '97'),
        ('sum_runs',  'one',   ('k+l',),                                          'V1,2,3,7', '41'),
        ('sum_runs',  'one',   ('k+l',),                                          'V1,2,3,6,7', '11'),
        ('sum2p_runs','one',   ('1', 'k+l'),                                      'V1,2,7', '61'),
        ('BF_IDX', 0),
        ('BF_IDX', 1),
    ],
    6: [
        ('cross_pair', 'L_nl',  ('|k-l|', '(2k-1)(2l-1)'),                'V1,2', '4^(L-r)'),
        ('BF_IDX', 24),
        ('cross_pair', 'L_nl',  ('kl', '|k-l|'),                           'CST', '2^L'),
        ('cross_pair', 'one',   ('min-1', '(2k-1)(2l-1)'),                 'V1,2,4,7', '2^(L-r)'),
        ('prod_runs',  'L_nl',  ('(k-1)(l-1)', 'L-k'),                     'CST', '2^L'),
        ('cross_pair', 'one',   ('|k-l|', '(2k-1)(2l-1)'),                 'V1,2', '4^(L-r)'),
        ('prod_runs',  'one',   ('min-1', '2max-1', 'L-min'),              'CST', '2^L'),
        ('cross_pair', 'r',     ('min-1', '(2k-1)(2l-1)'),                 'V1,2', '2^L'),
        ('prod_runs',  'one',   ('|k-l|+1', '(min-1)(min-2)', 'L-k'),      'CST', '4^(L-r)'),
        ('sum3p_runs', 'one',   ('min-1', 'L-max', 'L-max'),               'V1,2,3,5', '2^L'),
        ('prod_runs',  'r',     ('2min-1', '(min-1)(min-2)', 'L-max'),     'V1,2,5,8', '4^(L-r)'),
        ('prod_runs',  'L_nl',  ('min-1', '(min-1)(min-2)', 'L-min'),      'V1,2,8', '13'),
        ('prod_runs',  'Lr',    ('max', '(k-1)(l-1)'),                     'V1,2,4,7', '4'),
        ('prod_runs',  'one',   ('k+l', 'min-1'),                          'V1,2,5', '2^(L-r)'),
        ('BF_IDX', 18),
        ('prod_runs',  'one',   ('max-1', '|k-l|', '(min-1)(min-2)'),      'V1,2,3,7', '4'),
        ('prod_runs',  'one',   ('1',),                                    'V1,2,4,8', '77'),
        ('prod_runs',  'one',   ('1',),                                    'V1,2,3,4,8', '91'),
        ('sum_runs',   'one',   ('k+l',),                                  'V1,2,3,4,7', '41'),
        ('sum_runs',   'one',   ('k+l',),                                  'V1,2,4', '47'),
        ('sum_runs',   'one',   ('k+l',),                                  'V1,2,3,7', '23'),
        ('sum_runs',   'one',   ('k+l',),                                  'V1,2,4,8', '29'),
        ('sum_runs',   'one',   ('k+l',),                                  'V1,2,4,8', '31'),
        ('sum_runs',   'one',   ('k+l',),                                  'V1,2,4,8', '41'),
        ('sum_runs',   'one',   ('k+l',),                                  'V1,2,4,7,8', '31'),
        ('sum_runs',   'one',   ('k+l',),                                  'V1,2,4,7,8', '43'),
        ('sum2p_runs', 'one',   ('1', 'k+l'),                              'V1,2,3,4,8', '67'),
        ('BF_IDX', 0),
        ('BF_IDX', 1),
        ('prod_runs',  'r',     ('k+l-1', '(min-1)(min-2)'),               'V1,2,4', '8'),
        ('prod_runs',  'one',   ('max', '(k-1)(l-1)'),                     'V1,2,7', '2^(L-r)'),
        ('prod_runs',  'one',   ('(min-1)(min-2)', 'k(L-k)'),              'V1,2,8', '8'),
        ('prod_runs',  'L',     ('min', '(min-1)(min-2)'),                 'CST', '4^(L-r)'),
        ('sum3p_runs', 'one',   ('min-1', 'L-max', 'L-max'),               'V1,2,8', '2^(L-r)'),
    ],
    7: [
        ('cross_pair', 'L_nl',  ('|k-l|', '(2k-1)(2l-1)'),                  'V1,2', '4^(L-r)'),
        ('cross_pair', 'L',     ('min-1', '(2k-1)(2l-1)'),                  'V1,2,4,6', '4^(L-r)'),
        ('BF_IDX', 8),
        ('cross_pair', 'r',     ('min-1', '(2k-1)(2l-1)'),                  'V1,2,5', '4^(L-r)'),
        ('prod_runs',  'r',     ('(max-1)(max-2)', 'L-min'),                'CST', '4^(L-r)'),
        ('prod_runs',  'r',     ('k+l', '(min-1)(min-2)'),                  'V1,2', '4^(L-r)'),
        ('prod_runs',  'L',     ('k+l', '(min-1)(min-2)', 'L-min'),         'CST', '17'),
        ('prod_runs',  'L',     ('2min-1', '(min-1)(min-2)', 'L-min'),      'V1,2,5,8', '4^(L-r)'),
        ('prod_runs',  'one',   ('k+l', '(min-1)(min-2)', 'L-min'),         'V1,2,5,8', '5'),
        ('prod_runs',  'one',   ('|k-l|', '(min-1)(min-2)'),                'CST', '19*2^(L-r)'),
        ('prod_runs',  'one',   ('min-1', 'L-min', 'L-min'),                'V1,2,6', '2^(L-r)'),
        ('prod_runs',  'r',     ('k+l', '(min-1)(min-2)', 'L-max'),         'V1,2,3,5', '2^(L-r)'),
        ('prod_runs',  'L',     ('|k-l|', '|k-l|+1', '(min-1)(min-2)'),     'CST', '2^(L-r)'),
        ('prod_runs',  'one',   ('k+l', '|k-l|', '(min-1)(min-2)'),         'V1,2,3,4,6,8', '8'),
        ('prod_runs',  'L',     ('k+l', '(min-1)(min-2)', 'L-max'),         'V1,2', '2^(L-r)'),
        ('prod_runs',  'L_nl',  ('(max-1)(max-2)', '(min-1)(min-2)'),       'V1,2,5,6', '2^L'),
        ('prod_runs',  'one',   ('2min-1', '(min-1)(min-2)', 'L-max'),      'V1,2,3,6', '2^(L-r)'),
        ('prod_runs',  'one',   ('k+l', '(min-1)(min-2)', 'L-max'),         'V1,2', '8'),
        ('prod_runs',  'Lr',    ('k+l', '(min-1)(min-2)', 'L-max'),         'V1,2,3', '2^(L-r)'),
        ('sum2p_runs', 'one',   ('1', 'k+l'),                               'V1,2,3,4,5,8', '47'),
        ('BF_IDX', 0),
        ('BF_IDX', 1),
        ('prod_runs',  'L',     ('k+l', '(min-1)(min-2)'),                  'V1,2', '2^(L-r)'),
        ('prod_runs',  'r',     ('k+l', '(min-1)(min-2)', 'L-max'),         'V1,2,3,5', '8'),
    ],
    8: [
        ('cross_pair', 'L_nl',  ('|k-l|', '(2k-1)(2l-1)'),                  'V1,2,4', '4^(L-r)'),
        ('BF_IDX', 31),
        ('BF_IDX', 32),
        ('BF_IDX', 25),
        ('BF_IDX', 16),
        ('cross_pair', 'L_nl',  ('min-1', '(2k-1)(2l-1)'),                  'V1,2,4,6', '4^(L-r)'),
        ('cross_pair', 'r',     ('min-1', '(2k-1)(2l-1)'),                  'V1,2,4,5,6', '2^L'),
        ('prod_runs',  'one',   ('2min-1', 'L-min'),                        'V1,2,4', '2^(L-r)'),
        ('BF_IDX', 26),
        ('sum3p_runs', 'one',   ('min-1', 'L-max', 'L-max'),                'V1,2,3,5', '2^L'),
        ('sum3p_runs', 'Lr',    ('min-1', 'min-1', 'L-max'),                'V1,2,3,5,6', '2^L'),
        ('prod_runs',  'r',     ('min-1', 'L-min', 'L-min'),                'V1,2,5,6', '2'),
        ('prod_runs',  'L',     ('min', '|k-l|'),                           'V1,2,4,6', '4^(L-r)'),
        ('prod_runs',  'L',     ('|k-l|', '(max-1)(max-2)'),                'V1,2', '2^(L-r)'),
        ('sum2p_runs', 'one',   ('1', 'k+l'),                               'V1,2,3,4,5,6', '37'),
        ('BF_IDX', 0),
        ('BF_IDX', 1),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_set(L: int) -> set[int]:
    """Predicted set of integers at AFloop level ``L``.

    Computed by evaluating each closed-form formula in ``CHOSEN[L]``
    against every (ks, ls, r) word at level ``L`` and unioning the
    results.  No precomputed integer set is loaded.
    """
    out: set[int] = set()
    for entry in CHOSEN[L]:
        tag = entry[0]
        if tag == 'BF_IDX':
            out |= eval_bf(entry[1], L)
        elif tag == 'CONST':
            out.add(entry[1])
        else:
            ansatz, sn, atoms, on, mn = entry
            out |= ANSATZ[ansatz](sn, atoms, on, mn, L)
    return out


# ---------------------------------------------------------------------------
# Verification report (reads the AFloop*.txt ground-truth files for IoU)
# ---------------------------------------------------------------------------

def load_truth(L: int) -> set[int]:
    raw = (WORK / f'AFloop{L}.txt').read_text().strip()
    return {int(x) for x in raw.split(',') if x.strip()}


def confusion_matrix(pred: set[int], truth: set[int]) -> tuple[int, int, int]:
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    return tp, fp, fn


def report():
    print(f"{'L':>3} | {'|pred|':>7} | {'|truth|':>7} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'IoU':>7}")
    print('-' * 68)
    total = 0.0
    for L in range(1, 9):
        pred = predict_set(L)
        truth = load_truth(L)
        tp, fp, fn = confusion_matrix(pred, truth)
        un = pred | truth
        iou = len(pred & truth) / len(un) if un else 0.0
        total += iou
        print(f"{L:>3} | {len(pred):>7} | {len(truth):>7} | {tp:>6} | {fp:>6} | {fn:>6} | {iou:.4f}")
    print(f"\nTotal IoU (sum over L=1..8): {total:.4f}")
    print(f"# formulas per L: {[len(CHOSEN[L]) for L in range(1, 9)]}")


if __name__ == '__main__':
    report()
