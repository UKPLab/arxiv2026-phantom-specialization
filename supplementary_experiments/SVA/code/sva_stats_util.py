"""Canonical exact Wilcoxon signed-rank test for the SVA analyses.

scipy.stats.wilcoxon's method resolution does not midrank ties
consistently across code paths (documented discrepancy on the 1.4b
five-condition diffs: sva_aggregate saved 0.020424, the NB12-validated
aggregate path gave 0.020630, exact sign-flip enumeration with midranks
gives 0.019867). All SVA scripts use this single implementation:
exact enumeration of all 2^n sign flips of the average-ranked |diffs|
(zeros dropped, the "wilcox" zero method), one-sided greater.

TIE EXACTNESS: callers must pass diffs on which equality comparisons
are mathematically exact -- integer-scaled counts (e.g. (k+1)*own -
colsum out of a common denominator) or fractions.Fraction values when
denominators vary. Raw float accuracies carry ~5e-8 float noise that
splits true count-level ties. Ranking is done
in pure Python with exact == comparisons; average ranks are multiples
of 0.5, whose double sums are exact for the n used here.
"""

from itertools import product


def average_ranks(values):
    """Average (mid) ranks of `values` using exact == comparisons."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def exact_wilcoxon_greater(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = average_ranks([abs(x) for x in d])
    w_obs = sum(r for x, r in zip(d, ranks) if x > 0)
    count = 0
    for signs in product([0, 1], repeat=n):
        w = sum(r for s, r in zip(signs, ranks) if s)
        if w >= w_obs - 1e-9:
            count += 1
    return count / 2 ** n
