"""Count-Min Sketch: frequency estimates over a stream in sublinear memory,
plus streaming heavy-hitter tracking.

Guarantee (Cormode & Muthukrishnan 2005): with width w = ceil(e/eps) and
depth d = ceil(ln(1/delta)), for every key
    true <= estimate <= true + eps * N      with probability >= 1 - delta
where N is the total stream count. Estimates NEVER undercount.
"""

import math
import random

_MERSENNE_P = (1 << 61) - 1  # prime for universal hashing


class CountMinSketch:
    def __init__(self, eps=0.001, delta=0.01, seed=0, conservative=False):
        self.width = math.ceil(math.e / eps)
        self.depth = math.ceil(math.log(1 / delta))
        self.conservative = conservative
        rng = random.Random(seed)
        # Universal hash family: h(x) = ((a*x + b) mod p) mod width
        self.hashes = [(rng.randrange(1, _MERSENNE_P), rng.randrange(_MERSENNE_P))
                       for _ in range(self.depth)]
        self.table = [[0] * self.width for _ in range(self.depth)]
        self.total = 0

    def _rows(self, key):
        x = hash(key) & 0x7FFFFFFFFFFFFFFF
        for row, (a, b) in enumerate(self.hashes):
            yield row, ((a * x + b) % _MERSENNE_P) % self.width

    def add(self, key, count=1):
        self.total += count
        if not self.conservative:
            for row, col in self._rows(key):
                self.table[row][col] += count
        else:
            # Conservative update: only raise counters to the minimum needed
            # to keep the estimate consistent. Strictly less overestimation,
            # same never-undercount guarantee — but it breaks mergeability.
            cells = list(self._rows(key))
            new_est = min(self.table[r][c] for r, c in cells) + count
            for r, c in cells:
                self.table[r][c] = max(self.table[r][c], new_est)

    def query(self, key):
        """Point estimate: min across rows. Each row only ever OVERcounts
        (collisions add, never subtract), so min is the tightest row."""
        return min(self.table[r][c] for r, c in self._rows(key))


class HeavyHitters:
    """Track the top-k most frequent stream items with a CMS + candidate set.

    On each arrival, query the sketch; if the estimate beats the smallest
    tracked count, it evicts that entry. Memory: O(width*depth + k), never
    O(#distinct keys).
    """

    def __init__(self, k, eps=0.0005, delta=0.01, seed=0):
        self.k = k
        self.cms = CountMinSketch(eps, delta, seed, conservative=True)
        self.tracked = {}         # key -> est_count

    def add(self, key):
        self.cms.add(key)
        est = self.cms.query(key)
        if key in self.tracked:
            self.tracked[key] = est
        elif len(self.tracked) < self.k:
            self.tracked[key] = est
        else:
            victim = min(self.tracked, key=self.tracked.get)
            if est > self.tracked[victim]:
                del self.tracked[victim]
                self.tracked[key] = est

    def top(self):
        return sorted(self.tracked.items(), key=lambda kv: -kv[1])
