"""Anthropic/CodeSignal-style multi-level build problem: an in-memory
key-value database that grows a level at a time. Each level's features must
keep every earlier level's tests passing — the exercise is absorbing new
requirements without rewriting.

Level 1: set / get / delete / exists
Level 2: scan by key prefix (sorted output)
Level 3: TTL — records can expire; time is injected, never wall-clock
Level 4: backup / restore — snapshots preserve REMAINING ttl at restore time
"""


class Database:
    def __init__(self):
        self._data = {}      # key -> value
        self._expiry = {}    # key -> absolute expiry timestamp (if any)

    # ---------- level 1: basic operations ----------

    def set(self, key, value, timestamp=0, ttl=None):
        """ttl=None means the record never expires. Re-setting a key without
        ttl CLEARS any previous ttl — the new write wins entirely."""
        self._data[key] = value
        if ttl is not None:
            self._expiry[key] = timestamp + ttl
        else:
            self._expiry.pop(key, None)

    def get(self, key, timestamp=0):
        if not self._alive(key, timestamp):
            return None
        return self._data[key]

    def delete(self, key, timestamp=0):
        """Returns True iff the key existed (and was alive)."""
        alive = self._alive(key, timestamp)
        self._data.pop(key, None)
        self._expiry.pop(key, None)
        return alive

    def exists(self, key, timestamp=0):
        return self._alive(key, timestamp)

    # ---------- level 2: scans ----------

    def scan(self, prefix="", timestamp=0):
        """All live (key, value) pairs whose key starts with prefix, sorted
        by key. Expired records never appear."""
        return [(k, self._data[k]) for k in sorted(self._data)
                if k.startswith(prefix) and self._alive(k, timestamp)]

    # ---------- level 3: ttl plumbing ----------

    def _alive(self, key, timestamp):
        if key not in self._data:
            return False
        exp = self._expiry.get(key)
        return exp is None or timestamp < exp

    # ---------- level 4: backup / restore ----------

    def backup(self, timestamp):
        """Snapshot live records. Stores REMAINING ttl, not absolute expiry:
        a record with 5s left must have 5s left after restore, whenever the
        restore happens."""
        snap = {}
        for k, v in self._data.items():
            if not self._alive(k, timestamp):
                continue
            exp = self._expiry.get(k)
            snap[k] = (v, None if exp is None else exp - timestamp)
        return snap

    def restore(self, snapshot, timestamp):
        """Replace current state with the snapshot, re-basing ttls onto the
        restore time. State written after the backup is discarded."""
        self._data, self._expiry = {}, {}
        for k, (v, remaining) in snapshot.items():
            self.set(k, v, timestamp=timestamp, ttl=remaining)
