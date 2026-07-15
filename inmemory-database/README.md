# Multi-Level In-Memory Database (Anthropic/CodeSignal Style)

**Problem:** Build an in-memory key-value database in **four levels of
escalating requirements**, the format Anthropic's 90-minute CodeSignal
industry-coding screen uses (one problem, four levels, each building on the
last). The skill under test is not algorithms — it's **absorbing new
requirements without breaking earlier levels**: clean state modeling, edge
cases, and refactoring under time pressure.

Work it as intended: implement level 1, run only level-1 tests, then reveal
level 2, and so on. Every earlier test must stay green after each level.

---

## The Levels

**Level 1 — basics.** `set / get / delete / exists`. `delete` returns
whether the key existed. Overwrites win.

**Level 2 — scans.** `scan(prefix)` returns all live `(key, value)` pairs
with that key prefix, **sorted by key**.

**Level 3 — TTL.** Every operation takes a logical `timestamp`; `set` takes
an optional `ttl`. A record with ttl set at time `t` is alive on
`[t, t + ttl)` — **the expiry instant is exclusive**. Expired records are
invisible to every operation (`get`, `exists`, `scan`, and `delete`, which
must return False for an expired key). Re-setting a key *without* ttl clears
the old ttl.

**Level 4 — backup/restore.** `backup(timestamp)` snapshots live records;
`restore(snapshot, timestamp)` replaces all current state. The subtlety the
graders probe: snapshots must store **remaining** ttl, so a record with 7
seconds left at backup time has exactly 7 seconds left after a restore —
whenever that restore happens.

---

## Design Notes / Gotchas

- **Inject time; never call the clock.** Every method takes `timestamp`.
  This is what makes level 3 testable — and it's exactly how the real
  CodeSignal problem phrases it (`SET_AT`, `GET_AT`, ...).
- **Lazy expiry beats active expiry here.** Storing absolute expiry and
  checking liveness on read (`timestamp < expiry`) is a few lines and O(1);
  a background sweep or priority queue is over-engineering for the given
  operations. Be ready to *defend* that choice — the interview asks.
- **Expired ≠ deleted, but must behave identically.** The classic lost
  point: `delete` on an expired key returning True, or `scan` leaking
  expired records.
- **Remaining-ttl re-basing** is the level-4 trap. Storing absolute expiry
  in the snapshot fails any test where restore happens at a different time
  than backup. Convert to remaining at backup, re-base at restore.
- **Boundary discipline.** Alive on `[t, t+ttl)`: `get` at exactly
  `t + ttl` returns None. Off-by-one here fails hidden tests silently.
- **State layout matters for evolution.** Two dicts (`data`, `expiry`)
  absorb all four levels without a rewrite; entangling value and expiry in
  ad-hoc tuples from level 1 makes level 3 a refactor under the clock.

---

## Running the Smoke Test

```bash
pip install pytest
python -m pytest test_database.py -v      # tests grouped by level
```

---

## Discussion Questions (interview follow-ups)

- Level 5 candidates the real screens use: key **versioning/history**
  (`get_at` past values), **numeric operations with type errors**, or
  **capacity limits with eviction**. Sketch how your state layout absorbs
  each.
- When would lazy expiry become a real memory problem, and what's the
  minimal fix (expiry heap sweep amortized on writes)?
- `scan` is O(n log n) per call here. What structure gives O(log n + m)
  prefix scans (sorted container / trie), and at what write cost?
- How would you make this thread-safe: one lock, striped locks, or MVCC —
  and what does `backup` need under concurrency (consistent snapshot)?
