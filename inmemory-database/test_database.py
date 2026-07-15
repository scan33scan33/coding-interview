from database import Database


# ---------- level 1 ----------

def test_set_get_delete_exists():
    db = Database()
    assert db.get("a") is None
    db.set("a", 1)
    assert db.get("a") == 1
    assert db.exists("a")
    assert db.delete("a") is True
    assert db.delete("a") is False        # second delete: key already gone
    assert not db.exists("a")


def test_overwrite():
    db = Database()
    db.set("a", 1)
    db.set("a", 2)
    assert db.get("a") == 2


# ---------- level 2 ----------

def test_scan_prefix_sorted():
    db = Database()
    for k, v in [("user:2", "bob"), ("user:1", "ann"), ("post:9", "x")]:
        db.set(k, v)
    assert db.scan("user:") == [("user:1", "ann"), ("user:2", "bob")]
    assert db.scan("") == [("post:9", "x"), ("user:1", "ann"), ("user:2", "bob")]
    assert db.scan("nope:") == []


# ---------- level 3 ----------

def test_ttl_expiry_boundary():
    db = Database()
    db.set("a", 1, timestamp=10, ttl=5)
    assert db.get("a", timestamp=14) == 1
    assert db.get("a", timestamp=15) is None    # expiry instant is EXCLUSIVE
    assert not db.exists("a", timestamp=15)


def test_reset_clears_ttl():
    db = Database()
    db.set("a", 1, timestamp=0, ttl=5)
    db.set("a", 2, timestamp=3)                 # no ttl: record now permanent
    assert db.get("a", timestamp=1000) == 2


def test_expired_keys_hidden_from_scan_and_delete():
    db = Database()
    db.set("a", 1, timestamp=0, ttl=5)
    db.set("b", 2, timestamp=0)
    assert db.scan("", timestamp=10) == [("b", 2)]
    assert db.delete("a", timestamp=10) is False  # expired == nonexistent


def test_ttl_overwrite_uses_new_clock():
    db = Database()
    db.set("a", 1, timestamp=0, ttl=5)
    db.set("a", 1, timestamp=4, ttl=5)          # refreshed at t=4
    assert db.get("a", timestamp=8) == 1        # would be dead under old ttl
    assert db.get("a", timestamp=9) is None


# ---------- level 4 ----------

def test_backup_restore_rebases_remaining_ttl():
    db = Database()
    db.set("a", 1, timestamp=0, ttl=10)         # 7 remaining at backup
    db.set("b", 2, timestamp=0)                 # permanent
    snap = db.backup(timestamp=3)

    db.set("c", 3, timestamp=4)                 # post-backup write
    db.restore(snap, timestamp=100)

    assert db.get("c", timestamp=100) is None   # post-backup state discarded
    assert db.get("b", timestamp=10**9) == 2    # permanent survives
    assert db.get("a", timestamp=106) == 1      # 7 remaining: alive at +6
    assert db.get("a", timestamp=107) is None   # dead at +7


def test_backup_excludes_already_expired():
    db = Database()
    db.set("a", 1, timestamp=0, ttl=5)
    snap = db.backup(timestamp=50)
    db.restore(snap, timestamp=60)
    assert db.get("a", timestamp=60) is None
    assert db.scan("", timestamp=60) == []


def test_restore_is_a_full_replacement():
    db = Database()
    db.set("keep", 1, timestamp=0)
    snap = db.backup(timestamp=0)
    db.delete("keep")
    db.set("new", 2, timestamp=1)
    db.restore(snap, timestamp=5)
    assert db.get("keep", timestamp=5) == 1
    assert db.get("new", timestamp=5) is None
