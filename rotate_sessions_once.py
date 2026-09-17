from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("SOUNDLENS_DATA_DIR", "/data" if Path("/data").exists() else "."))
USERS = DATA_DIR / "soundlens_users.json"
MARKER = DATA_DIR / ".sessions_rotated_20260917"
BACKUPS = DATA_DIR / "soundlens_backups"

if MARKER.exists():
    print("[security] session rotation already completed")
    raise SystemExit(0)

if not USERS.exists():
    print("[security] no user database found; marking rotation complete")
    MARKER.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    raise SystemExit(0)

BACKUPS.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
backup = BACKUPS / f"soundlens_users_before_session_rotation_{stamp}.json"
shutil.copy2(USERS, backup)

data = json.loads(USERS.read_text(encoding="utf-8"))
tokens = data.get("tokens") if isinstance(data, dict) else None
count = len(tokens) if isinstance(tokens, dict) else 0
if not isinstance(data, dict):
    raise RuntimeError("User database is not a JSON object")

data["tokens"] = {}
tmp = USERS.with_name(f".{USERS.name}.session-rotation.tmp")
tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
os.replace(tmp, USERS)
MARKER.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
print(f"[security] rotated {count} existing sessions; backup={backup.name}")
