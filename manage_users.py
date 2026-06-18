"""
manage_users.py  —  Add, remove, list, or reset passwords for dashboard users.

Usage:
    python manage_users.py seed              # create default users from DEFAULT_USERS list
    python manage_users.py list              # show all users
    python manage_users.py add <username>    # add a new user (prompts for password)
    python manage_users.py remove <username> # remove a user
    python manage_users.py reset <username>  # change a user's password

To set default users: edit the DEFAULT_USERS list at the top of this file,
then run:  python manage_users.py seed

Passwords are stored as secure hashes in users.json — never in plain text.
"""
from __future__ import annotations

import getpass
import json
import sys
from pathlib import Path

try:
    from werkzeug.security import generate_password_hash, check_password_hash
except ImportError:
    print("ERROR: werkzeug not installed.  Run:  pip install werkzeug")
    sys.exit(1)

USERS_FILE = Path(__file__).parent / "users.json"

# ── Default users (plain text here, hashed on save) ──────────────────────────
# Edit these before running: python manage_users.py seed
DEFAULT_USERS = [
    {"username": "admin",      "password": "admin123",  "role": "admin"},
    {"username": "supervisor", "password": "slt2026",   "role": "viewer"},
]


def _load() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(users: list[dict]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


def _prompt_password(username: str) -> str:
    while True:
        pw  = getpass.getpass(f"Password for '{username}': ")
        pw2 = getpass.getpass("Confirm password: ")
        if pw != pw2:
            print("Passwords do not match. Try again.\n")
            continue
        if len(pw) < 6:
            print("Password must be at least 6 characters.\n")
            continue
        return pw


def cmd_list(args: list[str]) -> None:
    users = _load()
    if not users:
        print("No users found in users.json")
        return
    print(f"{'Username':<20} {'Role':<10}")
    print("-" * 32)
    for u in users:
        print(f"{u.get('username','?'):<20} {u.get('role','viewer'):<10}")


def cmd_add(args: list[str]) -> None:
    if not args:
        print("Usage: python manage_users.py add <username>")
        sys.exit(1)
    username = args[0].strip()
    users = _load()
    if any(u.get("username", "").lower() == username.lower() for u in users):
        print(f"User '{username}' already exists. Use 'reset' to change their password.")
        sys.exit(1)
    role = input("Role [admin/viewer, default=viewer]: ").strip().lower() or "viewer"
    if role not in ("admin", "viewer"):
        print("Unknown role — defaulting to 'viewer'.")
        role = "viewer"
    password = _prompt_password(username)
    users.append({
        "username": username,
        "password": generate_password_hash(password),
        "role": role,
    })
    _save(users)
    print(f"✓ User '{username}' ({role}) added successfully.")


def cmd_remove(args: list[str]) -> None:
    if not args:
        print("Usage: python manage_users.py remove <username>")
        sys.exit(1)
    username = args[0].strip()
    users = _load()
    new_users = [u for u in users if u.get("username", "").lower() != username.lower()]
    if len(new_users) == len(users):
        print(f"User '{username}' not found.")
        sys.exit(1)
    confirm = input(f"Remove user '{username}'? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    _save(new_users)
    print(f"✓ User '{username}' removed.")


def cmd_reset(args: list[str]) -> None:
    if not args:
        print("Usage: python manage_users.py reset <username>")
        sys.exit(1)
    username = args[0].strip()
    users = _load()
    for u in users:
        if u.get("username", "").lower() == username.lower():
            password = _prompt_password(username)
            u["password"] = generate_password_hash(password)
            _save(users)
            print(f"✓ Password for '{username}' updated.")
            return
    print(f"User '{username}' not found.")
    sys.exit(1)


def cmd_seed(args: list[str]) -> None:
    """Create all DEFAULT_USERS — skips any that already exist."""
    users = _load()
    existing = {u.get("username", "").lower() for u in users}
    added = 0
    for entry in DEFAULT_USERS:
        uname = entry["username"]
        if uname.lower() in existing:
            print(f"  skip  '{uname}' — already exists")
            continue
        users.append({
            "username": uname,
            "password": generate_password_hash(entry["password"]),
            "role":     entry["role"],
        })
        print(f"  ✓ added '{uname}' ({entry['role']}) with password '{entry['password']}'")
        added += 1
    _save(users)
    print(f"\nDone — {added} user(s) added.")


COMMANDS = {
    "list":   cmd_list,
    "add":    cmd_add,
    "remove": cmd_remove,
    "reset":  cmd_reset,
    "seed":   cmd_seed,
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in COMMANDS:
        print(__doc__)
        print("Commands:", ", ".join(COMMANDS))
        sys.exit(1)
    COMMANDS[args[0]](args[1:])


if __name__ == "__main__":
    main()
