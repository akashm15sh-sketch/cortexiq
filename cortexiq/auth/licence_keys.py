"""Licence key registry with SQLite persistence for login counts."""
from datetime import datetime
from ..database import SessionLocal, LicenceUsage

# Hardcoded demo keys
LICENCE_KEYS = {
    "CORTEX-DEMO-2025-PITCH": {
        "tier": "Researcher",
        "max_logins": 5,
        "valid_until": "2030-12-31",
        "owner": "Demo User",
    },
    "CORTEX-LAB-IIT-001": {
        "tier": "Lab",
        "max_logins": 5,
        "valid_until": "2030-12-31",
        "owner": "IIT Lab Demo",
    },
    "CORTEX-TRIAL-FREE-01": {
        "tier": "Explorer",
        "max_logins": 5,
        "valid_until": "2030-12-31",
        "owner": "Trial User",
    },
}


def _get_used_logins(key: str) -> int:
    db = SessionLocal()
    record = db.query(LicenceUsage).filter(LicenceUsage.licence_key == key).first()
    count = record.used_logins if record else 0
    db.close()
    return count


def _increment_logins(key: str):
    db = SessionLocal()
    record = db.query(LicenceUsage).filter(LicenceUsage.licence_key == key).first()
    if record:
        record.used_logins += 1
        record.last_used = datetime.utcnow()
    else:
        record = LicenceUsage(licence_key=key, used_logins=1)
        db.add(record)
    db.commit()
    db.close()


def validate_and_consume_key(key: str) -> tuple:
    """Validate a licence key and consume one login.
    Returns (info_dict, None) on success or (None, error_string) on failure."""
    key = key.strip()
    if key not in LICENCE_KEYS:
        return None, "Invalid licence key. Please check and try again."

    info = LICENCE_KEYS[key]
    used = _get_used_logins(key)

    # Check expiry
    expiry = datetime.strptime(info["valid_until"], "%Y-%m-%d")
    if datetime.utcnow() > expiry:
        return None, f"This licence key expired on {info['valid_until']}."

    # Check login count
    if used >= info["max_logins"]:
        return None, f"This licence key has been used {used}/{info['max_logins']} times. Purchase a new key at cortexiq.io."

    # Consume
    _increment_logins(key)
    used += 1

    return {
        "key": key,
        "tier": info["tier"],
        "owner": info["owner"],
        "used_logins": used,
        "max_logins": info["max_logins"],
        "remaining": info["max_logins"] - used,
    }, None
