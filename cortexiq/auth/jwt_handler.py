"""JWT token creation and verification."""
from datetime import datetime, timedelta
from jose import jwt, JWTError
from ..config import SECRET_KEY

ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8
TOKEN_EXPIRE_REMEMBER = 720  # 30 days


def create_token(user_id: int, username: str, email: str, tier: str, remember: bool = False) -> str:
    expire = TOKEN_EXPIRE_REMEMBER if remember else TOKEN_EXPIRE_HOURS
    payload = {
        "sub": str(user_id),
        "username": username,
        "email": email,
        "tier": tier,
        "exp": datetime.utcnow() + timedelta(hours=expire),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
