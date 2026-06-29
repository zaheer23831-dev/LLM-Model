"""
Minimal, dependency-free authentication.

Passwords are hashed with PBKDF2 (from Python's standard library — no extra
packages, no Windows build issues). Logins return a random session token that
the browser stores and sends back as `Authorization: Bearer <token>`.
"""
import hashlib
import secrets

from database import SessionLocal, User, SessionToken

_ITERATIONS = 100_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS).hex()
    return secrets.compare_digest(check, digest)


def create_user(username: str, password: str):
    """Returns user id, or None if the username is taken."""
    username = username.strip()
    with SessionLocal() as db:
        if db.query(User).filter_by(username=username).first():
            return None
        user = User(username=username, password_hash=hash_password(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user.id


def login(username: str, password: str):
    """Returns {token, user_id, username} on success, else None."""
    with SessionLocal() as db:
        user = db.query(User).filter_by(username=username.strip()).first()
        if not user or not verify_password(password, user.password_hash):
            return None
        token = secrets.token_urlsafe(32)
        db.add(SessionToken(token=token, user_id=user.id))
        db.commit()
        return {"token": token, "user_id": user.id, "username": user.username}


def logout(token: str):
    with SessionLocal() as db:
        row = db.get(SessionToken, token)
        if row:
            db.delete(row)
            db.commit()


def user_for_token(token):
    """Resolve a bearer token to {id, username}, or None."""
    if not token:
        return None
    with SessionLocal() as db:
        sess = db.get(SessionToken, token)
        if not sess:
            return None
        user = db.get(User, sess.user_id)
        return {"id": user.id, "username": user.username} if user else None
