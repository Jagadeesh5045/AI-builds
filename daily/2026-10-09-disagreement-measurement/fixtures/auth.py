"""Simple authentication helpers for the demo codebase."""

import hashlib


def hash_password(password):
    """Return the SHA-256 hex digest of a password string."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def authenticate(username, password, user_store):
    """Check username and password against the user store dict."""
    stored = user_store.get(username)
    if stored is None:
        return False
    return stored == hash_password(password)


def make_session_token(username):
    """Build a readable session token for a logged-in user."""
    return "session-" + username + "-active"
