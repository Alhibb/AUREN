"""Load `.env` from the project root and read settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Read `.env` next to app.py, even if Streamlit's cwd is a subfolder."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def secret(name: str):
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        return None


def setting(name: str, default=None):
    """Prefer environment / `.env`, then Streamlit secrets."""
    value = os.getenv(name)
    if value is not None:
        value = value.strip()
        if value:
            return value
    return secret(name) or default
