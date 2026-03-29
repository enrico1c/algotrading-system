"""
Secure credential storage — NEVER put API keys in source code or .env files.

Priority order for loading credentials:
  1. Windows Credential Manager (most secure — key never touches disk as plaintext)
  2. OS environment variables (set once at system level, not in code)
  3. Interactive prompt (first-time setup only, then saves to Credential Manager)

Usage:
    from utils.secrets import get_secret, store_secret

    api_key = get_secret("ETORO_API_KEY")   # loads from Credential Manager
    api_key = get_secret("ALPACA_API_KEY")
"""
from __future__ import annotations

import os
import sys
import getpass
from typing import Optional

from utils.logger import get_logger

logger = get_logger("utils.secrets")

# Service name in Windows Credential Manager
_SERVICE = "AlgoTrading"


def get_secret(key: str, prompt_if_missing: bool = False) -> Optional[str]:
    """
    Load a secret in priority order:
      1. Windows Credential Manager
      2. Environment variable
      3. Interactive prompt (only if prompt_if_missing=True)
    """
    # 1. Windows Credential Manager
    value = _load_from_credential_manager(key)
    if value:
        logger.debug(f"Loaded {key} from Windows Credential Manager")
        return value

    # 2. Environment variable
    value = os.environ.get(key)
    if value:
        logger.debug(f"Loaded {key} from environment variable")
        return value

    # 3. Interactive prompt (first-time setup)
    if prompt_if_missing:
        print(f"\n[SETUP] '{key}' not found in Credential Manager or environment.")
        print(f"Enter your {key} (input is hidden):")
        value = getpass.getpass(f"{key}: ").strip()
        if value:
            store_secret(key, value)
            print(f"  ✓ Saved to Windows Credential Manager as '{key}'")
            print(f"  You will not be prompted again.\n")
            return value

    logger.warning(f"Secret '{key}' not found. Run: python setup_secrets.py")
    return None


def store_secret(key: str, value: str) -> bool:
    """Save a secret to Windows Credential Manager."""
    try:
        import keyring
        keyring.set_password(_SERVICE, key, value)
        logger.info(f"Stored '{key}' in Windows Credential Manager")
        return True
    except ImportError:
        logger.warning("keyring not installed. Run: pip install keyring")
        return False
    except Exception as e:
        logger.error(f"Could not store secret: {e}")
        return False


def delete_secret(key: str) -> bool:
    """Remove a secret from Windows Credential Manager."""
    try:
        import keyring
        keyring.delete_password(_SERVICE, key)
        logger.info(f"Deleted '{key}' from Windows Credential Manager")
        return True
    except Exception as e:
        logger.warning(f"Could not delete secret '{key}': {e}")
        return False


def list_configured_secrets() -> list:
    """Return which secrets are currently configured (not their values)."""
    keys = ["ETORO_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    configured = []
    for k in keys:
        val = _load_from_credential_manager(k) or os.environ.get(k)
        configured.append((k, "✓ Configured" if val else "✗ Missing"))
    return configured


def _load_from_credential_manager(key: str) -> Optional[str]:
    try:
        import keyring
        return keyring.get_password(_SERVICE, key)
    except ImportError:
        return None
    except Exception:
        return None
