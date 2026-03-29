"""
One-time interactive setup wizard for API credentials.
Run this ONCE to store keys in Windows Credential Manager.
Keys are stored encrypted by Windows — never in your code or files.

Usage:
    python setup_secrets.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import getpass
from utils.secrets import store_secret, list_configured_secrets, get_secret

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         AlgoTrading — Credential Setup Wizard           ║
║                                                          ║
║  Keys are stored in Windows Credential Manager.         ║
║  They are NEVER written to files or source code.        ║
╚══════════════════════════════════════════════════════════╝
"""

def setup():
    print(BANNER)

    # Show current status
    print("Current credential status:")
    for key, status in list_configured_secrets():
        print(f"  {key:<25} {status}")
    print()

    print("Which broker do you want to configure?")
    print("  1. Alpaca  (recommended — free, instant access, no minimums)")
    print("  2. eToro   (requires developer key approval at api-portal.etoro.com)")
    print("  3. Both")
    print("  4. Exit")
    choice = input("\nChoice [1/2/3/4]: ").strip()

    if choice in ("1", "3"):
        _setup_alpaca()

    if choice in ("2", "3"):
        _setup_etoro()

    print("\nFinal status:")
    for key, status in list_configured_secrets():
        print(f"  {key:<25} {status}")

    print("\n✓ Setup complete. Run: python main.py forward")


def _setup_alpaca():
    print("\n── Alpaca Setup ──────────────────────────────────────")
    print("1. Go to https://app.alpaca.markets → sign up free")
    print("2. Go to: Home → API Keys → Generate New Key")
    print("3. Copy both the API Key ID and Secret Key")
    print()

    paper = input("Are you using Paper Trading? (yes/no) [yes]: ").strip().lower() or "yes"
    is_paper = paper != "no"

    api_key = getpass.getpass("Alpaca API Key ID: ").strip()
    secret_key = getpass.getpass("Alpaca Secret Key: ").strip()

    if api_key:
        store_secret("ALPACA_API_KEY", api_key)
    if secret_key:
        store_secret("ALPACA_SECRET_KEY", secret_key)

    # Store paper mode flag
    store_secret("ALPACA_PAPER", "true" if is_paper else "false")

    print(f"  ✓ Alpaca credentials stored (mode: {'paper' if is_paper else 'LIVE'})")
    if is_paper:
        print("  ⚠  Paper trading — no real money. Switch to live when ready.")


def _setup_etoro():
    print("\n── eToro Setup ───────────────────────────────────────")
    print("PREREQUISITE: You must have a developer API key from eToro.")
    print("Apply at: https://api-portal.etoro.com")
    print("(Approval takes 1–4 weeks. Use Alpaca paper trading in the meantime.)")
    print()

    has_key = input("Do you have your eToro API key? (yes/no): ").strip().lower()
    if has_key != "yes":
        print("  → Come back after approval. Using signal-only mode for now.")
        return

    api_key = getpass.getpass("eToro API Key (x-token): ").strip()
    if api_key:
        store_secret("ETORO_API_KEY", api_key)
        print("  ✓ eToro API key stored in Windows Credential Manager")


if __name__ == "__main__":
    setup()
