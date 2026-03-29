"""
Full automation setup — registers a Windows Task Scheduler job
that runs all strategies every market day at 09:35 AM (market open + 5min).

Usage:
    python automate.py install    # Register the scheduled task
    python automate.py remove     # Remove the scheduled task
    python automate.py status     # Check if task is running
    python automate.py run        # Run manually right now
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_DIR = Path(__file__).parent.resolve()
PYTHON = sys.executable
TASK_NAME = "AlgoTrading_DailySignals"
SCRIPT = PROJECT_DIR / "main.py"
LOG_FILE = PROJECT_DIR / "logs" / "scheduler.log"


def install():
    """Register daily Task Scheduler job (runs Mon–Fri at 09:35 AM)."""
    print(f"\nInstalling scheduled task: {TASK_NAME}")
    print(f"  Script : {SCRIPT}")
    print(f"  Python : {PYTHON}")
    print(f"  Time   : 09:35 AM, Monday–Friday")
    print(f"  Log    : {LOG_FILE}\n")

    # The command the scheduler will run
    run_cmd = f'"{PYTHON}" "{SCRIPT}" signal >> "{LOG_FILE}" 2>&1'

    # Build schtasks command
    schtasks_cmd = [
        "schtasks", "/create",
        "/tn", TASK_NAME,
        "/tr", run_cmd,
        "/sc", "weekly",
        "/d", "MON,TUE,WED,THU,FRI",
        "/st", "09:35",
        "/f",  # overwrite if exists
    ]

    result = subprocess.run(schtasks_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Task '{TASK_NAME}' installed successfully.")
        print(f"\nThe system will automatically run every market day at 09:35 AM.")
        print(f"Signals and trades will be logged to: {LOG_FILE}")
        print(f"\nTo also run the dashboard daily, install a second task:")
        print(f"  python automate.py install-dashboard")
    else:
        print(f"✗ Failed to install task: {result.stderr}")
        print("  Try running this script as Administrator.")


def install_dashboard():
    """Register a second task that regenerates the dashboard daily at 09:40 AM."""
    dash_cmd = f'"{PYTHON}" "{PROJECT_DIR / "dashboard.py"}" >> "{LOG_FILE}" 2>&1'
    schtasks_cmd = [
        "schtasks", "/create",
        "/tn", f"{TASK_NAME}_Dashboard",
        "/tr", dash_cmd,
        "/sc", "weekly",
        "/d", "MON,TUE,WED,THU,FRI",
        "/st", "09:40",
        "/f",
    ]
    result = subprocess.run(schtasks_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Dashboard task installed (runs at 09:40 AM daily).")
    else:
        print(f"✗ Failed: {result.stderr}")


def remove():
    """Remove the scheduled task."""
    for task in [TASK_NAME, f"{TASK_NAME}_Dashboard"]:
        result = subprocess.run(
            ["schtasks", "/delete", "/tn", task, "/f"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✓ Task '{task}' removed.")
        else:
            print(f"  Task '{task}' not found (already removed or never installed).")


def status():
    """Show current task status."""
    for task in [TASK_NAME, f"{TASK_NAME}_Dashboard"]:
        result = subprocess.run(
            ["schtasks", "/query", "/tn", task, "/fo", "LIST"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"\n── {task} ──")
            for line in result.stdout.splitlines():
                if any(k in line for k in ["Task Name", "Next Run", "Last Run", "Status", "Last Result"]):
                    print(f"  {line.strip()}")
        else:
            print(f"  {task}: NOT INSTALLED")


def run_now():
    """Run the signal generator manually right now."""
    print("Running signal generator...")
    os.chdir(PROJECT_DIR)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "signal"],
        cwd=str(PROJECT_DIR)
    )
    return result.returncode


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    dispatch = {
        "install": install,
        "install-dashboard": install_dashboard,
        "remove": remove,
        "status": status,
        "run": run_now,
    }
    if cmd in dispatch:
        dispatch[cmd]()
    else:
        print(__doc__)
