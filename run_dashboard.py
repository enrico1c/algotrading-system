# run_dashboard.py
import subprocess, webbrowser, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
exec(open('dashboard.py').read())
webbrowser.open('reports/dashboard.html')
