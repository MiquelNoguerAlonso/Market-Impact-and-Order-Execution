"""Run the complete numerical reproduction and verification workflow."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
for script in ['experiments.py','auction_examples.py','certificate_experiment.py',
               'orderbook_bridge.py','feedback_checks.py','partial_observation.py',
               'observation_summary_checks.py',
               'analytic_crossover_checks.py','verify.py','audit_checks.py']:
    print(f'Running {script}',flush=True)
    subprocess.run([sys.executable,str(root/'scripts'/script)],cwd=root,check=True)
print('All numerical outputs regenerated and verified.',flush=True)
