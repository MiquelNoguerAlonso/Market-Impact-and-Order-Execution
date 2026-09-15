"""Read-only verification of every file listed in the delivered manifest.

Run before rebuilding or regenerating results, which can change file bytes.
The manifest itself is not a cryptographic signature or authenticity proof.
"""
from pathlib import Path
import hashlib
import re
import sys

root = Path(__file__).resolve().parents[1]
failures = []
checked = 0
for line in (root / "MANIFEST_SHA256.txt").read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    match = re.fullmatch(r"([0-9a-f]{64})\s+\*?(.+)", line)
    if not match:
        failures.append("Malformed manifest entry")
        continue
    expected, name = match.groups()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append("Missing or invalid path: " + name)
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    checked += 1
    if actual != expected:
        failures.append("Hash mismatch: " + name)
if failures:
    print("\n".join(failures))
    sys.exit(1)
print(f"All {checked} manifest entries match.")
