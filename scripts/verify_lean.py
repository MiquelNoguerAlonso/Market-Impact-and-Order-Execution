#!/usr/bin/env python3
"""Build the pinned Lean companion and record its proof-dependency audit.

Run from any directory after fetching the Mathlib cache. --clean removes only
this project's generated .lake/build directory before rebuilding the proofs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "lean"
AUDIT = ROOT / "audit"
MATHLIB_REV = "c44e0c8ee63ca166450922a373c7409c5d26b00b"
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
THEOREMS = [
    "MarketImpact.loop_cost_eq_area_pairing",
    "MarketImpact.loop_cost_one_asset",
    "MarketImpact.RayData.ray_sign_conditions",
    "MarketImpact.value_convex",
    "MarketImpact.zero_cost_isGLB",
    "MarketImpact.zero_cost_nonempty",
]


def run(args, cwd=PROJECT, log=None):
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if log:
        (AUDIT / log).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args}\n{result.stdout}")
    return result.stdout.strip()


def code_only(source):
    """Remove Lean's nested block comments, line comments and string literals."""
    out, i, depth = [], 0, 0
    while i < len(source):
        if source.startswith("/-", i):
            depth += 1
            i += 2
        elif depth and source.startswith("-/", i):
            depth -= 1
            i += 2
            out.append(" ")
        elif depth:
            i += 1
        elif source.startswith("--", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
        elif source[i] == '"':
            i += 1
            while i < len(source):
                if source[i] == "\\":
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            out.append(" ")
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    AUDIT.mkdir(exist_ok=True)
    toolchain = (PROJECT / "lean-toolchain").read_text().strip()
    if toolchain != "leanprover/lean4:v4.19.0":
        raise RuntimeError(f"Unexpected toolchain: {toolchain}")
    mathlib = PROJECT / ".lake/packages/mathlib"
    revision = run(["git", "rev-parse", "HEAD"], cwd=mathlib)
    if revision != MATHLIB_REV:
        raise RuntimeError(f"Unexpected Mathlib revision: {revision}")
    run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=mathlib)
    sources = sorted((PROJECT / "MarketImpactCore").glob("*.lean"))
    prohibited = re.compile(r"\b(?:sorry|admit|axiom|native_decide|unsafe|implemented_by)\b")
    for source in sources:
        if prohibited.search(code_only(source.read_text())):
            raise RuntimeError(f"Prohibited proof-admission or trust token: {source.name}")
    if args.clean:
        shutil.rmtree(PROJECT / ".lake/build", ignore_errors=True)
    version = run(["lake", "env", "lean", "--version"])
    if "version 4.19.0," not in version:
        raise RuntimeError(f"Unexpected compiler: {version}")
    build = run(["lake", "build"], log="lean_build.log")
    if "warning:" in build.lower() or "error:" in build.lower():
        raise RuntimeError(f"Build contains diagnostics:\n{build}")
    output = run(["lake", "env", "lean", "MarketImpactCore/Audit.lean"],
                 log="lean_axioms.log")
    axioms = {}
    for theorem in THEOREMS:
        match = re.search(r"'" + re.escape(theorem) +
                          r"' depends on axioms:\s*\[([^]]*)\]", output)
        if match:
            found = {s.strip() for s in match[1].split(",") if s.strip()}
        elif f"'{theorem}' does not depend on any axioms" in output:
            found = set()
        else:
            raise RuntimeError(f"Missing axiom report for {theorem}")
        if not found <= ALLOWED:
            raise RuntimeError(f"Unexpected axioms for {theorem}: {found - ALLOWED}")
        axioms[theorem] = sorted(found)
    inputs = sources + [PROJECT / name for name in
                        ["lakefile.lean", "lake-manifest.json", "lean-toolchain"]]
    report = {
        "status": "passed",
        "lean_version": version,
        "toolchain": toolchain,
        "mathlib_commit": revision,
        "mathlib_tracked_sources_unmodified": True,
        "clean_project_build": args.clean,
        "compile_command": "lake build",
        "audit_command": "lake env lean MarketImpactCore/Audit.lean",
        "build_warnings": 0,
        "build_errors": 0,
        "prohibited_active_source_tokens": [],
        "allowed_axioms": sorted(ALLOWED),
        "reported_axioms": axioms,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in inputs},
        "scope": "Finite-grid loop cost, scalar ray sign conditions, convex value at feasible "
                 "quantities, and a zero-cost consistency example. The continuum theorem, "
                 "analytic-functional assumptions, attainment, uniqueness, KKT conditions "
                 "and execution certificates are not formalized here.",
        "delivering_environment_note": "audit/environment/README.md",
    }
    (AUDIT / "lean_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Lean build and axiom audit passed for {len(THEOREMS)} declarations.")


if __name__ == "__main__":
    main()
