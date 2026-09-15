#!/usr/bin/env python3
r"""
Forward-reference audit for the monograph.

Reads main.tex (following \input/\include), records the source order of every
\label, then reports every \ref/\eqref/\cref whose target is defined later in
the document, sorted by how far ahead it points.

A reference that points one or two sections ahead is normal. One that points
across a part boundary is the §18 -> Theorem 31.2 problem: the reader is asked
to accept a result they cannot yet see. Those are the ones to restate
self-containedly or move.

Usage:  python3 fwdref_audit.py main.tex
"""

import re
import sys
from pathlib import Path

INPUT_RE = re.compile(r'\\(?:input|include)\s*\{([^}]+)\}')
LABEL_RE = re.compile(r'\\label\s*\{([^}]+)\}')
REF_RE = re.compile(r'\\(?:c|C|eq|auto|name)?ref\s*\*?\s*\{([^}]+)\}')
SEC_RE = re.compile(r'\\(part|chapter|section|subsection)\s*\*?\s*\{')


def flatten(path, seen=None):
    """Yield (file, lineno, text) in source order, following \\input."""
    seen = seen or set()
    path = Path(path)
    if path in seen or not path.exists():
        return
    seen.add(path)
    for lineno, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
        if line.lstrip().startswith('%'):
            continue
        m = INPUT_RE.search(line)
        if m:
            child = m.group(1)
            for cand in (Path(child), Path(child + '.tex'), path.parent / child,
                         path.parent / (child + '.tex')):
                if cand.exists():
                    yield from flatten(cand, seen)
                    break
            continue
        yield (str(path), lineno, line)


def main(root):
    lines = list(flatten(root))
    if not lines:
        sys.exit(f'no source found at {root}')

    labels = {}      # label -> position
    sec_at = {}      # position -> (level, running section index)
    refs = []        # (position, file, lineno, label)

    sec_index = 0
    for pos, (fname, lineno, text) in enumerate(lines):
        if SEC_RE.search(text):
            sec_index += 1
        sec_at[pos] = sec_index
        for lab in LABEL_RE.findall(text):
            labels.setdefault(lab, pos)
        for group in REF_RE.findall(text):
            for lab in group.split(','):
                refs.append((pos, fname, lineno, lab.strip()))

    forward, missing = [], []
    for pos, fname, lineno, lab in refs:
        if lab not in labels:
            missing.append((fname, lineno, lab))
            continue
        tgt = labels[lab]
        if tgt > pos:
            gap = sec_at[tgt] - sec_at[pos]
            forward.append((gap, fname, lineno, lab, sec_at[pos], sec_at[tgt]))

    forward.sort(reverse=True)

    print(f'{len(lines)} source lines, {len(labels)} labels, {len(refs)} refs\n')

    print(f'FORWARD REFERENCES ({len(forward)}), worst first')
    print('-' * 68)
    for gap, fname, lineno, lab, here, there in forward:
        flag = '  <-- CROSSES FAR AHEAD' if gap >= 5 else ''
        print(f'  +{gap:>3} sections   {lab:<34} {fname}:{lineno}{flag}')
    if not forward:
        print('  none')

    if missing:
        print(f'\nUNDEFINED LABELS ({len(missing)})')
        print('-' * 68)
        for fname, lineno, lab in missing:
            print(f'  {lab:<34} {fname}:{lineno}')

    far = sum(1 for f in forward if f[0] >= 5)
    print(f'\n{far} reference(s) point five or more sections ahead.')
    return 1 if far else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'main.tex'))
