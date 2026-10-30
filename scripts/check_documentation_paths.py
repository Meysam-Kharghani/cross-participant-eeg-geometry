#!/usr/bin/env python3
from pathlib import Path
import re, sys
ROOT=Path(__file__).resolve().parents[1]
problems=[]
for md in ROOT.rglob("*.md"):
    text=md.read_text(encoding="utf-8",errors="ignore")
    for token in re.findall(r"`([^`]+)`", text):
        if not re.search(r"[/\\]|\.(?:py|csv|tsv|json|ya?ml|txt|md|sh|gz|png|svg|xlsx)$", token, re.I):
            continue
        if token.startswith(("http://","https://")) or any(c in token for c in [" ","*","<",">","|","="]):
            continue
        if not ((ROOT/token).exists() or (md.parent/token).exists()):
            problems.append((md.relative_to(ROOT).as_posix(),token))
if problems:
    for md,token in problems: print(f"FAIL {md}: missing referenced path `{token}`")
    sys.exit(1)
print("PASS: documentation path references resolve.")
