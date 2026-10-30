#!/usr/bin/env python3
from pathlib import Path
import ast,sys
r=Path(__file__).resolve().parents[1]; n=0
for p in r.rglob("*.py"):
    if any(x in p.parts for x in {".git","generated_figures"}): continue
    try: ast.parse(p.read_text(encoding="utf-8"),filename=str(p)); n+=1
    except Exception as e: print(f"FAIL: {p.relative_to(r)}: {e}"); sys.exit(1)
print(f"PASS: syntax checked for {n} Python files.")
