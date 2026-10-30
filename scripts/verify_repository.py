#!/usr/bin/env python3
from pathlib import Path
import hashlib,sys
r=Path(__file__).resolve().parents[1];m=r/"provenance/CHECKSUMS.sha256";ok=True;n=0
for line in m.read_text().splitlines():
 if not line.strip():continue
 h,rel=line.split("  ",1);p=r/rel; n+=1
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:print("FAIL:",rel);ok=False
print(f"PASS: verified {n} repository files." if ok else "FAIL: repository hashes.");sys.exit(0 if ok else 1)
