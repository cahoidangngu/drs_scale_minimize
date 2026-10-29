#!/usr/bin/env python3
"""Write CHECKSUMS.txt and build deliverables.zip (sources/ is never included)."""
import os, hashlib, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); EXP = os.path.dirname(PKG)
files = []
for r, d, fs in os.walk(PKG):
    d[:] = [x for x in d if x != "__pycache__"]
    for f in fs:
        if f != "CHECKSUMS.txt": files.append(os.path.join(r, f))
with open(os.path.join(PKG, "CHECKSUMS.txt"), "w") as o:
    for f in sorted(files): o.write(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {os.path.relpath(f, PKG)}\n")
zp = os.path.join(EXP, "deliverables.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(files + [os.path.join(PKG, "CHECKSUMS.txt")]): z.write(f, os.path.join("deliverables", os.path.relpath(f, PKG)))
print(zp, os.path.getsize(zp), len(files) + 1, "files")
