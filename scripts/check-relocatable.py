"""Fail if any Mach-O inside the app links against Homebrew or a home directory (usage: check-relocatable.py App)."""
import pathlib
import subprocess
import sys

app = pathlib.Path(sys.argv[1])
bad = []
count = 0
for f in app.rglob("*"):
    if not f.is_file() or f.is_symlink():
        continue
    if f.suffix not in (".so", ".dylib", "") and "/bin/" not in str(f):
        continue
    try:
        head = f.open("rb").read(4)
    except OSError:
        continue
    if head not in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf"):
        continue
    count += 1
    out = subprocess.run(["otool", "-L", str(f)], capture_output=True, text=True).stdout.splitlines()[1:]
    for line in out:
        dep = line.strip().split(" (")[0]
        if pathlib.Path(dep).name == f.name:
            continue  # LC_ID_DYLIB: the file's own install name, not a dependency
        if dep.startswith("/opt/homebrew") or dep.startswith("/Users/") or dep.startswith("/usr/local"):
            bad.append(f"{f.relative_to(app)} -> {dep}")
print(f"   scanned {count} Mach-O files")
if bad:
    print("NOT relocatable:\n  " + "\n  ".join(bad))
    sys.exit(1)
