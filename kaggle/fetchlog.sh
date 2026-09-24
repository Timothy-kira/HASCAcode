#!/bin/bash
# Usage: kaggle/fetchlog.sh <kernel-slug> [tail-chars]  — download a kernel's log and print its stdout/stderr text
k=$1; d=$(dirname "$0")/.logs/$k; rm -rf "$d"; mkdir -p "$d"
kaggle kernels output evelynyang02/$k -p "$d" --file-pattern '.*\.log$' -q >/dev/null 2>&1
python3 - "$d" "${2:-12000}" <<'PY'
import json, glob, re, sys
t = ''.join(x.get('data', '') for x in json.load(open(glob.glob(sys.argv[1] + '/*.log')[0])))
t = '\n'.join(l for l in t.splitlines() if not re.search(r'NbConvert|frozen_modules|Debugger warning|make the debugger|Debugging will proceed|to python to disable|MissingIDField|validate\(nb\)|SyntaxWarning|re\.sub\(', l))
print(t[-int(sys.argv[2]):])
PY
