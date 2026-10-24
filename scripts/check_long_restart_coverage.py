"""Ensure the deployed restart oracle actually exercises both next-run families."""
import json
from pathlib import Path
import subprocess,sys

r=subprocess.run([sys.executable,'scripts/check_xsa_resumption.py'],capture_output=True,text=True,check=True,timeout=90)
events=[json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]
variants={e['variant'] for e in events if e.get('event')=='xsa_variant_resume_passed'}
assert variants=={'ordinary','fixed','learned'},'Runtime oracle omitted ordinary continuation'
assert all(e['legacy_config_checked'] for e in events if e.get('event')=='xsa_variant_resume_passed')
print(json.dumps({'event':'long_restart_coverage_passed','variants':sorted(variants),'underlying_stdout':r.stdout}))
