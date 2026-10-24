"""Retokenize actual retained programs using existing anaconda tokenizers.

No model/project import, generation, program execution or environment changes.
Run /Users/a2mogus/anaconda3/bin/python scripts/audit_portable_crossscore_lengths.py.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tokenizers


root = Path(__file__).resolve().parents[1]
art = root / 'artifacts/kaggle/portable-proof-crossscore-v1'
tokpath = root / '.worktrees/gated-adjacent-warmstart-20261022/artifacts/kaggle/tools-gated-b034-20261022b/parent-b034/tokenizer.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(tokpath) == '0fa06854a98e82cd520c13f4c670040c772fec38f2cd2cacb5de11130bc42d74'
out = root / 'reports/evidence/portable_crossscore_canonical_lengths_20261023.json'
assert not out.exists(), 'Fresh evidence required'
load = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
pool = load(art / 'crossscore/candidate_pool.jsonl.bin')
scores = [load(art / f'crossscore/{name}_scores.jsonl') for name in ('b033', 'b034')]
tok = tokenizers.Tokenizer.from_file(str(tokpath))
count = 0
excluded = []
for row in pool:
    for j, code in enumerate(row['candidates']):
        length = len(tok.encode(code).ids) + 1  # Original EOS2 appended, never truncate.
        assert all(s[row['row_index']]['candidates'][j]['length_including_eos'] == length for s in scores)
        count += 1
        if length > 192:
            excluded.append({'row_index': row['row_index'], 'candidate_index': j, 'length': length})
assert count == 10430 and len(excluded) == 2
report = {'utc': datetime.now(timezone.utc).isoformat(), 'scope': 'All10430 actual unique generated-program strings, exact original tokenizer, canonical encode plus EOS without truncation. No model/project imports or generated-code execution.',
          'interpreter': sys.executable, 'tokenizers_version': tokenizers.__version__, 'tokenizer_sha256': sha(tokpath),
          'script_sha256': sha(Path(__file__)), 'pool_sha256': sha(art / 'crossscore/candidate_pool.jsonl.bin'),
          'all_canonical_lengths_match_both_actual_score_arrays': True, 'unique_programs': count,
          'scoreable': count - len(excluded), 'excluded': excluded,
          'case_reuse': 'Shared .venv lacks tokenizers; existing anaconda interpreter provides it. No install or environment modification; see existing C40 case.'}
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
