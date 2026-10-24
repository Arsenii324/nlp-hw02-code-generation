"""Bounded explicit-version recovery, verification and validation scoring."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PATTERN=r'^(?!.*(?:last_training\.pt|validation_legacy\.json|\.npy|\.npz)$)(?:.*\.(?:json|jsonl|log|ipynb|bin|py)|(?:.*/)?model\.pt)$'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kernel',required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--evidence-prefix',required=True)
    parser.add_argument('--reference-run', action='append', default=[], help='Additional LABEL=/path/to/run numeric diagnostic')
    parser.add_argument('--expected-hashes', default='{}', help='JSON absolute-path SHA256s, rechecked before every command')
    parser.add_argument('--kaggle',default='/Users/a2mogus/anaconda3/bin/kaggle')
    args=parser.parse_args()
    expected_hashes=json.loads(args.expected_hashes)
    def check_protocol():
        for name,digest in expected_hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Recovery protocol/input changed: '+name)
    check_protocol()
    if not re.fullmatch(r'[\w-]+/[\w-]+/[1-9][0-9]*',args.kernel):parser.error('Explicit owner/slug/version required')
    if not re.fullmatch(r'[\w-]+',args.evidence_prefix):parser.error('Use a filename-safe evidence prefix')
    if args.root.exists():raise ValueError('Fresh recovery directory required; never overwrite retained evidence')
    manifest=json.loads(args.manifest.read_text())
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Less than2GiB local space: inspect before recovery')
    args.root.mkdir(parents=True)
    report={'observed_utc':datetime.now(timezone.utc).isoformat(),'kernel':args.kernel,'root':str(args.root),
        'pattern':PATTERN,'optimizer_states_or_arrays_downloaded':False,'commands':[]}
    path=ROOT/'reports/evidence'/f'{args.evidence_prefix}_recovery_commands.json'
    def run(command,timeout=300):
        check_protocol()
        try:
            proc=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},
                capture_output=True,text=True,timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
            report['commands'].append({'command':[str(x) for x in command],'exit':None,'timeout_seconds':timeout,
                'stdout':decoded(exc.stdout),'stderr':decoded(exc.stderr),'failure':'TimeoutExpired'})
            path.write_text(json.dumps(report,indent=2)+'\n')
            raise
        report['commands'].append({'command':[str(x) for x in command],'exit':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
        path.write_text(json.dumps(report,indent=2)+'\n')
        if proc.returncode:raise RuntimeError('Recovery/check failed; inspect retained evidence '+str(path))
        check_protocol()
        return proc.stdout
    run([args.kaggle,'kernels','output',args.kernel,'--path',str(args.root),'--file-pattern',PATTERN,'--page-size','200','--quiet'])
    assert not list(args.root.rglob('last_training.pt')) and not list(args.root.rglob('*.npy'))
    run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/check_recovered_capacity.py'),
        '--root',str(args.root),'--manifest',str(args.manifest),
        '--validation',str(ROOT/'artifacts/kaggle/prepare-v2/cache/validation.json'),
        '--output',str(ROOT/'reports/evidence'/f'{args.evidence_prefix}_recovery_checks.json')])
    score=[sys.executable,str(ROOT/'scripts/score_primary_predictions.py')]
    for lane in manifest['lanes']:
        name=lane['id'].lower()
        for decoder in ('greedy','beam4-alpha06'):
            score+=['--run',f'{name}-{decoder}={args.root}/decoding-{name}/{decoder}']
    score+=['--run',f'b012-beam4={ROOT}/artifacts/kaggle/b011-b012-v1/decoding-b012/beam4-alpha06',
        '--output',str(ROOT/'reports/evidence'/f'{args.evidence_prefix}_numeric.json')]
    for reference in args.reference_run: score += ['--run', reference]
    if manifest.get('profile_decoding'):run(score)
    print(json.dumps({'kernel':args.kernel,'recovery_evidence':str(path),'verification':'passed',
        'proxy_scoring':'passed' if manifest.get('profile_decoding') else 'not_requested',
        'official_pass_at_1':False,'selected_bundle_changed':False,'optimizer_states_local':False}))


if __name__=='__main__':main()
