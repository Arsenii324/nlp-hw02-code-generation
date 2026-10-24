"""Generate the working and canonical inference notebooks from reviewable cells."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def notebook(path,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.validate(nb)
    nbf.write(nb,ROOT/path)


md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell

notebook('submit/inference.ipynb',[
md("""# NLP HW2 — selected Transformer inference

Load the trained from-scratch encoder–decoder and generate all test rows. This
notebook does **not submit** anything. The JSONL output is an internal artifact;
the assignment has not specified the Telegram bot schema.

On Colab select one T4 and upload `selected_bundle.zip` to `/content`. On Kaggle
attach the private staged bundle and pinned test data; the notebook uses one T4.
The bundle selects greedy or fixed beam4/alpha.6 decoding through its manifest.
It contains source, tokenizer and weights, no training dataset or credentials.
Test prompts never contributed to training, tokenizer fitting or selection.
"""),
code("""from pathlib import Path
import sys, zipfile, importlib.util, subprocess, os, json, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
KAGGLE = Path('/kaggle/working').is_dir()
ROOT = Path('/kaggle/working' if KAGGLE else '/content')
BASE = Path('/kaggle/temp/selected_model' if KAGGLE else '/content/selected_model')
BUNDLE = ROOT/'selected_bundle.zip'
if not BUNDLE.exists() and KAGGLE:
    candidates = list(Path('/kaggle/input').rglob('selected_bundle.bin'))
    assert len(candidates) == 1, 'Attach exactly one staged selected bundle.'
    BUNDLE = candidates[0]
if not BUNDLE.exists(): BUNDLE = ROOT/'baseline_bundle.zip'
assert BUNDLE.is_file(), 'Upload selected_bundle.zip or attach the staged Kaggle dataset.'
def safe_extract(archive, destination):
    names = archive.namelist()
    assert len(names) == len(set(names)) and archive.testzip() is None
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    archive.extractall(destination)
def read_bundle_manifest(archive, bundle_digest):
    if 'bundle_manifest.json' in archive.namelist():
        return json.loads(archive.read('bundle_manifest.json'))
    # The preserved B001 bundle predates manifests. Trust only its measured archive.
    assert bundle_digest == 'e0943c15c4f1f669f6331eb4a072afef12854bfdbc14c02588fc91234fb7efa3', 'Unknown manifestless bundle'
    return {'file_sha256':{n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()},
            'training_source_sha256':hashlib.sha256(archive.read('baseline_source.zip')).hexdigest()}
with zipfile.ZipFile(BUNDLE) as z:
    bundle_manifest = read_bundle_manifest(z, hashlib.sha256(BUNDLE.read_bytes()).hexdigest())
    metadata_names = {'bundle_manifest.json'} if 'bundle_manifest.json' in z.namelist() else set()
    assert set(z.namelist()) == set(bundle_manifest['file_sha256']) | metadata_names
    for name, digest in bundle_manifest['file_sha256'].items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    safe_extract(z, BASE)
assert hashlib.sha256((BASE/'baseline_source.zip').read_bytes()).hexdigest() == bundle_manifest['training_source_sha256']
missing = [x for x in ['tokenizers', 'pyarrow'] if importlib.util.find_spec(x) is None]
assert not (KAGGLE and missing), 'Kaggle preflight requires installed dependencies; no GPU-time installation.'
if missing: subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
SOURCE = BASE/'source'
with zipfile.ZipFile(BASE/'baseline_source.zip') as z: safe_extract(z, SOURCE)
DECODING = bundle_manifest.get('inference', {'decoder':'greedy', 'batch_size':256})
assert DECODING['decoder'] in ('greedy', 'beam') and DECODING['batch_size'] == 256
if DECODING['decoder'] == 'beam':
    assert DECODING['beam_width'] == 4 and DECODING['length_penalty'] == .6
    with zipfile.ZipFile(BASE/'inference_addon.zip') as z:
        expected = bundle_manifest['inference_source_sha256']
        assert set(z.namelist()) == set(expected) == {'src/nlp_hw02/beam.py', 'src/nlp_hw02/beam_inference.py'}
        for name, digest in expected.items(): assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
        safe_extract(z, SOURCE)
sys.path.insert(0, str(SOURCE/'src'))
import torch, time
from functools import partial
torch.set_num_threads(2)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0), 'Timing requires exactly one T4.'
print({'torch':str(torch.__version__), 'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),
       'bundle_sha256':hashlib.sha256(BUNDLE.read_bytes()).hexdigest(), 'decoding':DECODING})
"""),
code("""from tokenizers import Tokenizer
from nlp_hw02.model import load_model
from nlp_hw02.data import fetch
from nlp_hw02.inference import generate_questions
import pyarrow.parquet as pq
if DECODING['decoder'] == 'beam':
    from nlp_hw02.beam_inference import generate_beam_questions
    generate = partial(generate_beam_questions, beam_size=DECODING['beam_width'], length_penalty=DECODING['length_penalty'])
else: generate = generate_questions
load_start = time.perf_counter()
model, checkpoint = load_model(BASE/'model.pt')
tokenizer = Tokenizer.from_file(str(BASE/'tokenizer.json'))
torch.cuda.synchronize()
print('Checkpoint load seconds:', time.perf_counter()-load_start)
manifest = json.loads((SOURCE/'assignment/dataset_manifest.json').read_text())
expected_test = next(r['sha256'] for r in manifest['files'] if r['path'] == 'data/test/test.parquet')
if KAGGLE:
    candidates = list(Path('/kaggle/input').rglob('test.parquet'))
    candidates = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == expected_test]
    assert len(candidates) == 1, 'Attach exactly one pinned test Parquet.'
    TEST = candidates[0]
else:
    DATA = BASE/'data'
    fetch(manifest, DATA, ['data/test/test.parquet'])
    TEST = DATA/'data/test/test.parquet'
assert hashlib.sha256(TEST.read_bytes()).hexdigest() == expected_test
questions = pq.read_table(TEST, columns=['question'])['question'].to_pylist()
assert len(questions) == 1919
print('Test rows:', len(questions), 'checkpoint step:', checkpoint['step'])
"""),
md("""## Manifest-selected generation and timing

Batch 256, at most 192 generated tokens. Beam uses width4/length penalty .6;
older greedy bundles remain supported. Timing includes tokenization, transfers,
encoder, cached decoder and detokenization; excludes loading, warm-up and writing.
GPU work is synchronized around the timer. The warm-up prompt is from held-out
training questions. Do not tune decoding on test prompts. Validation proxy
agreement does not establish official pass@1.
"""),
code("""validation = json.loads((BASE/'validation.json').read_text())
generate(model, tokenizer, [validation[0]['question']], batch_size=DECODING['batch_size'])
predictions, timing = generate(model, tokenizer, questions, batch_size=DECODING['batch_size'])
assert len(predictions) == len(questions)
timing.update(gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(), selected_step=checkpoint['step'],
    decoder=DECODING['decoder'], test_parquet_sha256=expected_test,
    checkpoint_sha256=hashlib.sha256((BASE/'model.pt').read_bytes()).hexdigest(),
    bundle_sha256=hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
    training_source_sha256=bundle_manifest['training_source_sha256'],
    inference_source_sha256=bundle_manifest.get('inference_source_sha256', {}))
timing['meets_180_seconds_on_one_T4'] = timing['seconds'] <= 180
assert timing['meets_180_seconds_on_one_T4'], 'Measured generation exceeds the course budget.'
print(json.dumps(timing, indent=2))
OUT = ROOT/'canonical-inference' if KAGGLE else BASE/'inference_output'
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'test_timing.json').write_text(json.dumps(timing, indent=2))
with (OUT/'test_predictions.jsonl').open('w') as f:
    for i, prediction in enumerate(predictions):
        f.write(json.dumps({'row_index':i, 'code':prediction}, ensure_ascii=False)+'\\n')
print('Saved internal predictions:', OUT/'test_predictions.jsonl')
"""),
md("""Keep these outputs for review. No course submission is performed. The required
final PDF report is a separate later task.""")
])

notebook('notebooks/work.ipynb',[
md('''# NLP HW2 — baseline work notebook

The baseline is implemented in `src/nlp_hw02/`, with Colab orchestration in
`scripts/colab_job.py`. See `experiments/LOG.md` for hypotheses and measured
results, `reports/OPERATIONS.md` for problems and fixes, and
`docs/guides/COLAB_CLI.md` for agent operations. This notebook is for inspecting
the recovered run and manually launching reproducible stages on a cloud VM.
'''),
code('''from pathlib import Path
import json, sys
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
ARTIFACTS = ROOT/'artifacts/kaggle/b011-b012-v1/run-b012'
'''),
code('''for name in ['run_config.json', 'data_stats.json', 'metrics.json', 'test_timing.json']:
    path = ARTIFACTS/name if name != 'test_timing.json' else ROOT/'reports/evidence/b012_canonical_test_timing.json'
    print(name)
    print(json.dumps(json.loads(path.read_text()), indent=2) if path.exists() else 'Not recovered yet')
'''),
code('''path = ARTIFACTS/'validation_predictions.jsonl'
if path.exists():
    with path.open() as f:
        for _, line in zip(range(3), f):
            row = json.loads(line)
            print('QUESTION:', row['question'])
            print('REFERENCE:', row['code'])
            print('PREDICTION:', row['prediction'])
'''),
md('''## Reproduce the original Colab baseline

Upload and unpack `artifacts/baseline_source.zip`, then run (in the VM):

```bash
python /content/nlp-hw02/source/scripts/check_baseline.py
python /content/nlp-hw02/source/scripts/colab_job.py prepare
python /content/nlp-hw02/source/scripts/colab_job.py train --seconds 3600
python /content/nlp-hw02/source/scripts/colab_job.py test
```

For `check_baseline.py`, put unpacked `src` on PYTHONPATH or use the documented
`runpy` pattern. The stages insert source themselves. Cloud data downloads are
bounded and pinned. Use the CLI guide's background launch pattern for long jobs.
Do not synchronize the linked local HW1/HW2 venv. No generated code is executed.
The selected representative-data B012 run is reproduced from the frozen
resident width/depth manifest and CPU cache described in `docs/guides/KAGGLE_CLI.md`;
the original four-shard Colab stages above describe B001, not B012.
''')
])
print('Wrote and validated work/inference notebook structures')
