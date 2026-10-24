import os, sys, json, time, math, hashlib, zipfile
from pathlib import Path
from types import SimpleNamespace

BASE = Path('/workspace/nlp-job-20261023a')
EVAL_DIR = BASE / 'antigravity-gelu512-long262k-eval'
SRC_DIR = BASE / 'antigravity-gelu768-fullpop531k-eval/source'

sys.path.insert(0, str(SRC_DIR / 'src'))

import torch
from nlp_hw02.model import load_model
from nlp_hw02.tokenizer import Tokenizer, BOS, EOS, decode
from nlp_hw02.inference import padded_tokens
from nlp_hw02.arithmetic import evaluate_code

def equal(a, b):
    return a == b if isinstance(a, int) and isinstance(b, int) else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

def select_plurality(codes, scores):
    if not codes or len(codes) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Nonempty candidates and finite scores required")
    groups = []
    supported = 0
    for i, code in enumerate(codes):
        result = evaluate_code(code)
        if not result.supported:
            continue
        supported += 1
        group = next((g for g in groups if equal(g["value"], result.value)), None)
        if group is None:
            group = {"value": result.value, "indices": []}
            groups.append(group)
        group["indices"].append(i)
    if not groups:
        return 0, {"supported_candidates": 0, "winning_votes": 0, "numeric_groups": 0}
    best = max(groups, key=lambda g: (len(g["indices"]), max(scores[i] for i in g["indices"])))
    index = max(best["indices"], key=lambda i: scores[i])
    return index, {"supported_candidates": supported, "winning_votes": len(best["indices"]), "numeric_groups": len(groups)}

def generate_candidates(model, tok, questions, count):
    args = SimpleNamespace(temperature=0.8)
    group = [{'question': q} for q in questions]
    src = padded_tokens(tok, [r['question'] for r in group], model.cfg.max_source).cuda().repeat_interleave(count, dim=0)
    with torch.autocast('cuda', dtype=torch.float16):
        memory, valid = model.encode(src)
        cache = model.new_cache(memory)
        token = torch.full((len(group)*count,), BOS, device='cuda', dtype=torch.long)
        done = torch.zeros_like(token, dtype=torch.bool)
        score = torch.zeros_like(token, dtype=torch.float32)
        length = torch.zeros_like(token)
        history = []
        for t in range(model.cfg.max_target):
            logits = model.decode_step(token, t, cache, valid).float() / args.temperature
            values, indices = logits.topk(min(50, logits.shape[-1]), dim=-1)
            logprob = values.log_softmax(-1)
            draw = torch.multinomial(logprob.exp(), 1)
            token = indices.gather(1, draw).squeeze(1)
            score += torch.where(done, 0., logprob.gather(1, draw).squeeze(1))
            length += (~done).long()
            token = torch.where(done, torch.full_like(token, EOS), token)
            history.append(token)
            done |= token.eq(EOS)
            if done.all():
                break
        ids = torch.stack(history, 1).cpu().tolist()
        scores = (score / ((5 + length.float()) / 6).pow(0.6)).cpu().tolist()
    records = []
    for j in range(len(group)):
        codes = [decode(tok, ids[j*count+k]) for k in range(count)]
        weights = scores[j*count:(j+1)*count]
        caps = [EOS not in ids[j*count+k] for k in range(count)]
        lengths = [row.index(EOS) if EOS in row else len(row) for row in ids[j*count:(j+1)*count]]
        records.append({'candidates': codes, 'scores': weights, 'capped': caps, 'generated_lengths': lengths})
    return records

def run():
    ptr_path = BASE / 'wide-fullpopulation-run-c/training/latest_checkpoint.json'
    ptr = json.loads(ptr_path.read_text())
    slot = ptr.get('slot', 'durable-slot1')
    model_path = BASE / f'wide-fullpopulation-run-c/training/{slot}/model.pt'
    tok_path = BASE / 'reference-gelu768-final400k-20261024a/projection/tokenizer.json'
    q_path = BASE / 'antigravity-gelu768-fullpop531k-eval/test_questions.json'
    
    print(f'Loading GELU512 model for test generation from {model_path} (slot: {slot})')
    t0 = time.perf_counter()
    model, ckpt = load_model(str(model_path), device='cuda:0')
    tok = Tokenizer.from_file(str(tok_path))
    questions = json.loads(q_path.read_text())
    print(f'Model loaded in {time.perf_counter() - t0:.2f}s. Test questions: {len(questions)}')
    
    count = 8
    label = 'test_n8'
    pool_path = EVAL_DIR / f'{label}_candidates.jsonl'
    prediction_path = EVAL_DIR / f'{label}_predictions.jsonl'
    
    torch.cuda.synchronize()
    began = time.perf_counter()
    torch.manual_seed(20261002)
    torch.cuda.manual_seed_all(20261002)
    torch.cuda.reset_peak_memory_stats()
    
    completed = 0
    with torch.inference_mode(), open(pool_path, 'w', encoding='utf-8') as pool, open(prediction_path, 'w', encoding='utf-8') as preds:
        for offset in range(0, len(questions), 32):
            group = questions[offset:offset+32]
            records = generate_candidates(model, tok, group, count)
            for j, (question, record) in enumerate(zip(group, records)):
                index, info = select_plurality(record['candidates'], record['scores'])
                row = {
                    'row_index': offset + j,
                    'question_sha256': hashlib.sha256(question.encode()).hexdigest(),
                    'samples': count,
                    **record,
                    'selected_index': index,
                    **info
                }
                pool.write(json.dumps(row, ensure_ascii=False) + '\n')
                preds.write(json.dumps({'code': record['candidates'][index]}, ensure_ascii=False) + '\n')
                completed += 1
            if (completed % 256) == 0 or completed == len(questions):
                print(f'Progress: {completed}/{len(questions)} ({time.perf_counter()-began:.1f}s)')
                pool.flush()
                preds.flush()
                
    torch.cuda.synchronize()
    total_time = time.perf_counter() - began
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f'Done! Total time: {total_time:.2f}s, Peak memory: {peak_mem:.1f} MB')
    
    timing = {
        'rows': completed,
        'samples_per_question': count,
        'total_candidates': completed * count,
        'complete_path_seconds': total_time,
        'peak_allocated_mb': peak_mem,
        'checkpoint_step': ckpt.get('step'),
        'checkpoint_val_loss': ckpt.get('validation_loss'),
    }
    with open(EVAL_DIR / f'{label}_timing.json', 'w') as f:
        json.dump(timing, f, indent=2)

if __name__ == '__main__':
    run()
