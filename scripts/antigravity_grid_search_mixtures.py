import json, math, hashlib, functools
from pathlib import Path
from nlp_hw02.arithmetic import evaluate_code as raw_evaluate_code

@functools.lru_cache(maxsize=100000)
def evaluate_code(code):
    return raw_evaluate_code(code)

def equal(a, b):
    return a == b if isinstance(a, int) and isinstance(b, int) else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

def select_plurality(codes, scores):
    groups = []
    for i, code in enumerate(codes):
        result = evaluate_code(code)
        if not result.supported:
            continue
        group = next((g for g in groups if equal(g["value"], result.value)), None)
        if group is None:
            group = {"value": result.value, "indices": []}
            groups.append(group)
        group["indices"].append(i)
    if not groups:
        return 0
    best = max(groups, key=lambda g: (len(g["indices"]), max(scores[i] for i in g["indices"])))
    return max(best["indices"], key=lambda i: scores[i])

def main():
    val_768_fp = [json.loads(line) for line in open("artifacts/antigravity_gelu768_fullpop531k_eval/validation_n8_candidates.jsonl")]
    val_512_400k = [json.loads(line) for line in open("artifacts/paid-wide-validation-20261024a/gelu-400000-result/reference/validation_n8_candidates.jsonl")]
    val_512_new = [json.loads(line) for line in open("artifacts/antigravity_gelu512_long262k_eval/validation_n8_candidates.jsonl")]

    refs = json.loads(Path("reports/evidence/primary_reference_consensus.json").read_text())["groups_by_question_sha256"]
    questions = json.loads(Path("artifacts/paid-gelu768-validation-20261024a/final400k/inputs/validation_questions.json").read_text())
    qhashes = [hashlib.sha256(q.encode()).hexdigest() for q in questions]

    solvable_indices = [i for i, q in enumerate(qhashes) if refs[q]["status"] == "supported_consensus"]
    print(f"Solvable rows: {len(solvable_indices)}")

    # Pre-warm cache for all candidates
    for pool in [val_768_fp, val_512_400k, val_512_new]:
        for row in pool:
            for code in row["candidates"]:
                evaluate_code(code)
    print("Pre-warmed AST evaluation cache.")

    # Standalone baseline scores:
    for name, cands_list in [
        ("768_FP (N=8)", [(val_768_fp[i]["candidates"][:8], val_768_fp[i]["scores"][:8]) for i in range(len(questions))]),
        ("512_400k (N=8)", [(val_512_400k[i]["candidates"][:8], val_512_400k[i]["scores"][:8]) for i in range(len(questions))]),
        ("512_new (N=8)", [(val_512_new[i]["candidates"][:8], val_512_new[i]["scores"][:8]) for i in range(len(questions))]),
    ]:
        c = sum(equal(evaluate_code(cands[select_plurality(cands, sc)]).value, refs[qhashes[i]]["consensus_value"])
                for i, (cands, sc) in enumerate(cands_list) if i in solvable_indices and evaluate_code(cands[select_plurality(cands, sc)]).supported)
        print(f"Standalone {name:20s}: {c}/{len(solvable_indices)} ({c/len(solvable_indices)*100:.2f}%)")

    print("\nRunning grid search across mixture combinations...")
    results = []
    for n_768 in [6, 8]:
        for n_400k in [0, 1, 2, 3, 4]:
            for n_new in [0, 1, 2, 3, 4, 6]:
                if n_400k + n_new == 0:
                    continue
                # Est. latency constraint on single Tesla T4:
                # Load models: 16.7s
                # Inference: 768 is ~13.1s/sample, 512 is ~6.5s/sample
                # Total must be <= 175s for safe <=180s bound
                lat = 16.7 + n_768 * 13.125 + (n_400k + n_new) * 6.5 + 2.5
                if lat > 175.0:
                    continue
                correct = 0
                for i in solvable_indices:
                    q = qhashes[i]
                    ref = refs[q]
                    cands = val_768_fp[i]["candidates"][:n_768] + val_512_400k[i]["candidates"][:n_400k] + val_512_new[i]["candidates"][:n_new]
                    scores = val_768_fp[i]["scores"][:n_768] + val_512_400k[i]["scores"][:n_400k] + val_512_new[i]["scores"][:n_new]
                    idx = select_plurality(cands, scores)
                    res = evaluate_code(cands[idx])
                    if res.supported and equal(res.value, ref["consensus_value"]):
                        correct += 1
                results.append((correct, lat, n_768, n_400k, n_new))

    results.sort(key=lambda x: (-x[0], x[1]))
    print(f"\nTop 15 Mixture Configurations by Validation Score:")
    print(f"{'Configuration':45s} | {'Score':12s} | {'T4 Est. Latency':15s}")
    print("-" * 78)
    for score, lat, n_768, n_400k, n_new in results[:15]:
        pct = score / len(solvable_indices) * 100
        cfg = f"768(N={n_768}) + 512_old(N={n_400k}) + 512_new(N={n_new})"
        print(f"{cfg:45s} | {score}/{len(solvable_indices)} ({pct:.2f}%) | {lat:.1f}s")

if __name__ == "__main__":
    main()
