"""Bootstrap CI for v2 experiments only."""
import json, sys
sys.path.insert(0, '.')
from evaluation.evaluator import RAGEvaluator

def load_scores(exp_id):
    with open(f'results/{exp_id}/scores_per_question.json') as f:
        return [s['total_score'] for s in json.load(f)]

exps = {}
for e in ['R0','R1','R3','R4','E1','E3','E5']:
    exps[e] = load_scores(e)

pairs = [
    ('R4','R3','Reranker vs Hybrid RRF'),
    ('E1','R3','HyDE vs Hybrid RRF'),
    ('E3','R3','Classifier v2 vs Hybrid RRF'),
    ('E5','R4','Authority vs R4'),
    ('R4','R1','Reranker vs Dense'),
    ('R1','R0','Dense vs No RAG'),
]

print(f"{'Comparison':<35} {'Diff':>8} {'CI Low':>8} {'CI High':>8} {'p-val':>8} {'Sig?':>6}")
print("-" * 77)
for a, b, label in pairs:
    ci = RAGEvaluator.bootstrap_compare(exps[a], exps[b])
    sig = 'YES' if ci['significant'] else 'no'
    diff = ci['observed_diff']
    lo = ci['ci_95'][0]
    hi = ci['ci_95'][1]
    pv = ci['p_value']
    print(f"{label:<35} {diff:>+8.3f} {lo:>8.3f} {hi:>8.3f} {pv:>8.3f} {sig:>6}")
