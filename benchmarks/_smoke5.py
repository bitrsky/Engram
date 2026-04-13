"""Quick: 5 QA pairs with deep search, parallel."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from benchmarks.bench_locomo import evaluate_conversation
from benchmarks.download_datasets import load_locomo
from benchmarks.think_adapter import create_think_fn

data = load_locomo()
think = create_think_fn('anthropic/claude-haiku-3-5-20241022')

sample = data[0]
sample['qa'] = sample['qa'][:5]

t0 = time.time()
results = evaluate_conversation(sample, [3, 5, 10], use_deep_search=True, think_fn=think)
elapsed = time.time() - t0

for r in results:
    print(f"[{r['category']}] R@10={r['recall@10']} MRR={r['mrr']:.2f} | {r['question'][:50]}")

print(f"\n{len(results)} QA pairs in {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s/qa)")
