import json

r = json.load(open("runs/sweep_rm3.json"))
r.sort(key=lambda x: -x["ndcg@10"])
for x in r[:15]:
    print(f"docs={x['fb_docs']:<4} terms={x['fb_terms']:<3} "
          f"lam={x['lam']:<5} {x['ndcg@10']:.4f}")