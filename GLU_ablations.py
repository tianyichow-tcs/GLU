import json
import numpy as np
import pandas as pd
from collections import defaultdict
from glob import glob
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

K = 10
EPS = 1e-6


def softplus(x):
    return np.logaddexp(0.0, x)


def recompute_she_R_mean(df, k=K, use_softplus=False, eps=EPS, dynamic_k=False, au_only=False):
    results = []
    for _, row in df.iterrows():
        top100 = json.loads(row["top100_logits"])
        T = len(top100)
        s_tilde = float(row["S_tilde"])

        k1 = k / (1.0 + s_tilde) if dynamic_k else float(k)
        k_int = max(1, int(k1))
        k_w = min(k_int, T)

        shetoku = []
        for tok_logits in top100:
            topk = np.array(tok_logits[:k_int], dtype=np.float64)
            if use_softplus:
                alpha = softplus(topk)
                eu = k1 / (alpha + eps).sum()
            else:
                alpha = np.maximum(topk, 0.0)
                eu = k1 / (alpha + 1.0).sum()
            probs = softmax(topk)
            se = -np.sum(probs * np.log2(probs.clip(1e-12)))
            shetoku.append(-se if au_only else -(se * eu))

        results.append(np.sort(shetoku)[:k_w].mean())
    return results


# Load files, skip incomplete ones by comparing row counts within each dataset
all_files = sorted(glob("output/*-unified-uq-full.csv"))
dataset_files = defaultdict(list)
for f in all_files:
    name = f.split("/")[-1].replace("-unified-uq-full.csv", "")
    dataset = name.split("-", 1)[1]
    dataset_files[dataset].append(f)

complete_files = []
for dataset, files in sorted(dataset_files.items()):
    counts = {f: len(pd.read_csv(f, usecols=["id"])) for f in files}
    max_count = max(counts.values())
    for f, count in counts.items():
        name = f.split("/")[-1].replace("-unified-uq-full.csv", "")
        if count < max_count:
            print(f"[SKIP] {name}: {count}/{max_count} samples")
        else:
            complete_files.append(f)

print()

# (column, label) — label is what prints in the header
ABLATIONS = [
    ("GLU",               "GLU"),               # baseline: (1+S_tilde_mean) * she_R_mean
    ("GLU_rawS_mean",     "GLU_rawS_mean"),      # S_alpha = mean across layers
    ("GLU_rawS_best",     "GLU_rawS_best"),      # S_alpha = max across layers
    ("GLU_stilde_best",   "GLU_stilde_best"),    # S_tilde from best layer instead of mean
    ("GLU_softplus",      "GLU_softplus"),       # softplus+eps evidence instead of relu+1
    ("GLU_dynK",          "GLU_dynK"),           # k shrinks with complexity: k1=K/(1+S_tilde)
    ("GLU_AU",            "GLU_AU"),             # drop EU, use Shannon entropy only
]

header = f"{'dataset':35s}" + "".join(f"  {lbl:>14s}" for _, lbl in ABLATIONS)
print(header)
print("-" * len(header))

for f in sorted(complete_files):
    name = f.split("/")[-1].replace("-unified-uq-full.csv", "")
    df = pd.read_csv(f)

    s_alpha_best  = df["S_alpha_per_layer"].apply(lambda x: max(json.loads(x)))
    s_tilde_best  = s_alpha_best / (1.0 + np.log(df["T"]))

    df["GLU"]             = (1.0 + df["S_tilde"])  * df["she_R_mean"]
    df["GLU_rawS_mean"]   = (1.0 + df["S_alpha"])   * df["she_R_mean"]
    df["GLU_rawS_best"]   = (1.0 + s_alpha_best)    * df["she_R_mean"]
    df["GLU_stilde_best"] = (1.0 + s_tilde_best)    * df["she_R_mean"]
    df["she_R_mean_sp"] = recompute_she_R_mean(df, use_softplus=True)
    df["GLU_softplus"]  = (1.0 + df["S_tilde"])  * df["she_R_mean_sp"]
    df["she_R_mean_dk"] = recompute_she_R_mean(df, dynamic_k=True)
    df["GLU_dynK"]      = (1.0 + df["S_tilde"])  * df["she_R_mean_dk"]
    df["she_R_mean_au"] = recompute_she_R_mean(df, au_only=True)
    df["GLU_AU"]        = (1.0 + df["S_tilde"])  * df["she_R_mean_au"]

    row_out = f"{name:35s}"
    for col, _ in ABLATIONS:
        sub = df.dropna(subset=["label", col])
        auroc = roc_auc_score(sub["label"], sub[col])
        row_out += f"  {auroc:>14.4f}"
    print(row_out)
