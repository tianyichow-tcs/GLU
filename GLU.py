import pandas as pd
from sklearn.metrics import roc_auc_score
from glob import glob

files = sorted(glob('/export/home/jomedina/collision-entropy/response_entropy/output/*-unified-uq.csv'))

dfs = {}
for f in files:
    df = pd.read_csv(f)
    df['GLU'] = (1 + df['S_tilde']) * df['she_R_mean']
    name = f.split('/')[-1].replace('-unified-uq.csv', '')
    dfs[name] = df

    sub = df.dropna(subset=['label', 'GLU'])
    n_drop = len(df) - len(sub)
    auroc = roc_auc_score(sub['label'], sub['GLU'])
    print(f"{name:30s}  AUROC(GLU): {auroc:.4f}  (dropped {n_drop})")