# %% [markdown]
# # WEAR 2026 — Step F (v2): per-subject class-prior balancing (transductive post-processing)
# v2: wider grid, "free-null" mode (only the 18 activities are equalised, null share left to the model since it
# varies 0.14-0.57 across subjects), and alternating balance <-> propagation.
# Every subject performs all 18 activities (each ~3.3% of tiles in train) with ~40% null. Stage2 predictions per test
# subject are unbalanced (e.g. push-ups 1.1% vs push-ups-complex 4.7%), hinting at systematic confusion between variants.
# Sinkhorn-style scaling finds per-subject class weights w so the mean prediction moves towards a target share r;
# final P' = P * w**lam. Target null share tau and strength lam are tuned on stage2 OOF (subject-grouped).

# %%
import glob, os
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.metrics import f1_score

def find(name): return os.path.dirname(glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0])
P, S2, C = find("tr_meta.csv"), find("oof_stage2.npy"), find("tr_pairs.parquet")
OUT = "/kaggle/working"
tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
y = tr_meta.label.values; tr_sbj, te_sbj = tr_meta.sbj.values, te_meta.sbj_id.values
oof = np.load(f"{S2}/oof_stage2.npy"); pte = np.load(f"{S2}/te_stage2.npy")
tr_pairs = pd.read_parquet(f"{C}/tr_pairs.parquet"); te_pairs = pd.read_parquet(f"{C}/te_pairs.parquet")
def mf1(p, w0=1.0): q = p.copy(); q[:, 0] *= w0; return f1_score(y, q.argmax(1), average="macro")
def best_null(p): return max((round(mf1(p, w), 4), w) for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2])

# %%
# How stable are class shares per subject in train?
share = pd.crosstab(tr_sbj, y, normalize="index")
print("null share per subject: mean %.3f, std %.3f, min %.3f, max %.3f" % (share[0].mean(), share[0].std(), share[0].min(), share[0].max()))
act = share.drop(columns=0).div(1 - share[0], axis=0)
print("activity share among non-null: mean %.4f, per-subject std (avg over classes) %.4f, CV %.2f" % (act.values.mean(), act.std(0).mean(), (act.std(0) / act.mean(0)).mean()))

# %%
def rownorm(A):
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return sp.diags(1 / d) @ A
def propagate(P2, pairs, alpha=0.7):
    n = len(P2); W = sp.csr_matrix((pairs.p.values, (pairs.gi.values, pairs.gj.values)), (n, n))
    return (1 - alpha) * P2 + alpha * (rownorm(W + W.T) @ P2)

def balance(Pm, sbj, tau, lam, iters=100):
    """tau=None -> free-null: equalise the 18 activity shares, keep null weight 1."""
    out = Pm.copy()
    for s in np.unique(sbj):
        m = sbj == s; Q = Pm[m]; w = np.ones(19)
        for _ in range(iters):
            Z = Q * w; Z /= Z.sum(1, keepdims=True); mean = Z.mean(0)
            if tau is None: r = np.r_[mean[0], np.full(18, (1 - mean[0]) / 18)]
            else: r = np.r_[tau, np.full(18, (1 - tau) / 18)]
            w *= (r / (mean + 1e-9)) ** 0.5
            if tau is None: w /= w[0]
        out[m] = Q * w ** lam
    return out / out.sum(1, keepdims=True)

oofp = propagate(oof, tr_pairs); ptep = propagate(pte, te_pairs)
print("stage2 OOF raw", best_null(oof), "| after propagation", best_null(oofp))
res = []
for tau in [None, 0.40, 0.45, 0.50, 0.55]:
    for lam in [0.75, 1.0, 1.5, 2.0]:
        f, w0 = best_null(balance(oofp, tr_sbj, tau, lam)); res.append(dict(tau=tau, lam=lam, rounds=1, f1=f, null_w=w0))
        print(res[-1], flush=True)
res = pd.DataFrame(res).sort_values("f1", ascending=False)
# alternating balance <-> propagation for the two best settings
def alt(Pm, sbj, pairs, tau, lam, rounds):
    Q = Pm
    for _ in range(rounds): Q = propagate(balance(Q, sbj, tau, lam), pairs)
    return Q
for _, r0 in res.head(2).iterrows():
    tau = None if pd.isna(r0.tau) else r0.tau
    for rounds in [2, 3]:
        f, w0 = best_null(alt(oof, tr_sbj, tr_pairs, tau, r0.lam, rounds)); res = pd.concat([res, pd.DataFrame([dict(tau=tau, lam=r0.lam, rounds=rounds, f1=f, null_w=w0)])])
        print(res.iloc[-1].to_dict(), flush=True)

# %%
res = res.sort_values("f1", ascending=False); print(res.head(8).to_string())
b = res.iloc[0]; tau = None if pd.isna(b.tau) else b.tau
if b.rounds == 1: q_oof, q = balance(oofp, tr_sbj, tau, b.lam), balance(ptep, te_sbj, tau, b.lam)
else: q_oof, q = alt(oof, tr_sbj, tr_pairs, tau, b.lam, int(b.rounds)), alt(pte, te_sbj, te_pairs, tau, b.lam, int(b.rounds))
q_oof[:, 0] *= b.null_w
pf = f1_score(y, q_oof.argmax(1), average=None)
print("per-class F1 after balancing:", dict(enumerate(pf.round(3))))
print("per-subject F1:", {s: round(f1_score(y[tr_sbj == s], q_oof[tr_sbj == s].argmax(1), average="macro"), 3) for s in np.unique(tr_sbj)})
q[:, 0] *= b.null_w
pred = q.argmax(1)
print("test pred share per subject:"); print(pd.crosstab(te_sbj, pred, normalize="index").round(3).to_string())
pd.DataFrame({"id": te_meta.id, "target_feature": pred}).to_csv(f"{OUT}/submission.csv", index=False)
print("best", b.to_dict())
