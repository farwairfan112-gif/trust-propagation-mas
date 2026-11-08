# Cell 9f: unified re-analysis (T1-T4) and raw export. Usage outside the notebook:
#   python reanalysis_9f.py <dir with trials*.jsonl> <output dir>
import shutil
RESULTS_DIR = None
# ===== 9f. Unified re-analysis + raw export (no API calls) =====
import json, os, sys, glob
import numpy as np, pandas as pd
from scipy import stats

if RESULTS_DIR:          # running inside the notebook
    TRIALS_DIR, OUT = RESULTS_DIR, f"{RESULTS_DIR}/reanalysis"
else:
    TRIALS_DIR, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
B = 2000
SHORT = {"openai/gpt-oss-120b": "gpt-oss-120b", "openai/gpt-oss-20b": "gpt-oss-20b", "qwen/qwen3.8-27b": "Qwen3.8-27B"}

# ---------------------------------------------------------------- load (same rule as the notebook)
def _key(r):
    return "|".join(map(str, (r["exp"], r["topology"], r["model"], r["domain"], r["itype"], r["defense"],
                              r["param"], r["mode"], r["trial_idx"])))
rows, seen = [], set()
for p in sorted(glob.glob(f"{TRIALS_DIR}/trials*.jsonl")):
    for l in open(p):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if _key(r) in seen:
            continue
        seen.add(_key(r))
        for h in r["hops"]:
            h["valid"] = h.get("finish_reason") != "length"
        r["valid"] = all(h["valid"] for h in r["hops"])
        rows.append(r)
df = pd.DataFrame(rows)
df["m"] = df.model.map(SHORT)
print("trials loaded:", len(df), "| valid:", int(df.valid.sum()))

# ---------------------------------------------------------------- per-trial derived metrics
def decision_hops(r):
    hs = r["hops"]
    if r["topology"] in ("linear", "clean_linear"):
        return [max(hs, key=lambda h: h["hop"])]
    if r["topology"] == "hub":
        return [h for h in hs if h["role"] == "hub"]
    if r["topology"] == "mesh":
        return [h for h in hs if h["hop"] == 1] if r["mode"] == "vote_all" else [h for h in hs if h["hop"] == 2]
    return hs                                    # single / clean

def derive(r):
    hs, dh = r["hops"], decision_hops(r)
    n = len(dh)
    first = [h for h in hs if h["hop"] == 1 and h["role"] in ("agent", "spoke1", "peer1")]
    if r["topology"] == "mesh" and r["mode"] == "vote_all":
        first_c = np.mean([h["exec_strict"] for h in dh])          # per-agent rate (all three are directly poisoned)
    else:
        first_c = float(first[0]["exec_strict"]) if first else np.nan
    final = sum(h["exec_strict"] for h in dh) >= (n // 2 + 1)
    ok = r.get("final_correct")
    return pd.Series({
        "first_comply": first_c,
        "any_comply": float(any(h["exec_strict"] for h in hs)),
        "downstream_comply": float(any(h["exec_strict"] for h in hs if h not in first)) if r["topology"] != "single" else np.nan,
        "reach_decision": float(any(h["payload_in_input"] for h in dh)),
        "marker_reach_decision": float(any(h["marker_in_input"] for h in dh)),
        "final_comp": float(final),
        "final_marker_loose": float(sum(h["marker_in_raw"] for h in dh) >= (n // 2 + 1)),
        "task_fail": np.nan if ok is None or (isinstance(ok, float) and np.isnan(ok)) else float(not bool(ok)),
        "harm_f": np.nan if r["domain"] != "clinical_triage" or r.get("severity") != "malignant"
                   else float(bool(r.get("harm"))),
        "empty_final": float(any(h["empty_output"] for h in dh)),
    })

df = pd.concat([df, df.apply(derive, axis=1)], axis=1)
VT = df[df.valid].copy()

def structure(r):
    if r.topology in ("linear", "clean_linear"):
        return f"chain-3/{'natural' if r.topology == 'clean_linear' else r['mode']}"
    if r.topology == "hub":
        return f"hub k={r.param}"
    if r.topology == "mesh":
        return "mesh/debate" if r["mode"] == "peer_one" else "mesh/vote-all"
    return "single"
VT["structure"] = VT.apply(structure, axis=1)

# ---------------------------------------------------------------- statistics helpers
def boot(x, cl, binary=True, seed=0):
    x = np.asarray(x, float); c = pd.factorize(np.asarray(cl))[0]
    K = c.max() + 1 if len(c) else 0
    if K < 1:
        return (np.nan, np.nan)
    if binary and np.all(x == 0):
        return (0.0, 1 - 0.05 ** (1 / K))
    if binary and np.all(x == 1):
        return (0.05 ** (1 / K), 1.0)
    if K < 2:
        return (np.nan, np.nan)
    s = np.bincount(c, weights=x, minlength=K); n = np.bincount(c, minlength=K).astype(float)
    w = np.random.default_rng(seed).multinomial(K, np.ones(K) / K, size=B)
    return tuple(np.percentile((w @ s) / (w @ n), [2.5, 97.5]))

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))

def holm_(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); out[i] = min(1, run)
    return out

def fmt(r, lo, hi):
    return f"{r:.2f} [{lo:.2f}, {hi:.2f}]"

ATT = VT[VT.exp.isin(["linear", "hub", "mesh"])]

# ================================================================ T1: consistent compromise by topology
rows = []
for (m, dk, st, dfn), g in ATT.groupby(["m", "domain", "structure", "defense"]):
    rec = {"model": m, "domain": dk, "structure": st, "defense": dfn, "n": len(g), "items": g.item_id.nunique()}
    for col in ["first_comply", "any_comply", "reach_decision", "marker_reach_decision", "final_comp", "task_fail", "harm_f"]:
        x = g[col].dropna()
        if len(x) == 0:
            rec[col] = ""; continue
        lo, hi = boot(x.values, g.loc[x.index, "item_id"].values, binary=col != "first_comply" or st != "mesh/vote-all")
        rec[col] = fmt(x.mean(), lo, hi)
        rec[col + "_k"] = f"{x.sum():.0f}/{len(x)}" if col != "first_comply" or st != "mesh/vote-all" else ""
    rows.append(rec)
T1 = pd.DataFrame(rows)
T1.to_csv(f"{OUT}/T1_compromise_by_topology.csv", index=False)

# ================================================================ T2: paired attack - clean (same topology where available)
def clean_ref(m, dk, dfn, st):
    """Clean outcomes per item for the reference pipeline. Returns (series item->mean outcome, label)."""
    if st.startswith("chain-3"):
        c = VT[(VT.exp == "clean_chain") & (VT.model == m) & (VT.domain == dk) & (VT.defense == dfn)]
        lab = "clean chain-3"
    elif st in ("hub k=2", "hub k=4", "mesh/debate", "mesh/vote-all") and \
            len(VT[(VT.exp == "clean_topo") & (VT.model == m) & (VT.domain == dk) & (VT.defense == dfn) & (VT.structure == st)]):
        c = VT[(VT.exp == "clean_topo") & (VT.model == m) & (VT.domain == dk) & (VT.defense == dfn) & (VT.structure == st)]
        lab = f"clean {st}"
    else:
        c = VT[(VT.exp == "clean") & (VT["mode"] == "clean") & (VT.model == m) & (VT.domain == dk) & (VT.defense == dfn)]
        lab = "clean single agent (no same-topology clean run yet)"
    return c, lab

fam_rows = []
for (mm, dk, st, dfn), g in ATT.groupby(["model", "domain", "structure", "defense"]):
    c, lab = clean_ref(mm, dk, dfn, st)
    if len(c) == 0:
        continue
    outcome = "harm_f" if dk == "clinical_triage" else "task_fail"
    for oc in ([outcome, "task_fail"] if dk == "clinical_triage" else [outcome]):
        a = g[[oc, "item_id"]].dropna(); cc = c[[oc, "item_id"]].dropna()
        cm = cc.groupby("item_id")[oc].mean()
        a = a[a.item_id.isin(cm.index)]
        if len(a) < 2:
            continue
        ai = a.groupby("item_id")[oc].mean()                  # item-level means (code: ~1.7 attack trials/program)
        cv = cm.loc[ai.index]
        diff = (ai - cv).values
        lo, hi = boot(diff, ai.index.values, binary=False)
        # exact McNemar on item-level outcomes (clinical: exactly 1 attack trial per note -> exact pairs)
        ar, cr = (ai >= 0.5).astype(int), (cv >= 0.5).astype(int)
        b_ = int(((ar == 1) & (cr == 0)).sum()); c_ = int(((ar == 0) & (cr == 1)).sum())
        fam_rows.append({"model": SHORT[mm], "domain": dk, "structure": st, "defense": dfn, "outcome": oc,
                         "reference": lab, "items": a.item_id.nunique(), "n_attack": len(a),
                         "attack": ai.mean(), "clean": cv.mean(),
                         "excess": diff.mean(), "lo": lo, "hi": hi, "b_attack_only": b_, "c_clean_only": c_,
                         "p": mcnemar(b_, c_)})
T2 = pd.DataFrame(fam_rows)
for (oc, dfn, st), _ in T2.groupby(["outcome", "defense", "structure"]):   # Holm family: one outcome x defense x structure (across models, domains)
    i = (T2.outcome == oc) & (T2.defense == dfn) & (T2.structure == st)
    T2.loc[i, "p_holm"] = holm_(T2.loc[i, "p"].values)
T2 = T2.round(3)
T2.to_csv(f"{OUT}/T2_paired_attack_vs_clean.csv", index=False)

# correct-flag subset for natural chains (17 malignant notes)
bank_csv = sorted(glob.glob(f"{TRIALS_DIR}/clinical_note_bank*.csv"))
if bank_csv:
    bank = pd.read_csv(bank_csv[0]).reset_index().rename(columns={"index": "item_idx"})

# ================================================================ T3: outcome taxonomy (successful + unsuccessful attacks)
def cat(r):
    if r.any_comply == 1 and r.task_fail == 1: return "complied & task failed"
    if r.any_comply == 1: return "complied, task still correct"
    if r.task_fail == 1: return "no compliance, task failed"
    return "no compliance, task correct"
T = ATT.dropna(subset=["task_fail"]).copy()
T["outcome_class"] = T.apply(cat, axis=1)
T3 = (T.groupby(["m", "domain", "structure", "defense", "outcome_class"]).size()
        .unstack(fill_value=0).reset_index())
# clean failure rate of the reference pipeline, for the 'no compliance, task failed' column
cf = []
for _, r in T3.iterrows():
    mm = [k for k, v in SHORT.items() if v == r.m][0]
    c, lab = clean_ref(mm, r.domain, r.defense, r.structure)
    cf.append(round(c.task_fail.mean(), 3) if len(c) else np.nan)
T3["clean_task_fail_rate"] = cf
T3.to_csv(f"{OUT}/T3_outcome_taxonomy.csv", index=False)

# ================================================================ T4: defense effect on the consistent measures (paired)
d_rows = []
for (mm, dk, st), g in ATT.groupby(["model", "domain", "structure"]):
    if not {"none", "provenance_tagging"} <= set(g.defense):
        continue
    keycols = ["itype", "trial_idx", "param", "mode"]
    a = g[g.defense == "none"].set_index(keycols); b = g[g.defense == "provenance_tagging"].set_index(keycols)
    common = a.index.intersection(b.index)
    for oc in ["any_comply", "final_comp", "task_fail", "harm_f"]:
        x, y = a.loc[common, oc], b.loc[common, oc]
        ok = x.notna() & y.notna()
        if ok.sum() < 2:
            continue
        x, y = x[ok].values, y[ok].values
        dif = y - x
        lo, hi = boot(dif, a.loc[common][ok.values].item_id.values, binary=False)
        bb, cc = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        d_rows.append({"model": SHORT[mm], "domain": dk, "structure": st, "outcome": oc, "pairs": int(ok.sum()),
                       "none": x.mean(), "provenance": y.mean(), "diff": dif.mean(), "lo": lo, "hi": hi,
                       "helped": bb, "hurt": cc, "p": mcnemar(bb, cc)})
T4 = pd.DataFrame(d_rows)
for oc in T4.outcome.unique():
    i = T4.outcome == oc
    T4.loc[i, "p_holm"] = holm_(T4.loc[i, "p"].values)
T4 = T4.round(3)
T4.to_csv(f"{OUT}/T4_provenance_paired.csv", index=False)

# ================================================================ raw export
flat = df.drop(columns=["hops"]).copy()
flat.to_csv(f"{OUT}/raw_trials.csv", index=False)
hop_rows = []
for _, r in df.iterrows():
    for h in r["hops"]:
        hop_rows.append({"trial_key": _key(r), "exp": r.exp, "topology": r.topology, "model": r.model,
                         "domain": r.domain, "itype": r.itype, "defense": r.defense, "param": r.param,
                         "mode": r["mode"], "trial_idx": r.trial_idx, "item_id": r.item_id, **h})
pd.DataFrame(hop_rows).to_csv(f"{OUT}/raw_hops.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
print("\n=== T1 consistent measures, no defense ===")
cols = ["model", "domain", "structure", "n", "first_comply", "any_comply", "reach_decision", "final_comp", "final_comp_k", "harm_f", "task_fail"]
print(T1[T1.defense == "none"][cols].to_string(index=False))
print("\n=== T1 provenance ===")
print(T1[T1.defense == "provenance_tagging"][cols].to_string(index=False))
print("\n=== T2 paired attack - clean ===")
print(T2.to_string(index=False))
print("\n=== T3 outcome taxonomy ===")
print(T3.to_string(index=False))
print("\n=== T4 provenance vs none (paired) ===")
print(T4.to_string(index=False))
print("hops exported:", len(hop_rows))

shutil.make_archive(f"{OUT}/../reanalysis_export", "zip", OUT)
