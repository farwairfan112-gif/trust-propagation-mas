"""Builds results_tables.tex from results_v3/reanalysis CSVs. Re-run after each notebook run."""
import sys, pandas as pd
D, OUT = sys.argv[1], sys.argv[2]
T1 = pd.read_csv(f"{D}/T1_compromise_by_topology.csv"); T2 = pd.read_csv(f"{D}/T2_paired_attack_vs_clean.csv")
T3 = pd.read_csv(f"{D}/T3_outcome_taxonomy.csv");        T4 = pd.read_csv(f"{D}/T4_provenance_paired.csv")
TBD = r"\tbd"
ST = [("chain-3/natural","Chain, natural"),("chain-3/passthrough","Chain, pass-through"),("hub k=2","Hub, $k{=}2$"),
      ("hub k=4","Hub, $k{=}4$"),("mesh/debate","Debate mesh"),("mesh/vote-all","Vote-all mesh")]
GPT = ["gpt-oss-120b","gpt-oss-20b"]
def done(m, st):   # a cell is complete only when attack and same-structure clean runs are both in T2 with that reference
    if not st.startswith("hub") and not st.startswith("mesh"): return True
    r = T2[(T2.model==m)&(T2.structure==st)]
    return len(r) > 0 and not r.reference.str.contains("single agent").any()
def n(x): return f"{x:+.2f}".replace("-", "$-$")
def rate(m, dk, st, dfn, col):
    r = T1[(T1.model==m)&(T1.domain==dk)&(T1.structure==st)&(T1.defense==dfn)]
    if not done(m, st) or r.empty or pd.isna(r.iloc[0][col]): return TBD
    return f"{float(str(r.iloc[0][col]).split()[0]):.2f}"
def exc(m, dk, st, dfn, oc, ci=True):
    r = T2[(T2.model==m)&(T2.domain==dk)&(T2.structure==st)&(T2.defense==dfn)&(T2.outcome==oc)]
    if not done(m, st) or r.empty: return TBD
    r = r.iloc[0]; v = f"{n(r.excess)} [{n(r.lo)}, {n(r.hi)}]" if ci else f"{n(r.excess)}"
    return f"\\textbf{{{v}}}" if r.p_holm < 0.05 else v
def topo(dfn, label, cap):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{"+cap+"}", r"\label{"+label+"}", r"\scriptsize", r"\setlength{\tabcolsep}{1.8pt}",
         r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{Clinical triage} & \multicolumn{3}{c}{Code repair}\\", r"\cmidrule(lr){3-6}\cmidrule(l){7-9}",
         r"Model & Structure & $C_1$ & $C_F$ & Harm excess & Fail exc. & $C_1$ & $C_F$ & Fail excess\\", r"\midrule"]
    for m in GPT:
        for i,(st,name) in enumerate(ST):
            L.append((m if i==0 else "") + f" & {name} & {rate(m,'clinical_triage',st,dfn,'first_comply')} & {rate(m,'clinical_triage',st,dfn,'final_comp')}"
                f" & {exc(m,'clinical_triage',st,dfn,'harm_f')} & {exc(m,'clinical_triage',st,dfn,'task_fail',False)}"
                f" & {rate(m,'code_repair',st,dfn,'first_comply')} & {rate(m,'code_repair',st,dfn,'final_comp')} & {exc(m,'code_repair',st,dfn,'task_fail')}\\\\")
        L.append(r"\midrule" if m != GPT[-1] else r"\bottomrule")
    return "\n".join(L + [r"\end{tabular}", r"\end{table}"])
def prov():
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Paired effect of provenance tagging: provenance minus no defence on the same case and framing. Any compliance: some agent in the pipeline complied. Task failure with 95\% item-bootstrap CI. Bold: Holm-adjusted exact McNemar $p<0.05$.}",
         r"\label{tab:provenance}", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Any compliance} & \multicolumn{2}{c}{Task failure}\\", r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
         r"Model & Structure & Clinical & Code & Clinical & Code\\", r"\midrule"]
    for m in GPT + ["Qwen3.8-27B"]:
        sts = ST if m != "Qwen3.8-27B" else ST[:2]
        for i,(st,name) in enumerate(sts):
            def d(dk, oc, ci):
                if not done(m, st): return TBD
                r = T4[(T4.model==m)&(T4.domain==dk)&(T4.structure==st)&(T4.outcome==oc)]
                if r.empty: return "--"
                r = r.iloc[0]; v = f"{n(r['diff'])} [{n(r.lo)}, {n(r.hi)}]" if ci else f"{n(r['diff'])}"
                return f"\\textbf{{{v}}}" if r.p_holm < 0.05 else v
            L.append((m if i==0 else "") + f" & {name} & {d('clinical_triage','any_comply',False)} & {d('code_repair','any_comply',False)}"
                     f" & {d('clinical_triage','task_fail',True)} & {d('code_repair','task_fail',True)}\\\\")
        L.append(r"\midrule" if m != "Qwen3.8-27B" else r"\bottomrule")
    return "\n".join(L + [r"\end{tabular}", r"\end{table}"])
def taxo():
    cols = ["complied & task failed","complied, task still correct","no compliance, task correct","no compliance, task failed"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Outcome of every attacked trial: CF = some agent complied and the final output failed the task; CC = some agent complied but the final output was still correct; NC = no agent complied and the output was correct; NF = no agent complied but the output failed (the clean failure rate of the same structure is 0.28--0.35 in triage and 0.13--0.29 in code). Counts out of 40 trials (39 where a trial was invalid).}",
         r"\label{tab:taxonomy}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Clinical triage} & \multicolumn{2}{c}{Code repair}\\", r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
         r"Model & Structure & none & provenance & none & provenance\\",
         r" & & CF/CC/NC/NF & CF/CC/NC/NF & CF/CC/NC/NF & CF/CC/NC/NF\\", r"\midrule"]
    for m in GPT:
        for i,(st,name) in enumerate(ST):
            def c(dk, dfn):
                if not done(m, st): return TBD
                r = T3[(T3.m==m)&(T3.domain==dk)&(T3.structure==st)&(T3.defense==dfn)]
                return "/".join(str(int(r.iloc[0][k])) for k in cols) if len(r) else "--"
            L.append((m if i==0 else "") + f" & {name} & {c('clinical_triage','none')} & {c('clinical_triage','provenance_tagging')} & {c('code_repair','none')} & {c('code_repair','provenance_tagging')}\\\\")
        L.append(r"\midrule" if m != GPT[-1] else r"\bottomrule")
    return "\n".join(L + [r"\end{tabular}", r"\end{table}"])
cap0 = (r"Outcomes by pipeline structure, no defence. $C_1$: compliance of the directly poisoned agent; $C_F$: final compromise, i.e.\ a majority of "
        r"decision-point outputs executes the instruction. Excess: attacked minus clean run of the \emph{same} structure on the same cases (harm: malignant "
        r"note triaged ROUTINE, 20 notes; failure: wrong priority or failed unit tests), with 95\% item-bootstrap CI. Bold: Holm-adjusted exact McNemar $p<0.05$. "
        r"")
cap1 = cap0.replace("no defence", "provenance tagging (residual effect under the defence)")
open(OUT,"w").write("% auto-generated by make_tables.py from results_v3/reanalysis; do not edit by hand\n\n"
    + topo("none","tab:topology",cap0) + "\n\n" + prov() + "\n\n" + taxo() + "\n")
