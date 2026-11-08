# Trust Propagation and Containment in Multi-Agent LLM Pipelines

Code, prompts and results for the paper

> **Trust Propagation and Containment in Multi-Agent LLM Pipelines: An Empirical Study of Prompt Injection Attacks and Defenses**

The study follows an injected instruction hop by hop through multi-agent LLM pipelines and compares every attacked pipeline with a clean run of the same pipeline on the same cases. It uses two tasks with a known correct answer: clinical triage of 40 notes built from PAD-UFES-20 metadata, and repair of 24 QuixBugs programs checked by their unit tests. In total there are 5,752 trials with three open-weight models, six injection framings, four pipeline structures and four defences.

## Main results

| Finding | Result |
|---|---|
| Natural depth-3 chains, last agent executes the instruction (both `gpt-oss` models) | 0 of 40 in every cell |
| Extra malignant notes triaged ROUTINE vs. the same clean chain | +0.40 (`gpt-oss-120b`, Holm *p* = 0.016), +0.60 (`gpt-oss-20b`, Holm *p* = 0.001) |
| Extra failed programs vs. the same clean chain | +0.60, +0.78 and +0.50 (all Holm *p* < 0.001) |
| Hub-and-spoke and debate, final compromise | 0 of 478 trials (no defence), 0 of 479 (provenance tagging) |
| Vote-all mesh, extra task failure vs. clean vote-all | +0.40 to +0.79 (all Holm *p* < 0.001) |
| Keyword filter, compliant outputs detected | 0 of 2,233 |

The full tables are in `results/`. `results/results_tables.tex` is the exact LaTeX of Tables 3–5 in the paper.

## Repository layout

```
notebook/   trust_propagation.ipynb      full experiment notebook (Kaggle), outputs cleared
src/        prompts_payloads_defences.py task prompts, all six payloads, the three defences, agents and topologies
config/     model_config.json            models and API settings used in every trial
data/       clinical_note_bank_v3.csv    the 40 clinical notes
analysis/   reanalysis_9f.py             computes T1-T4 from the trial records
            make_tables.py               builds the paper's result tables from T1-T4
            structure_figure_9g.py       draws Fig. 3 (structure figure)
results/    T1-T4 CSV files and results_tables.tex from the final run
figures/    figures used in the paper
```

## Settings

- Models: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, served through the Groq API
- Temperature 0.3, at most 1,024 completion tokens
- `gpt-oss`: reasoning effort `low`. Qwen: reasoning disabled, `<think>` blocks removed before scoring
- Notes written by `qwen/qwen3.8-27b` from PAD-UFES-20 metadata

## Running the experiments

1. Open `notebook/trust_propagation.ipynb` on Kaggle and attach the PAD-UFES-20 metadata.
2. Add a Groq API key as the Kaggle secret `GROQ_API_KEY`. Without a key the notebook runs in mock mode, which checks the code but gives meaningless results.
3. Run all cells. Trials are saved to `results_v3/trials_v35.jsonl`. If a run stops at the daily API limit, the next run continues from the saved trials.

## Rebuilding the tables

```
pip install -r requirements.txt
python analysis/reanalysis_9f.py <folder with trials*.jsonl> <output folder>
python analysis/make_tables.py <output folder> results_tables.tex
```

Running `make_tables.py` on the files in `results/` gives `results/results_tables.tex` exactly.

## Data

- PAD-UFES-20: Pacheco et al., *Data in Brief* 32 (2020). Only the structured metadata is used.
- QuixBugs: Lin et al. (2017). The 24 programs and their 197 tests are inside the notebook.

The per-trial and per-hop records with every raw model output are available from the corresponding author (farwairfan112@gmail.com) on reasonable request.

## Citation

```bibtex
@inproceedings{irfan2026trust,
  title     = {Trust Propagation and Containment in Multi-Agent {LLM} Pipelines: An Empirical Study of Prompt Injection Attacks and Defenses},
  author    = {Irfan, Farwa and Randhawa, Muhammad Saad and Imtiaz, Usama},
  booktitle = {International Conference on Intelligent Digital Systems and Sustainable Technology (ICIDST 2026)},
  year      = {2026},
  note      = {Submitted}
}
```

## Licence

Code: MIT (see `LICENSE`). The clinical notes are derived from PAD-UFES-20 and follow that dataset's terms.
