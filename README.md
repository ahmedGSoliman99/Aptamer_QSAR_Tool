# Aptamer QSAR Tool

A beginner-friendly Streamlit platform for RNA/DNA aptamer QSAR modeling.

## What it does

- Accepts RNA and DNA aptamer sequences from manual text, FASTA, CSV, Excel, or TXT files.
- Calculates nucleic-acid descriptors: length, GC fraction, base composition, dinucleotide/trinucleotide composition, entropy, approximate molecular weight, extinction proxy, melting-temperature approximations, G-rich motif counts, homopolymer runs, and self-complementarity proxy.
- Lets users enter aptamer-target interaction descriptors for every aptamer, including hydrogen bonds, hydrophobic contacts, pi-stacking, electrostatic contacts, salt bridges, metal coordination, van der Waals contacts, water bridges, base-pairing contacts, base-stacking contacts, and target-contact counts.
- Trains regression or classification QSAR models with scikit-learn.
- Predicts and ranks new aptamers.
- Generates new model-guided aptamer candidates by mutating a selected seed aptamer, scoring candidates with the trained model, and ranking predicted improvements.
- Exports descriptor tables, model leaderboards, predictions, and reports.

## Run locally on Windows

Double-click `run_app.bat`.

If you prefer manual setup:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8503
```

## Deploy on Streamlit Cloud

- Repository: `ahmedGSoliman99/Aptamer_QSAR_Tool`
- Branch: `main`
- Main file path: `app.py`
- Recommended Python: `3.12` or `3.11`

## Example data

`data/example_aptamers.csv` contains synthetic DNA/RNA aptamers with `Kd_nM`, `ActivityScore`, and interaction-count columns. Replace it with real experimental, docking, molecular-dynamics, or curated data for research.

## Developer

Developed by Ahmed G. Soliman.
Portfolio: https://sites.google.com/view/ahmed-g-soliman/home
