# MoPE: Mixture of Pathway Experts

MoPE learns molecular risk prediction from routine H&E whole-slide images (WSIs) using RNA sequencing only during training. The histology branch has one expert per Hallmark pathway (50 experts × 16 morphology slots, 64-d tokens). During training, an RNA branch encodes each patient's 50 Hallmark pathway vectors into pathway tokens, and the distribution of each histology expert over 24 shared reference (memory) vectors is distilled towards the distribution of the matching RNA pathway token (Kullback–Leibler transfer loss). The objective adds the task loss, an RNA reconstruction loss, a vector-quantization style memory regularizer and a slot-diversity term. At inference only the H&E features are needed.

## Repository layout

```
mope/                 model, datasets, training and inference code
  model.py            MoPE model
  data.py             classification (slide-level) and survival (patient-level) datasets
  engine.py           training loop, memory initialization, early stopping, metrics
  inference.py        prediction helpers
  config.py           JSON config + command-line overrides
train.py              five-fold training
evaluate.py           fold-wise predictions and metrics for trained checkpoints
infer_external.py     H&E-only inference on an external cohort
configs/              one config per task
scripts/              launchers (task, ablations, external validation)
tools/                RNA-seq -> Hallmark pathway inputs
splits/               patient-level TCGA splits used in the paper
data/hallmark/        MSigDB Hallmark gene sets (v2026.1)
analysis/             statistical and pathway analyses reported in the paper
tests/                smoke tests
```

## Installation

```bash
conda env create -f environment.yml
conda activate mope
```

or `pip install -r requirements.txt` in a Python 3.11 environment. Experiments were run with Python 3.11, PyTorch 2.8.0, scikit-learn 1.7.1, statsmodels 0.14.5, lifelines 0.30.0 and SciPy 1.16.1 on a single NVIDIA A100 GPU. Run all commands from the repository root.

## Data preparation

### 1. WSI features (TRIDENT + UNI2)

Download the TCGA diagnostic H&E slides from the [Genomic Data Commons](https://portal.gdc.cancer.gov). Segment tissue, tile non-overlapping 896 × 896-pixel patches at 20× and encode each patch with [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) using [TRIDENT](https://github.com/mahmoodlab/TRIDENT):

```bash
python run_batch_of_slides.py --task all --wsi_dir <wsi_dir> --job_dir <job_dir> \
    --patch_encoder uni_v2 --mag 20 --patch_size 896
```

This writes one `<slide_id>.h5` per slide (dataset `features`, shape N × 1536) to `<job_dir>/20x_896px_0px_overlap/features_uni_v2/`, which is the `--feature_dir` used below.

### 2. RNA-seq to Hallmark pathway inputs

Download the TCGA RNA-seq gene expression of each cohort from [UCSC Xena](https://xenabrowser.net/datapages/) and arrange it as a comma-separated table whose first column is the sample barcode (`TCGA-XX-XXXX-01`) and whose remaining columns are gene symbols. Then run, once per cohort:

```bash
python tools/build_pathway_inputs.py --rna_csv BRCA_mRNA.csv \
    --gmt data/hallmark/h.all.v2026.1.Hs.symbols.gmt --out_dir pathways/BRCA
```



### 3. Label tables

Each task needs a CSV with one row per slide:

- classification: `case_id`, `slide_id` and the label column named in the config (`label_col`; rows whose value equals `positive_label`, case-insensitive, are positive);
- survival: `case_id`, `slide_id`, `survival_months` and `censorship` (1 = censored).

`slide_id` must match the feature file names. Labels come from public sources: research-based Oncotype DX recurrence scores for TCGA-BRCA (Howard et al.), clinical PR status, the Buffa hypoxia score (median split) for TCGA-LUAD, somatic EGFR mutation calls, the pan-glioma molecular classification for the GBMLGG mesenchymal state, and TCGA clinical follow-up for overall survival.

## Training and evaluation

```bash
FEATURE_DIR=features PATHWAY_DIR=pathways/BRCA/patient_matrices CSV_PATH=tables/brca_odx.csv \
    bash scripts/run_task.sh brca_odx
```

This trains the five fold models (`train.py`) and then writes fold-wise predictions and metrics (`evaluate.py`) to `results/brca_odx/`. Equivalent direct calls:

```bash
python train.py --config configs/brca_odx.json --feature_dir features \
    --pathway_dir pathways/BRCA/patient_matrices --csv_path tables/brca_odx.csv --results_dir results/brca_odx
python evaluate.py --config configs/brca_odx.json --feature_dir features \
    --pathway_dir pathways/BRCA/patient_matrices --csv_path tables/brca_odx.csv --results_dir results/brca_odx
```








## License

Apache License 2.0 (see `LICENSE`). The Hallmark gene sets in `data/hallmark/` are from the Molecular Signatures Database (https://www.gsea-msigdb.org) and are subject to the MSigDB terms of use.
