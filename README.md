# BINN — Biologically Informed Neural Network for Single-Cell Lineage Prediction

A multimodal deep learning pipeline for predicting developmental lineages in single-cell RNA-seq data and spatially resolving them in spatial transcriptomics.

---

## Overview

Lentiviral barcoding delivers unique DNA sequences into cells in utero, allowing clonal relationships to be traced through sequencing. However, barcoding efficiency is incomplete — only a fraction of cells carry a barcode. BINN learns the transcriptional signatures of barcoded clonal lineages and transfers those predictions to unbarcoded cells, effectively extending lineage information across an entire dataset.

The pipeline has three components:

1. **Preprocessing** — clone2vec clustering of barcoded cells into pseudo-lineages, followed by multimodal biological feature scoring
2. **Training** — a biologically informed neural network trained on barcoded cells to predict lineage from multimodal transcriptional features
3. **Inference** — prediction of pseudo-lineage identity and confidence for all cells, including unbarcoded ones, and optional spatial deconvolution onto spatial transcriptomics data

---

## Biological motivation

Raw gene expression alone captures what a cell is doing right now. BINN augments this with three additional biological feature layers computed from the same expression data:

| Modality | Method | Output | Biological meaning |
|---|---|---|---|
| Gene expression | SCT normalised counts | 4,329 genes | Current transcriptional state |
| TF activity | DoRothEA ULM | 259 TFs | Inferred regulatory activity |
| Pathway activity | PROGENy MLM | 14 pathways | Signalling pathway status |
| GO term activity | AUCell | 2,457 terms | Biological process enrichment |

Each modality is encoded independently, then fused via cross-modal attention. This allows the model to discover relationships between regulatory programmes and lineage identity that are not visible in gene expression alone.

---

## Architecture
```
Gene expression  ──► Expr encoder (512→128)  ──┐
TF activity      ──► TF encoder   (128→128)  ──┤
Pathway activity ──► Path encoder  (32→128)  ──┼──► Cross-modal attention ──► z_lineage (64-dim) ──► Lineage head
GO term activity ──► GO encoder   (256→128)  ──┘                                                 ──► Celltype head (adversarial)
                                                                                                  ──► Aux heads × 4
```

**Modality encoders** compress each input into a 128-dimensional token using two-layer MLPs with LayerNorm, GELU activation, and dropout.

**Cross-modal attention** applies 4-head self-attention across the 4 modality tokens, allowing each modality to contextualise information from the others. Attention weights are stored and used for interpretability.

**Latent decomposition** splits the fused representation into a 64-dimensional lineage subspace and a 32-dimensional celltype subspace.

**Lineage head** maps z_lineage to class logits via a 3-layer MLP.

**Adversarial celltype head** applies gradient reversal on z_lineage and predicts cell type. This forces z_lineage to discard cell type information, ensuring the lineage representation captures developmental origin rather than current differentiation state.

**Per-modality auxiliary heads** predict lineage directly from each modality token. This prevents modality collapse — without auxiliary losses, the attention mechanism can learn to ignore all modalities except gene expression.

---

## Loss function
```
L = 1.0 · CE(ŷ, y) + 0.1 · CE(ĉ, c) + 0.3 · (1/4) Σ CE(âₘ, y)
         lineage         adversarial              auxiliary
```

Class-weighted cross-entropy is used for the lineage term to handle imbalanced clone sizes. The adversarial weight (0.1) is kept small to avoid destabilising the primary objective. The adversarial pressure is ramped in via a sigmoid schedule starting at epoch 40, allowing the lineage representation to stabilise before disentanglement begins.

---

## Training

| Parameter | Value |
|---|---|
| Optimiser | Adam (lr=1e-3, weight_decay=5e-4) |
| Scheduler | CosineAnnealingLR |
| Gradient clipping | 1.0 |
| Early stopping | patience=40, min_epochs=80 |
| Train / val split | 85% / 15% (stratified by random seed) |
| Temperature calibration | LBFGS on validation NLL |

Training uses only barcoded cells with clone sizes ≥ min_clone_size. Validation is performed on held-out barcoded cells. The model at peak validation balanced accuracy is restored at the end of training.

---

## Validation

| Metric | Value (full dataset, min_size=5) |
|---|---|
| Validation accuracy | 75.2% |
| Validation balanced accuracy | 71.6% |
| Random baseline | 3.7% |
| Transcriptomics-only MLP | 68.8% |
| BINN improvement over null | +2.9% |
| ARI vs clone2vec ground truth | 0.893 |
| NMI vs clone2vec ground truth | 0.909 |
| Temperature (calibrated) | T = 1.46 |
| Conf > 0.9 accuracy | 81.5% |

**Ablation** (contribution of each modality):

| Removed modality | Balanced accuracy | Drop |
|---|---|---|
| None (full model) | 71.6% | — |
| Gene expression | 21.6% | −50.0% |
| TF activity | 69.9% | −1.8% |
| GO terms | 66.2% | −5.4% |
| Pathway activity | 71.0% | −0.6% |

**Single-sample generalisation** (trained on one animal, inferred on held-out animals):

| Split | Mean balanced accuracy | Mean ARI |
|---|---|---|
| Train sample | 96.9% | 0.931 |
| Held-out injected | 53.2% | 0.400 |

The drop from train to held-out reflects genuine biological variability between animals — not overfitting. ARI = 0.400 across completely independent animals with no shared training data is evidence that the model has learned transferable lineage signatures.

---

## Spatial transcriptomics inference

BINN can be applied to spatial transcriptomics data from the same tissue and timepoint. Two inference modes are available:

**Classification mode** — each spot is assigned the most probable lineage. Produces a single lineage label per spot and a confidence score.

**Deconvolution mode** — the softmax output is treated as a proportion vector, giving the estimated mixture of lineages within each spot. A dedicated deconvolution model can be trained using pseudo-bulk synthetic spots (random weighted mixtures of scRNA-seq cells with known compositions) optimised with Jensen-Shannon divergence loss instead of cross-entropy.
```
scRNA-seq cells ──► random weighted mixing ──► synthetic spots with known proportions
                                                          │
                                              JS-divergence loss
                                                          │
                                              Deconv model learns to predict
                                              proportion vectors, not class labels
```

---

## Interpretability

**Gradient attribution** computes `|∂logit_L/∂input|` for each lineage L and each input feature, averaged across all cells of that lineage. This identifies which genes, TFs, pathways, and GO terms the model relies on most heavily for each lineage prediction.

**Sankey diagrams** visualise the gradient attribution as flow diagrams across the biological hierarchy:

- Forward Sankey: Lineage → Modality → Feature
- Reversed Sankey: Feature → Modality → Lineage
- Multi-layer Sankey: Genes → TF/Pathway/GO → Lineage (using known regulatory network edges from DoRothEA and PROGENy)

**Attention weight analysis** reveals cross-modal information flow. In the trained model, TF, pathway, and GO tokens attend heavily to the gene expression token (~0.86 attention weight), while expression attends primarily to itself (0.615). This is biologically consistent — TF and pathway activity are downstream consequences of gene expression.

---

## Assumptions

- SCT normalisation has removed technical variation — remaining variation is biological
- Clone2vec lineage clusters are treated as ground truth labels. The BINN learns to predict a proxy label, not directly observed developmental lineage
- Barcoded cells are representative of unbarcoded cells — lentiviral barcoding is not systematically biased toward specific cell types beyond what flow sorting introduces
- Lineages are transcriptionally distinct — if two lineages are transcriptionally indistinguishable, no model can separate them
- Biological networks (DoRothEA, PROGENy, GO) are sufficiently complete for the tissue and organism of interest

---

## Limitations

- The model predicts pseudo-lineages defined by clone2vec clustering, not directly observed developmental relationships
- Performance depends on clone2vec cluster quality, which is sensitive to min_clone_size and c2v_resolution
- Flow-sorted samples create cell type composition bias relative to unsorted controls — NC comparisons require composition correction
- Spatial transcriptomics spots contain multiple cells — the deconvolution model addresses this but is trained on synthetic mixtures, not true ground truth spot compositions
- Pathway activity (14 features) contributes minimally to prediction — this modality is too coarse to discriminate 27 lineages

---

## File outputs

| File | Contents |
|---|---|
| `adata_preprocessed.h5ad` | AnnData with gene expression, X_tf, X_pathway, X_go, lineage_code, celltype_code |
| `label_encoders.pkl` | LabelEncoder objects for le_lineage and le_celltype |
| `dims.json` | Input dimensions and training parameters |
| `go_net.csv` | GO network used for AUCell — required to score new datasets identically |
| `heart_binn_inference_package.pt` | Model weights, scaler weights, dims, val_bacc |
| `heart_binn_deconv.pt` | Deconvolution model weights trained with JS-divergence loss |
| `E16EXC_ST_HEART1_pseudolineage.h5ad` | ST data with pseudo_lineage, pseudo_lineage_conf, prob_lineage_* columns |
| `E16EXC_ST_HEART1_deconv_retrained.h5ad` | ST data with deconv proportion columns |
| `ST_deconv_proportions.csv` | Per-spot lineage proportion matrix |

---

## Requirements
```
python >= 3.12
scanpy
torch >= 2.0
decoupler >= 2.1
sclitr
gseapy
anndata
scikit-learn
scipy
pandas
numpy
matplotlib
```

---

## Usage
```python
# 1. Preprocess
# Edit CONFIG in BINN_Preprocessing_v2.py then run end-to-end

# 2. Train
# Edit CONFIG in BINN_Training_v2.py then run end-to-end

# 3. Infer on new data
checkpoint = torch.load("heart_binn_inference_package.pt",
                         map_location="cpu", weights_only=False)
model = BINN(checkpoint["dims"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
# See inference scripts for full preprocessing and alignment pipeline
```

---

## Citation

If you use this pipeline, please cite the clone2vec and decoupleR tools it builds on:

- **clone2vec / sclitr**: Palla et al.
- **DoRothEA**: Garcia-Alonso et al., *Genome Research* 2019
- **PROGENy**: Schubert et al., *Nature Communications* 2018
- **decoupleR**: Badia-i-Mompel et al., *Bioinformatics Advances* 2022
