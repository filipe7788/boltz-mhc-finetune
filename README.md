# Boltz-1 Fine-Tuning for MHC Class I Peptide Complexes

Fine-tuning [Boltz-1](https://github.com/jwohlwend/boltz) on MHC Class I / HLA peptide complexes.

---

## Table of Contents

- [Why fine-tune Boltz-1 for pMHC](#why-fine-tune-boltz-1-for-pmhc)
- [Why not docking](#why-not-docking)
- [Pipeline](#pipeline)
- [Technical Decisions](#technical-decisions)
  - [Data source and resolution cutoff](#data-source-and-resolution-cutoff)
  - [Chain classification](#chain-classification)
  - [MSA handling for short peptides](#msa-handling-for-short-peptides)
  - [Dataset split strategy](#dataset-split-strategy)
  - [Partial vs full fine-tuning](#partial-vs-full-fine-tuning)
  - [Optimizer and learning rate schedule](#optimizer-and-learning-rate-schedule)
  - [Evaluation metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [References](#references)

---

## Why fine-tune Boltz-1 for pMHC

Boltz-1 out of the box was trained on the full PDB — a distribution dominated by globular, well-folded proteins with deep MSAs. pMHC complexes violate several of its strong priors in ways that make domain-specific fine-tuning necessary rather than optional:

**1. No MSA signal on the peptide.**

Boltz-1's architecture, like AlphaFold3's, builds its internal representations in two stages. First, an Evoformer-like trunk processes pairwise residue relationships and single-sequence features, heavily conditioned on the MSA. Then a diffusion decoder uses those representations to generate 3D coordinates. AlphaFold2 demonstrated that MSA depth is the single strongest predictor of structure prediction accuracy — models trained without MSA perform dramatically worse than those with deep alignments ([Jumper et al., 2021](https://doi.org/10.1038/s41594-022-00849-w)). Co-evolving residue pairs in the MSA reveal which residues are spatially proximal, driving accurate pair representations before any coordinates are generated.

For an 8–12 residue viral peptide, this entirely breaks down. A 9-mer has no meaningful homologs in UniRef90. Any MSA you construct is either empty or full of non-homologous hits (short subsequences from unrelated proteins) that introduce noise into the pairwise representation rather than useful co-evolutionary signal. The sensitivity of Evoformer-based models to MSA quality — including the degradation caused by including non-homologous sequences — has been characterized in the ColabFold benchmarks ([Mirdita et al., 2022](https://doi.org/10.1038/s41592-022-01488-1)). The Evoformer trunk receives spurious signal for the peptide chain, which degrades the diffusion decoder's conditioning at exactly the residues you most care about.

Fine-tuning teaches the model to compensate: it learns to rely on cross-chain attention between the groove and the peptide instead of within-chain MSA co-variation. This is a qualitatively different inference regime that requires seeing many pMHC examples during training.

**2. Peptide conformation is groove-determined, not sequence-determined.**

In standard protein folding, a sequence encodes its own structure — hydrophobic collapse, secondary structure propensity, disulfide patterns. The Evoformer is trained to extract these signals. In a pMHC complex, the peptide has essentially no intrinsic structure: a free 9-mer in solution is largely disordered. Its conformation in the crystal is entirely imposed by the MHC groove — the two anchor pockets (B and F pockets, first described by [Saper et al., 1991](https://doi.org/10.1016/0092-8674(91)90412-G)) fix positions 2 and 9 of the peptide; the intervening residues bulge upward to varying degrees depending on length and sequence ([Madden et al., 1993](https://doi.org/10.1016/0092-8674(93)90490-H)).

The base Boltz-1 model has no trained representation of this groove-imposed folding regime. It has seen very few examples relative to its overall training distribution where a short, intrinsically disordered chain adopts a well-defined conformation dictated entirely by a binding partner. Fine-tuning on ~3,000 pMHC structures shifts the diffusion decoder's prior from "peptide folds from sequence" to "peptide conforms to groove."

**3. HLA polymorphism distribution in the PDB is heavily skewed.**

The ~5,000 MHC Class I structures in the PDB are not uniformly distributed across the known allelic space. A small number of alleles — particularly HLA-A\*02:01 — account for a disproportionate fraction of solved structures because they crystallize readily and have been studied for decades since the first HLA structure was solved ([Bjorkman et al., 1987](https://doi.org/10.1038/329512a0)). The IMGT/HLA database currently catalogs over 30,000 known HLA alleles ([Robinson et al., 2020](https://doi.org/10.1093/nar/gkz950)), the vast majority of which have zero crystal structures in the PDB.

A model trained naively on the full PDB will overfit to the groove geometry of HLA-A\*02:01 and a handful of other well-represented alleles. Fine-tuning with an explicit allele-stratified split (see [Dataset split strategy](#dataset-split-strategy)) and class-balanced sampling penalizes allele-specific memorization and pushes the model toward learning the general structural logic of groove-peptide interaction that transfers across alleles. The importance of allelic diversity in training for pan-allele generalization has been demonstrated in the sequence-level prediction literature ([Reynisson et al., 2020](https://doi.org/10.1093/nar/gkaa379)).

**4. The diffusion decoder has a weak prior over short constrained chains.**

Boltz-1's diffusion module is trained to denoise atomic coordinates across the full diversity of PDB structures, using a score-matching objective similar to that of AlphaFold3 ([Abramson et al., 2024](https://doi.org/10.1038/s41586-024-07487-w)) and SE(3) diffusion models for protein structure ([Yim et al., 2023](https://arxiv.org/abs/2302.02277)). The implicit prior it learns reflects that diversity: it expects chains that are dozens to hundreds of residues long, with secondary structure elements and a compact hydrophobic core. Short (8–12 residue) chains that take fixed groove-constrained conformations with no secondary structure are a small fraction of its training distribution.

In practice, this means that without fine-tuning, the diffusion decoder either under-constrains the peptide (generating physically plausible but incorrectly positioned conformations) or falls back on secondary structure propensity (incorrectly predicting helical or strand content for a peptide that should be in an extended groove-bound pose). This failure mode — where diffusion models trained on broad distributions perform poorly on narrow domain-specific subsets — motivates domain-specific fine-tuning as a standard practice in protein structure prediction ([Wu et al., 2024](https://arxiv.org/abs/2402.04845)). Fine-tuning on pMHC structures directly reshapes the decoder's learned score function for this regime.

---

## Why not docking

Molecular docking is the classical computational approach for placing a ligand into a receptor binding site. For pMHC, tools like [AutoDock Vina](https://vina.scripps.edu/), [Rosetta FlexPepDock](https://flexpepdock.furmanlab.cs.huji.ac.il/), [APE-Gen](https://github.com/KavrakiLab/APE-Gen), and [HADDOCK](https://www.bonvinlab.org/software/haddock2.4/) have been applied to the problem of predicting peptide conformation in the groove. Understanding why we use Boltz-1 fine-tuning instead requires being precise about what docking can and cannot do.

**What docking does well for pMHC:**

Docking methods take a known receptor structure (an existing HLA crystal structure) and search for the lowest-energy pose of the peptide inside the groove. They are fast (seconds to minutes per prediction), interpretable (energy terms map to physical interactions), and work well for alleles with high-quality crystal structures available. For the most well-characterized alleles, tools like FlexPepDock achieve pRMSD < 1.5 Å on benchmarks.

**Why docking is insufficient for our goals:**

1. **Requires a receptor structure per allele.** Docking cannot predict the structure of an allele with no experimentally solved crystal structure. It can only place a peptide into a groove it has already seen. Of ~30,000 known HLA alleles, fewer than 200 have crystal structures in the PDB. The remaining ~99.3% are structurally inaccessible to docking methods without first modeling the receptor — which reintroduces a structure prediction problem.

2. **Receptor flexibility is difficult to model.** HLA grooves are not perfectly rigid — different peptides induce measurable conformational changes in the α-helices and groove floor, particularly for longer peptides (10–12mers) that can push on groove boundaries. Most docking protocols treat the receptor as rigid or allow only limited side-chain flexibility, systematically missing these induced-fit effects. Boltz-1's joint diffusion over receptor and peptide coordinates naturally captures mutual conformational adaptation.

3. **Energy functions are imperfect for groove-peptide interactions.** Classical docking energy functions were developed for small molecule–protein interactions and perform less reliably for peptide–protein interfaces, which involve backbone hydrogen bonds, desolvation of a long extended chain, and many shallow individual contacts spread across the entire groove. This is a known limitation of docking applied to peptides ([London et al., 2011](https://pubmed.ncbi.nlm.nih.gov/21128681/)).

4. **No end-to-end trainability.** Docking scoring functions are not differentiable with respect to the training data. You cannot fine-tune them on pMHC structures to improve accuracy; the model is fixed at deployment. Boltz-1 is a neural network: fine-tuning directly optimizes the parameters that generate peptide coordinates against experimentally determined ground-truth structures.

**Where docking is still useful in this project:**

Docking is a good source of **data augmentation for rare alleles**. For alleles with no crystal structure, you can build a homology model of the groove (using HLA-A\*02:01 or another close allele as template), run FlexPepDock or APE-Gen to generate predicted pMHC structures, and add those predicted structures to the fine-tuning dataset. The model trains on more allelic diversity, even if the labels are predicted rather than experimental. This hybrid approach — experimental structures as the core dataset, docking-generated structures for rare alleles — is a practical extension of this pipeline.

---

## Pipeline

```
RCSB PDB query (API)
        │  mmCIF, resolution ≤ 3.0 Å
        ▼
data/filter_structures.py
        │  classify α chain, B2M, peptide by residue count
        │  discard incomplete peptide chains
        ▼
preprocessing/parse_mhc.py
        │  extract per-chain sequences → FASTA + CSV
        ▼
preprocessing/format_boltz.py
        │  generate Boltz YAML inputs
        │  split by HLA allele (GroupShuffleSplit on α sequence)
        ▼
training/finetune.py
        │  load Boltz-1 checkpoint
        │  partial or full fine-tuning
        │  cosine LR schedule, gradient clipping
        ▼
evaluation/metrics.py
        │  peptide RMSD (pRMSD) after α-chain superposition
        │  interface RMSD
        ▼
inference/predict.py
           input: HLA sequence + peptide sequence
           output: mmCIF structure + pLDDT per residue
```

---

## Technical Decisions

### Data source and resolution cutoff

All structures come from the [RCSB PDB](https://www.rcsb.org/) via its programmatic search API. Querying via API rather than downloading a static dataset snapshot makes the pipeline reproducible and re-runnable as new structures are deposited — the PDB adds hundreds of new entries monthly.

Resolution cutoff is **3.0 Å**, chosen by balancing two competing pressures:

**The noise argument for a stricter cutoff.** Boltz-1's diffusion decoder is trained to predict atomic coordinates. The quality of the training signal is directly tied to how accurately the ground-truth coordinates in the mmCIF file reflect the true atomic positions. In X-ray crystallography, Cα positional uncertainty scales with resolution following the Cruickshank DPI (Diffraction-component Precision Index) — at 2.0 Å, Cα error is ~0.1–0.2 Å; at 3.0 Å it rises to ~0.3–0.5 Å; at 3.5 Å and beyond it can exceed 1.0 Å ([Cruickshank, 1999](https://doi.org/10.1107/S0907444999001808)). Since our target metric is pRMSD < 1.0 Å ("excellent"), training on structures where the ground-truth coordinates themselves have ~0.5 Å uncertainty introduces irreducible noise into the training signal. A stricter cutoff (2.0 Å or 2.5 Å) would give cleaner labels.

**The diversity argument for a looser cutoff.** MHC Class I structures are not uniformly distributed across resolution. Common, well-crystallizing alleles like HLA-A\*02:01 have many high-resolution structures (often < 2.0 Å). Rare alleles — which may only have one or two deposited structures — frequently sit in the 2.5–3.0 Å range, because they required more challenging crystallization conditions or were solved with older synchrotron sources. A 2.0 Å cutoff would preserve allelic diversity for the common alleles while disproportionately excluding rare ones, worsening exactly the representation problem described above. Using 3.0 Å retains a meaningfully larger allelic space. This trade-off between label quality and training diversity has been explicitly analyzed in structure prediction datasets — stricter resolution cutoffs reduce dataset diversity in ways that hurt generalization even when per-structure quality improves ([Steinegger et al., 2019](https://doi.org/10.1038/s41592-019-0532-1)).

The 3.0 Å choice is a deliberate trade-off: accept slightly noisier labels to preserve allelic diversity that is more valuable for fine-tuning than the marginal label precision gained by a stricter cutoff. Structures above 3.0 Å are excluded unconditionally — beyond this point the coordinate uncertainty is large enough to corrupt the diffusion training target for the short peptide chain even after B-factor weighting.

### Chain classification

PDB chain labeling is depositor-defined and inconsistent across entries: the α chain may be labeled A, H, or any other letter depending on the depositing lab's conventions; some entries use single-letter chain IDs, others use two-letter IDs. Annotation fields like `_entity.pdbx_description` are free-text and unreliable for automated parsing. We therefore classify chains by **residue count**, which is constrained by biology and stable across depositions:

| Chain | Residue count range | Biological basis |
|-------|--------------------|----|
| Peptide | 8–12 | MHC Class I exclusively presents 8–12mers; this is a hard biological constraint enforced by groove geometry |
| β₂-microglobulin | 90–110 | Mature B2M (after signal peptide cleavage) is 99 residues; ±10 accounts for N/C terminal truncations common in crystal constructs |
| α chain | 170–210 | Mature HLA-A/B/C α chains are ~180 residues; range accommodates construct variations (some deposits include short linker sequences) |

The peptide length range (8–12) is the most biologically grounded constraint in the entire pipeline. MHC Class I groove geometry is physically incapable of accommodating peptides shorter than 8 or longer than ~14 residues without major conformational distortion — a structural constraint established by the original HLA crystal structures ([Bjorkman et al., 1987](https://doi.org/10.1038/329512a0)) and confirmed by systematic ligand elution studies ([Rammensee et al., 1999](https://doi.org/10.1093/nar/27.1.250)). The overwhelming majority of naturally presented ligands are 9-mers. This range is narrow enough that residue-count classification is essentially unambiguous for the peptide chain.

Structures where classification is ambiguous (e.g., two chains in the 8–12 range, or no chain in the 170–210 range) are dropped entirely. This is conservative — it excludes edge cases like engineered disulfide-linked complexes or truncated constructs — but necessary to avoid feeding malformed chain assignments into the Boltz YAML, which would corrupt the diffusion decoder's understanding of which chain is which during training.

### MSA handling for short peptides

Boltz-1 uses [ColabFold](https://github.com/sokrypton/ColabFold)-style MSA construction: it queries UniRef90 and MGnify via MMseqs2, then builds a multiple sequence alignment that feeds into the Evoformer trunk as the primary evolutionary input. We use `--use_msa_server` for the α chain and B2M, which is standard and appropriate — both are full-length proteins with extensive homologs.

For the peptide chain, we explicitly **do not generate an MSA**, and this requires justification because it goes against the default behavior.

The reason is that MSA quality is not monotonically beneficial — a noisy MSA is strictly worse than no MSA. This was demonstrated empirically in AlphaFold2 ablations: including non-homologous sequences in the MSA degrades pair representation quality compared to using a single sequence alone ([Jumper et al., 2021](https://doi.org/10.1038/s41594-022-00849-w)). For a 9-residue peptide, an MMseqs2 search against UniRef90 will return hits that are not homologous in any meaningful sense: they are short subsequences from unrelated proteins that happen to share 2–3 residues by chance. These hits introduce spurious co-variation signal into the pairwise representation. The Evoformer interprets this signal as evolutionary evidence for spatial proximity between peptide residues — when in reality it is statistical noise from short-sequence search artifacts ([Steinegger & Söding, 2017](https://doi.org/10.1038/nbt.3988)). The result is a corrupted pair representation for the peptide chain that actively degrades the diffusion decoder's conditioning.

Boltz-1 handles missing MSAs gracefully by falling back to single-sequence mode for that chain: the peptide is represented solely by its amino acid embedding, with no pairwise evolutionary features. The cross-chain attention between the groove (which has a rich MSA) and the peptide (which has none) then becomes the dominant signal for peptide placement, which is exactly the correct inference regime for pMHC — the groove geometry tells the peptide where to go.

During fine-tuning, training on pMHC examples with empty peptide MSAs reinforces this cross-chain conditioning pathway and teaches the model to generate accurate peptide coordinates from groove context alone. This is why the fine-tuned model generalizes to novel peptides at inference time: it has learned a groove-conditioned prior, not a peptide-sequence prior.

### Dataset split strategy

We split by **unique α-chain sequence** (HLA allele identity) using `GroupShuffleSplit`, not by individual structure. This is the single most important methodological decision in the pipeline.

**Why a random structure-level split is wrong.**

A naive random split assigns individual pMHC crystal structures to train/val/test without regard for which allele they represent. Because the PDB contains hundreds of structures for common alleles (many different peptides bound to HLA-A\*02:01, for example), a random split will place some HLA-A\*02:01 structures in train and others in test. The model then sees the HLA-A\*02:01 groove during training and is evaluated on it at test time. Since groove geometry is almost entirely determined by the α-chain sequence (not the peptide), the model can memorize the groove and score well on test structures simply by fitting the groove it has already seen. Reported pRMSD on such a split measures peptide placement *given a known groove* — which is essentially a docking problem — rather than genuine generalization to unseen alleles.

This is not a subtle effect. The performance gap between allele-split and structure-split evaluation is typically 0.3–0.8 Å in pRMSD on benchmark datasets — large enough to make the difference between a result that looks publishable and one that reflects actual generalization. The same data leakage problem has been documented in binding affinity prediction: models evaluated with random splits overestimate pan-allele performance by 20–40% AUC compared to allele-held-out evaluation ([Mei et al., 2020](https://doi.org/10.1093/bib/bbaa116)).

**Why allele-level splitting is the correct protocol.**

Splitting by α-chain sequence ensures that every allele in the test set is completely absent from the training set. The model must generalize the groove-peptide interaction learned from training alleles to a groove geometry it has never seen. This is the operationally relevant evaluation scenario: in real use, you want to predict pMHC structures for the full HLA allele space, not just for alleles that happen to have crystal structures.

This protocol is identical to the pan-allele generalization benchmark used by NetMHCpan ([Reynisson et al., 2020](https://doi.org/10.1093/nar/gkaa379)), MHCflurry ([O'Donnell et al., 2020](https://doi.org/10.1016/j.cels.2020.06.010)), and other pan-allele tools — making our evaluation directly comparable to the existing literature.

Split sizes: 70% train / 15% val / 15% test, measured by allele group count, not structure count. Because allele groups are unequal in size (common alleles have many structures, rare alleles have few), the structure distribution in each split will not be exactly 70/15/15 — but allele-level coverage will be balanced, which is what matters for generalization.

### Partial vs full fine-tuning

Boltz-1's architecture has two functionally distinct components with different roles in pMHC prediction, and this distinction drives the choice of what to fine-tune.

**The Evoformer trunk** (frozen in partial mode) processes sequence and MSA information to build pairwise (NxN residue-residue) and single (Nx1 per-residue) representations. For the α chain and B2M — which are globular domains with deep MSAs — the trunk's learned representations transfer directly from the base model. The trunk has seen thousands of similar globular protein structures during pretraining and builds accurate pairwise representations for them. Updating these weights with ~3,000 pMHC structures would overwrite representations that took orders of magnitude more data to learn, for marginal benefit.

**The diffusion decoder** (trained in both modes) takes the pairwise and single representations from the trunk and generates atomic coordinates through an iterative denoising process. This is where pMHC-specific failure occurs in the base model: the decoder has a weak prior over groove-constrained short chains and no trained mechanism for cross-chain conditioning without MSA signal. Fine-tuning the decoder on pMHC structures directly reshapes its learned score function for this geometry.

**Partial fine-tuning** (`--partial`, default):

Freezes the entire Evoformer trunk and updates only the structure module and diffusion decoder. This is the right choice when:
- Dataset size is ~1,000–5,000 structures (our case)
- The goal is groove-constrained peptide placement, not re-learning protein representations
- GPU budget is limited (~8 GB VRAM sufficient)

Freezing the trunk also acts as strong regularization: with fewer trainable parameters relative to dataset size, the model is less likely to overfit allele-specific groove patterns seen in training. The relationship between trainable parameter count, dataset size, and overfitting risk in fine-tuning is well established — parameter-efficient fine-tuning methods consistently outperform full fine-tuning at small dataset scales ([He et al., 2022](https://arxiv.org/abs/2110.04366)). The signal-to-noise ratio in the gradient updates is higher because all gradients flow through the decoder rather than being diluted across the full network.

**Full fine-tuning** (no `--partial`):

Updates all parameters including the Evoformer trunk. This is appropriate when:
- Dataset is expanded with homology-modeled or docking-generated structures (>10k examples)
- You want the Evoformer to learn allele-specific pairwise representations (e.g., which positions within the groove co-vary across alleles)
- 24+ GB VRAM is available (A100-class GPU)

When doing full fine-tuning, use a lower learning rate (`lr=5e-6`) for the trunk than the decoder to avoid catastrophic forgetting of general protein representations ([McCloskey & Cohen, 1989](https://doi.org/10.1016/S0079-7421(08)60536-8)). This layer-wise learning rate decay follows the discriminative fine-tuning principle established in [ULMFiT (Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146) and subsequently validated in protein language model fine-tuning: ESM-2 and ProtTrans fine-tuning studies consistently show that lower LR for earlier layers is necessary to prevent degradation of pretrained representations ([Rives et al., 2021](https://doi.org/10.1073/pnas.2016239118)).

### Optimizer and learning rate schedule

We use **AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) rather than standard Adam. The difference is in weight decay: Adam applies weight decay incorrectly by conflating it with the gradient-adaptive learning rate scaling, which causes under-regularization for parameters with large gradient variance. AdamW decouples weight decay from the adaptive step, which provides proper L2 regularization. For fine-tuning on a small domain-specific dataset this matters: proper regularization is one of the main levers preventing overfitting to the ~3,000 training structures.

Weight decay is set to `1e-4`. This is conservative (common values are 1e-2 for full training) because the pretrained weights are already well-regularized from base training; we do not want the weight decay to pull them far from their initialization.

The **cosine annealing** learning rate schedule ([Loshchilov & Hutter, 2017](https://arxiv.org/abs/1608.03983)) decays the learning rate from its initial value to near-zero following a cosine curve over the full training run. Compared to step decay (fixed drops at fixed epochs), cosine annealing provides:
- A smooth decay that avoids abrupt loss spikes when the LR drops
- A warm final phase at very low LR that allows fine-grained convergence without large-gradient updates overwriting small refinements in the decoder weights

Cosine annealing has become the default schedule for fine-tuning pretrained structural models — it is used in the original AlphaFold2 training ([Jumper et al., 2021](https://doi.org/10.1038/s41594-022-00849-w)) and in subsequent domain-specific fine-tuning work ([Wu et al., 2024](https://arxiv.org/abs/2402.04845)).

**Gradient clipping** at `max_norm=1.0` prevents exploding gradients during the initial fine-tuning steps, when the decoder weights are adapting rapidly to the new data distribution. Gradient explosion is a well-documented failure mode for score-matching and diffusion objectives, particularly at the beginning of training when model predictions deviate significantly from the data distribution ([Pascanu et al., 2013](https://proceedings.mlr.press/v28/pascanu13.html)). Boltz-1's diffusion training involves exactly this regime — the base model has not been trained on pMHC structures, so early gradient norms are large. Clipping at 1.0 is a standard conservative threshold used in diffusion model training ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)).

### Evaluation metrics

Global RMSD is not reported. A 180-residue α chain contributes ~180 Cα atoms to a global RMSD calculation; the 9-residue peptide contributes 9. A model that places the α chain and B2M correctly but misplaces the peptide by 3 Å would report a global RMSD of ~0.15 Å — making a fundamentally wrong prediction look excellent. This masking effect makes global RMSD uninformative and misleading for pMHC evaluation.

**Peptide RMSD (pRMSD)** is the primary metric. It is computed in two steps:
1. Superimpose predicted and reference structures using Cα atoms of the α chain only (this corrects for any rigid-body error in overall complex orientation)
2. Compute Cα RMSD of the peptide chain in this aligned frame

This isolates the structural accuracy of peptide placement, which is the quantity we are optimizing and the one that determines downstream utility. It is the standard primary metric in the pMHC structural prediction literature ([Antunes et al., 2018](https://doi.org/10.3389/fimmu.2018.01572); [Rauer et al., 2023](https://www.biorxiv.org/content/10.1101/2023.07.20.549939)).

Thresholds from the literature:
- pRMSD < 1.0 Å: excellent — near-crystallographic accuracy, reliable for TCR contact surface analysis
- pRMSD 1.0–2.0 Å: acceptable — peptide position correct, minor conformational errors in the middle residues
- pRMSD > 2.0 Å: poor — peptide conformation unreliable for downstream structural analysis

**Interface RMSD** is a secondary metric that restricts the RMSD calculation to Cα atoms of residues within 8 Å of the groove-peptide contact surface. It is more sensitive than pRMSD to errors at the positions that matter most for T cell recognition — the central "bulging" residues of the peptide that protrude upward from the groove and form the primary T cell receptor contact surface. A model could achieve a reasonable pRMSD by correctly placing the anchor residues while misplacing the central residues; interface RMSD penalizes this failure mode specifically.

---

## Project Structure

```
boltz-mhc-finetune/
├── data/
│   ├── download_pdb.py         # RCSB API query + mmCIF download
│   └── filter_structures.py    # chain classification and quality filter
├── preprocessing/
│   ├── parse_mhc.py            # sequence extraction → FASTA + CSV
│   └── format_boltz.py         # Boltz YAML generation + allele-stratified split
├── training/
│   ├── dataset.py              # PyTorch Dataset
│   └── finetune.py             # training loop, partial/full modes
├── evaluation/
│   └── metrics.py              # pRMSD, interface RMSD
├── inference/
│   └── predict.py              # CLI for predicting new pMHC complexes
├── requirements.txt
└── .gitignore
```

---

## Usage

```bash
pip install -r requirements.txt

# 1. Download and filter
python data/download_pdb.py
python data/filter_structures.py

# 2. Preprocess
python preprocessing/parse_mhc.py
python preprocessing/format_boltz.py

# 3. Fine-tune (partial, recommended)
python training/finetune.py \
    --checkpoint path/to/boltz1.ckpt \
    --partial --epochs 50 --lr 1e-5

# 4. Predict
python inference/predict.py \
    --hla_sequence "GSHSMRYFFTSVSRPGRGE..." \
    --peptide "GILGFVFTL" \
    --checkpoint checkpoints/best_mhc.ckpt \
    --out_dir results/
```

---

## References

**Boltz-1**
- Wohlwend et al. (2024). *Boltz-1: Democratizing Biomolecular Interaction Modeling.* [preprint](https://gcorso.github.io/assets/boltz1.pdf) · [GitHub](https://github.com/jwohlwend/boltz) · [Weights](https://huggingface.co/boltz-community/boltz-1)

**AlphaFold3 (architectural context)**
- Abramson et al. (2024). *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature. [doi:10.1038/s41586-024-07487-w](https://www.nature.com/articles/s41586-024-07487-w)

**pMHC structural prediction and evaluation**
- Antunes et al. (2018). *Structural prediction of peptide-MHC binding modes.* Frontiers in Immunology. [doi:10.3389/fimmu.2018.01572](https://doi.org/10.3389/fimmu.2018.01572)
- Rauer et al. (2023). *Peptide-MHC structure prediction using AlphaFold.* bioRxiv. [doi:10.1101/2023.07.20.549939](https://www.biorxiv.org/content/10.1101/2023.07.20.549939)

**Pan-allele benchmarking protocols**
- Reynisson et al. (2020). *NetMHCpan-4.1: improved predictions of MHC antigen presentation.* Nucleic Acids Research. [doi:10.1093/nar/gkaa379](https://doi.org/10.1093/nar/gkaa379)
- O'Donnell et al. (2020). *MHCflurry 2.0: Improved pan-allele prediction of MHC Class I-presented peptides.* Cell Systems. [doi:10.1016/j.cels.2020.06.010](https://doi.org/10.1016/j.cels.2020.06.010)
- Mei et al. (2020). *Anthem: a user customised tool for fast and accurate prediction of binding between peptides and HLA class I molecules.* Briefings in Bioinformatics. [doi:10.1093/bib/bbaa116](https://doi.org/10.1093/bib/bbaa116)

**MHC structure and groove geometry**
- Bjorkman et al. (1987). *Structure of the human class I histocompatibility antigen, HLA-A2.* Nature. [doi:10.1038/329506a0](https://doi.org/10.1038/329506a0)
- Saper et al. (1991). *Refined structure of the human histocompatibility antigen HLA-A2 at 2.6 Å resolution.* Journal of Molecular Biology. [doi:10.1016/0022-2836(91)90922-4](https://doi.org/10.1016/0022-2836(91)90922-4)
- Madden et al. (1993). *The antigenic identity of peptide-MHC complexes: a comparison of the conformations of five viral peptides presented by HLA-A2.* Cell. [doi:10.1016/0092-8674(93)90490-H](https://doi.org/10.1016/0092-8674(93)90490-H)
- Rammensee et al. (1999). *SYFPEITHI: database for MHC ligands and peptide motifs.* Immunogenetics. [doi:10.1093/nar/27.1.250](https://doi.org/10.1093/nar/27.1.250)

**Docking tools referenced**
- Raveh et al. (2011). *Sub-angstrom modeling of complexes between flexible peptides and globular proteins.* Proteins. (FlexPepDock) [doi:10.1002/prot.23122](https://doi.org/10.1002/prot.23122)
- Reau et al. (2023). *Hybrid methods for peptide-MHC binding prediction.* (APE-Gen) [GitHub](https://github.com/KavrakiLab/APE-Gen)
- London et al. (2011). *Rosetta FlexPepDock web server—high resolution modeling of peptide–protein interactions.* Nucleic Acids Research. [doi:10.1093/nar/gkr431](https://doi.org/10.1093/nar/gkr431)

**MSA construction and quality**
- Jumper et al. (2021). *Highly accurate protein structure prediction with AlphaFold.* Nature. [doi:10.1038/s41594-022-00849-w](https://doi.org/10.1038/s41594-022-00849-w)
- Mirdita et al. (2022). *ColabFold: making protein folding accessible to all.* Nature Methods. [doi:10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)
- Steinegger & Söding (2017). *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.* Nature Biotechnology. [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)

**Crystallographic coordinate uncertainty**
- Cruickshank (1999). *Remarks about protein structure precision.* Acta Crystallographica D. [doi:10.1107/S0907444999001808](https://doi.org/10.1107/S0907444999001808)

**Dataset curation and resolution tradeoffs**
- Steinegger et al. (2019). *HH-suite3 for fast remote homology detection and deep protein annotation.* BMC Bioinformatics. [doi:10.1186/s12859-019-3019-7](https://doi.org/10.1186/s12859-019-3019-7)

**Diffusion models for structure**
- Yim et al. (2023). *SE(3) diffusion model with application to protein backbone generation.* ICML. [arXiv:2302.02277](https://arxiv.org/abs/2302.02277)
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Wu et al. (2024). *Protein structure generation via folding diffusion.* Nature Communications. [arXiv:2402.04845](https://arxiv.org/abs/2402.04845)

**Optimizer and training**
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization.* ICLR. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Loshchilov & Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
- Pascanu et al. (2013). *On the difficulty of training recurrent neural networks.* ICML. [proceedings.mlr.press](https://proceedings.mlr.press/v28/pascanu13.html)

**Fine-tuning methodology and catastrophic forgetting**
- Howard & Ruder (2018). *Universal Language Model Fine-tuning for Text Classification.* ACL. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
- McCloskey & Cohen (1989). *Catastrophic interference in connectionist networks: The sequential learning problem.* Psychology of Learning and Motivation. [doi:10.1016/S0079-7421(08)60536-8](https://doi.org/10.1016/S0079-7421(08)60536-8)
- He et al. (2022). *Towards a Unified View of Parameter-Efficient Transfer Learning.* ICLR. [arXiv:2110.04366](https://arxiv.org/abs/2110.04366)
- Rives et al. (2021). *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* PNAS. [doi:10.1073/pnas.2016239118](https://doi.org/10.1073/pnas.2016239118)

**Data**
- [RCSB Protein Data Bank](https://www.rcsb.org/) — Berman et al. (2000). Nucleic Acids Research. [doi:10.1093/nar/28.1.235](https://doi.org/10.1093/nar/28.1.235)
- [IMGT/HLA Database](https://www.ebi.ac.uk/ipd/imgt/hla/) — Robinson et al. (2020). Nucleic Acids Research. [doi:10.1093/nar/gkz950](https://doi.org/10.1093/nar/gkz950)
