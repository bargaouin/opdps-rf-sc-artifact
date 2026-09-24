# OpDPS-RF-SC: Anonymous Reproducibility Artifact

This repository contains the anonymous implementation and reproducibility
artifacts accompanying our submission on **single-channel RF signal separation
using a complex neural-operator separator, residual diffusion refinement, and
DPM-Solver++ inference**.

The artifact contains the model implementation, experiment configurations,
training and evaluation scripts, Slurm launch scripts, tests, and the
paper-facing outputs used to reproduce the reported experiments.

## Repository Structure

```text
configs/          Experiment and model configurations
paper_artifacts/  Small paper-facing results and reproducibility artifacts
scripts/          Training, evaluation, and utility scripts
slurm/            Example Slurm launch scripts
src/              Core model, diffusion, and evaluation implementation
tests/            Unit and smoke tests



## Environment Setup

We recommend using a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-rfchallenge.txt

# Fourier-Enhanced Single-Channel Radio-Frequency Signal Separation

This repository contains the code, configurations, checkpoints, evaluation
scripts, and experiment logs for our ICASSP submission.

The paper studies a narrow architectural question:

> Does global Fourier-domain mixing add useful information beyond a strong
> local temporal separator for single-channel complex waveform separation?

The contribution is not a new Fourier transform or a new spectral primitive.
Instead, we perform a controlled architecture study comparing local-only,
spectral-only, capacity-matched local, and hybrid local+Fourier models under
the same data-generation and evaluation protocol.

---

## 1. Main models

We evaluate four deterministic separators.

### Local Temporal Separator (LTS)

A one-dimensional temporal encoder-decoder with:

- residual temporal convolutions,
- three stride-2 downsampling stages,
- skip connections,
- known interference-class conditioning,
- residual-to-input prediction.

Parameter count: approximately 8.94M.

### Capacity-Matched Temporal Separator (CM-LTS)

The same temporal architecture as LTS, widened to approximately match the
capacity of FETS.

Parameter count: approximately 15.00M.

This control asks whether the FETS improvement can be explained by parameter
count alone.

### Direct Fourier Neural Operator (Direct FNO)

A spectral-only control that removes the temporal encoder-decoder and applies
Fourier-operator blocks directly at a single resolution.

Parameter count: approximately 0.38M.

This model is a structural control, not a capacity-matched competitor.

### Fourier-Enhanced Temporal Separator (FETS)

FETS retains the temporal encoder-decoder and inserts six hybrid operator
blocks at the compressed bottleneck.

Each hybrid block contains:

1. a local temporal convolution branch, and
2. a global Fourier branch.

The two branches are combined before the residual block update.

Parameter count: approximately 14.84M.

---

## 2. What the paper claims — and what it does not claim

The evidence supports the following claim:

> Under the fixed-length, known-interferer RFChallenge setting used here,
> the hybrid FETS architecture outperforms both a nearly parameter-matched
> local-only model and a spectral-only control, with the largest improvement
> occurring for structured CommSignal3 interference at low SINR.

We do **not** claim that:

- Fourier processing is universally superior to local processing;
- the bottleneck is the uniquely optimal location for Fourier mixing;
- the observed NMSE improvement implies a BER or FER improvement;
- the method handles unknown interference classes;
- the reported bootstrap intervals measure training-seed uncertainty;
- the current architecture has been optimized over spectral rank, mode count,
  or Fourier placement.

These are deliberately left as future experiments.

---

## 3. Dataset and fixed evaluation protocol

We use the single-channel RFChallenge dataset.

The signal of interest is CommSignal2.

The two interference classes are:

- EMISignal1
- CommSignal3

All complex waveforms are converted to two real channels:

x = [Re(y), Im(y)].

The fixed evaluation cache contains 2,200 mixtures:

- 2 interference classes,
- 11 SINR values,
- 100 examples per interference/SINR condition.

SINR values range from +3 dB to -12 dB in 1.5 dB increments.

All models are evaluated on exactly the same frozen cache.

This makes the model comparisons paired with respect to the generated
mixture, interference class, and SINR.

---

## 4. Data split and checkpoint selection

The source-recording split is performed before any window extraction or
mixture construction.

Training windows therefore cannot be alternate crops of an evaluation
recording.

For reproducibility, the repository contains the exact split and cache
construction scripts used in the paper.

Checkpoint selection is based on the source-level holdout criterion and is
performed before the final fixed-cache evaluation.

Selected checkpoints:

| Model | Selected step |
|---|---:|
| FETS | 90k |
| LTS | 110k |
| CM-LTS | 110k |
| Direct FNO | 112k |

The different selected steps are intentional: each architecture is evaluated
at its own best holdout checkpoint rather than at a common terminal step.

The exact checkpoint-selection histories are available under:

`logs/`

and the corresponding configuration files are under:

`configs/`

---

## Training-seed limitation

Each reported architecture was trained once.

The paired bootstrap does not address optimization or initialization
variability.

For that reason, we do not interpret the small difference between LTS and
CM-LTS as meaningful.

The central comparison is the substantially larger FETS versus CM-LTS gap on
identical evaluation mixtures.

Repeated independent training runs of FETS and CM-LTS are the most important
statistical extension of this study.
