# Generative Modelling Case Study — GANs

Coursework for the Advanced Research Topics in Data Science module. The project
builds Generative Adversarial Networks (GANs) from scratch on synthetic 2D data
and then applies them to three real-world datasets.

## Contents

| File | Description |
|------|-------------|
| `Part1_GANs_from_Scratch.ipynb` | Part 1 — GANs from scratch on 2D data: sine-wave GAN, a noisy parametric curve, a 2D spiral, and an architecture comparison. |
| `Part2_1_OCTMNIST_DCGAN.ipynb` | Part 2.1 — DCGAN on OCTMNIST retinal images, evaluated with FID. Extension: conditional GAN (cGAN) generating a chosen class on demand. |
| `Part2_2_CICIDS_GAN.ipynb` | Part 2.2 — Tabular GAN generating synthetic network-traffic feature vectors (CICIDS 2017), compared with PCA and t-SNE. Extension: all-days data. |
| `Part2_3_QuickDraw_DCGAN.ipynb` | Part 2.3 — DCGAN on the QuickDraw "birthday cake" sketches, evaluated with FID. Extension: other categories (cat, house). |
| `Modelling case study.docx` | Written report. |

## Running the notebooks

All notebooks were developed in **Google Colab** with a **GPU runtime**
(Runtime → Change runtime type → T4 GPU). To run one:

1. Open the notebook in Colab.
2. Enable the GPU runtime.
3. Run all cells top to bottom.

The image notebooks (2.1, 2.3) need a GPU; Part 1 and Part 2.2 will run on CPU.

## Datasets

- **OCTMNIST** — downloaded automatically via the `medmnist` package (with a
  direct Zenodo mirror as a fallback).
- **CICIDS 2017** — the daily CSV files; the notebook loads the Wednesday file
  for the core task and all days for the extension. Placed on Google Drive and
  unzipped inside the notebook.
- **QuickDraw** — category `.npy` bitmap files pulled directly from Google's
  public QuickDraw storage.

## Main dependencies

`torch`, `torchvision`, `medmnist`, `scikit-learn`, `pandas`, `numpy`,
`matplotlib`, `scipy` (FID is computed with a standalone InceptionV3 feature
extractor). Most are preinstalled on Colab; the notebooks pip-install the rest.

## Methods overview

Each application follows the same pattern: load and explore the data, build a
generator/discriminator pair, train them adversarially while tracking both
losses, generate synthetic samples, and compare them to the real data both
visually and quantitatively (FID for images, PCA/t-SNE for the tabular data).
