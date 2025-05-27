# General Binding Affinity Guidance for Diffusion Models in Structure-Based Drug Design (DecompDiff Part)

![Figure 1](figs/fig1.png)

Official codebase for the paper:  
**"General Binding Affinity Guidance for Diffusion Models in Structure-Based Drug Design"**  
[[arXiv:2406.16821](https://arxiv.org/abs/2406.16821)]

---

**BADGER** is a general binding affinity guidance framework for diffusion models in structure-based drug discovery (SBDD). It introduces two complementary strategies:

- **Classifier Guidance:** Gradient-based plug-and-play guidance using a pretrained binding affinity classifier.
- **Classifier-Free Guidance:** Guidance integrated directly into the diffusion model's training, removing the need for external classifiers.

These methods enable general binding affinity-guided molecular design using diffusion models.

> This code builds heavily on [TargetDiff](https://github.com/guanjq/targetdiff) and [DecompDiff](https://github.com/bytedance/DecompDiff). We thank the authors for their contributions.

---

## 📦 Setup

### 1. Environment Setup

Create the conda environment:

```bash
conda env create -f BADGER.yml
```

---

### 2. Download Data & Checkpoints

#### 📁 Data

Please follow the instructions from [DecompDiff](https://github.com/bytedance/DecompDiff).  
Place the downloaded data under the `./data` directory.

#### 🧠 Checkpoints

Download pretrained checkpoints from:  
[checkpoints link](https://doi.org/10.5281/zenodo.15523148)

---

## 🚀 Usage

#### Train Classifier

```bash
python scripts/train_classifier.py configs/training_classifier.yml
```

#### Sample with Classifier

```bash
python scripts/sample_diffusion_decomp.py configs/sampling_drift.yml \
  --outdir {path_to_output_dir} --prior_mode {ref_prior,beta_prior}
```

#### Evaluate Samples

##### Use Pre-Sampled Molecules (for reproduction)

Download from:  
**[Zenodo placeholder link]** (to be updated)


```bash
python scripts/evaluate_diffusion.py output_eva_diff/decompdiff_paper_mol/unguide_ref_prior \
  --docking_mode {vina_dock,vina_score} --protein_root {path_to_protein_root e.g. data/test_set}
```

---

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{jian2024generalbindingaffinityguidance,
  title={General Binding Affinity Guidance for Diffusion Models in Structure-Based Drug Design},
  author={Yue Jian and Curtis Wu and Danny Reidenbach and Aditi S. Krishnapriyan},
  year={2024},
  eprint={2406.16821},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2406.16821}
}
```

please also cite the related foundational works:

```bibtex
@misc{guan20233dequivariantdiffusiontargetaware,
  title={3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction},
  author={Jiaqi Guan and Wesley Wei Qian and Xingang Peng and Yufeng Su and Jian Peng and Jianzhu Ma},
  year={2023},
  eprint={2303.03543},
  archivePrefix={arXiv},
  primaryClass={q-bio.BM},
  url={https://arxiv.org/abs/2303.03543}
}

@misc{guan2024decompdiffdiffusionmodelsdecomposed,
  title={DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design},
  author={Jiaqi Guan and Xiangxin Zhou and Yuwei Yang and Yu Bao and Jian Peng and Jianzhu Ma and Qiang Liu and Liang Wang and Quanquan Gu},
  year={2024},
  eprint={2403.07902},
  archivePrefix={arXiv},
  primaryClass={q-bio.BM},
  url={https://arxiv.org/abs/2403.07902}
}
```

---

Feel free to open issues or discussions for help or feedback!
