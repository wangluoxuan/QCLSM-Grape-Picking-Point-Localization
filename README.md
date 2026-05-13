# QCLSM-Grape-Picking-Point-Localization


This repository provides the public code and example materials associated with the manuscript submitted to **The Visual Computer**:

**QC-CVPP:Quality-Controlled Collaborative Visual Perception Pipeline for Robotic Fruit Harvesting and Precise Localization**

The repository is intended to improve the transparency and reproducibility of the proposed visual perception pipeline for grape picking-point localization. The current public version covers the complete experimental workflow of the proposed pipeline, including Grounding DINO-based ROI generation, SegFormer-SFG segmentation, and SJAR-Loc picking-point localization, with corresponding scripts, interfaces, and descriptions.

After acceptance, the final archived release will be preserved on Zenodo with a permanent DOI.

 1. Overview

The proposed pipeline consists of three main stages:

1. Grounding DINO-based ROI generation for open-vocabulary grape cluster localization.
2. SegFormer-SFG segmentation for fine-grained stem and fruit segmentation.
3. SJAR-Loc picking-point localization based on stem structure, stable junction analysis, and geometric constraints.

The repository provides source code, example data format, running instructions, and input/output interface descriptions for the above pipeline.

---

2. Repository Structure

The repository is organized as follows:

```text
QCLSM-Grape-Picking-Point-Localization/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── experiments/
│   ├── DINO/
│   │   └── run_roi_generation.py
│   │
│   ├── SegFormer_SFG/
│   │   ├── README.md
│   │   ├── config.py
│   │   └── train_segformer_sfg_public.py
│   │
│   └── SJAR_LOC/
│       ├── README.md
│       └── run_sjar_loc.py
│
└── examples/
    ├── images/
    ├── masks/
    └── outputs/
```


The main folders are described below:

experiments/DINO/: wrapper script for Grounding DINO-based ROI generation.
experiments/SegFormer_SFG: SegFormer-SFG segmentation training, testing, and inference interface.
experiments/SJAR_LOC/: SJAR-Loc picking-point localization interface.
examples/: representative input images, example masks, and visualization outputs.

3. Installation

Clone this repository:

Create a Python environment:
conda create -n qclsm python=3.9
conda activate qclsm


Install dependencies:
pip install -r requirements.txt

Recommended environment:
Python >= 3.8
PyTorch >= 1.10
transformers
opencv-python
numpy
Pillow
scikit-learn
tqdm

4. Data Preparation

For the SegFormer-SFG segmentation stage, the dataset should be organized as follows:


data/
├── mask_aug/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   │
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   │
│   └── test/
│       ├── images/
│       └── masks/


Each image should have a corresponding mask file with the same file name. For example:


data/mask_aug/train/images/sample_001.jpg
data/mask_aug/train/masks/sample_001.png
The mask should be a single-channel label image. The label definitions are:

| Label ID | Class |
|---|---|
| 0 | Background |
| 1 | Stem |
| 2 | Fruit |

---

5. Dataset
The GBPD dataset can be accessed through [GBPD Dataset](https://doi.org/10.1016/j.compag.2023.108362). The Sweet Pepper dataset is available at [Sweet Pepper Dataset](https://datasetninja.com/sweet-pepper), and the Synthetic Plants dataset is available at [Synthetic Plants Dataset](https://datasetninja.com/synthetic-plants).Due to data-sharing restrictions and project confidentiality requirements, the full self-built dataset is not publicly released at this stage.

6. Grounding DINO ROI Generation

The Grounding DINO stage is used to generate candidate grape cluster regions. This repository provides a wrapper-style script. Users should install the official Grounding DINO implementation and download the corresponding pretrained weights separately.

Example command:
python experiments/DINO/run_roi_generation.py

The default text prompt is:
grape cluster

The output of this stage includes:
1. candidate bounding boxes in JSON format
2. expanded ROI regions
3. ROI visualization results

7. SegFormer-SFG Training

SegFormer-SFG is the core segmentation module released in this repository. It is used to segment grape stems and fruits within candidate grape cluster regions.

Example training command:
python experiments/SegFormer_SFG/train_segformer_sfg_public.py \
  --data-root ./data/mask_aug \
  --output-dir ./outputs/segformer_sfg \
  --epochs 80 \
  --batch-size 4 \
  --lr 6e-5


8. License

This project is released under the MIT License.

9. Acknowledgement

This public repository is organized for academic transparency and reproducibility. The Grounding DINO-based ROI generation stage is implemented as a wrapper around the official Grounding DINO model. The SegFormer-SFG segmentation stage is implemented based on the SegFormer semantic segmentation framework.
