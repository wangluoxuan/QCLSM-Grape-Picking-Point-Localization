# QCLSM-Grape-Picking-Point-Localization

This repository provides a simplified public implementation of the core segmentation module used in the quality-controlled large-and-small model collaborative visual perception pipeline for grape picking-point localization.

## Associated Paper

This code is directly associated with the manuscript submitted to *The Visual Computer*:

**Quality Controlled Large and Small Model Collaborative Visual Perception Pipeline for Grape Picking Point Localization**

The repository is provided to improve the transparency and reproducibility of the proposed visual perception pipeline. The current public version includes the core implementation of SegFormer-SFG, including FSA-RG, FSGM, the segmentation loss, dataset interface, and basic training and evaluation pipeline.

After acceptance, the final code release will be archived on Zenodo with a permanent DOI.

## Overview

The proposed pipeline contains three stages:

1. Open-vocabulary candidate region generation using Grounding DINO.
2. Fine-grained fruit and stem segmentation using SegFormer-SFG.
3. Topology-aware picking-point localization using SJAR-Loc.

This public repository focuses on the SegFormer-SFG segmentation stage.

## Key Modules

The released code includes the following components:

- `FSAGate`: foreground structure-aware response enhancement module.
- `FSGM`: foreground structure-guided modulation module for slender stem structure modeling.
- `SegFormer-SFG`: SegFormer-B2 with FSA-RG and FSGM modules.
- `CE + Dice + Boundary loss`: hybrid segmentation loss for fruit and stem parsing.
- Basic dataset loading, training, validation, and visualization utilities.

## Requirements

Please install the following dependencies:

```bash
pip install -r requirements.txt
