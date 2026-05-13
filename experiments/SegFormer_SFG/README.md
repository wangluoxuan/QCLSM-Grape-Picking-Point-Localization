# SegFormer-SFG Training, Testing, and Inference

This folder contains the experimental code for the SegFormer-SFG segmentation stage. SegFormer-SFG is the fine-grained segmentation module in the proposed grape picking-point localization pipeline. It is mainly used to perform pixel-level segmentation of grape stems and fruits within candidate grape cluster regions.

In the complete pipeline, Grounding DINO first generates candidate grape cluster regions. SegFormer-SFG then performs fine-grained segmentation of stems and fruits inside these regions. Finally, SJAR-Loc computes the picking point based on the predicted segmentation masks.

## 1. Task Description

This stage is formulated as a semantic segmentation task. The class definitions are as follows:

| Label ID | Class |
|---|---|
| 0 | Background |
| 1 | Stem |
| 2 | Fruit |

The model takes RGB grape images or ROI images cropped from candidate boxes as input and outputs pixel-level segmentation masks for stem and fruit regions.

## 2. File Structure

The recommended file structure is:

```text
experiments/SegFormer_SFG/
├── README.md
├── config.py
├── train_segformer_sfg_public.py
