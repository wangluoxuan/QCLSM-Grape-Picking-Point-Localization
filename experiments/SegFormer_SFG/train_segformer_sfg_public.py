"""
Simplified public implementation of SegFormer-SFG for fruit/stem segmentation.

This script provides the core model components used in the paper:
- FSA-RG style foreground structure attention gate (FSAGate)
- FSGM foreground structure-guided modulation module
- SegFormer-B2 based semantic segmentation training interface

Dataset format:
    data/mask_aug/
    ├── train/images, train/masks
    ├── val/images,   val/masks
    └── test/images,  test/masks

Mask labels:
    0 = background
    1 = stem
    2 = fruit

This public version is intended for academic reference and method reproduction.
"""

import argparse
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


# ---------------------------
# Basic configuration
# ---------------------------
NUM_CLASSES = 3
ID2LABEL = {0: "background", 1: "stem", 2: "fruit"}
LABEL2ID = {"background": 0, "stem": 1, "fruit": 2}
EVAL_CLASS_IDS = [1, 2]


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------
# Dataset
# ---------------------------
def resolve_split_dirs(root: str, split: str) -> Tuple[str, str]:
    split_dir = os.path.join(root, split)
    img_dir = os.path.join(split_dir, "images")
    mask_candidates = [
        os.path.join(split_dir, "masks"),
        os.path.join(split_dir, "mask"),
        os.path.join(split_dir, "labels"),
        os.path.join(split_dir, "label"),
    ]

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    mask_dir = next((d for d in mask_candidates if os.path.isdir(d)), None)
    if mask_dir is None:
        raise FileNotFoundError(f"Mask directory not found under: {split_dir}")

    return img_dir, mask_dir


def find_mask(mask_dir: str, stem: str) -> str:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        path = os.path.join(mask_dir, stem + ext)
        if os.path.exists(path):
            return path
    return ""


class StemFruitDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, img_size: int = 512, train: bool = True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.train = train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image_exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(image_exts)]
        files.sort()
        self.files = [f for f in files if find_mask(mask_dir, os.path.splitext(f)[0])]

        if not self.files:
            raise RuntimeError(f"No matched image-mask pairs found in {img_dir} and {mask_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(self.img_dir, name)
        mask_path = find_mask(self.mask_dir, stem)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.train and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        mask = mask.astype(np.int64)
        mask[(mask < 0) | (mask > 2)] = 0

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).contiguous()
        return img, mask


def build_loaders(data_root: str, batch_size: int, num_workers: int, img_size: int, seed: int):
    train_img, train_mask = resolve_split_dirs(data_root, "train")
    val_img, val_mask = resolve_split_dirs(data_root, "val")
    test_img, test_mask = resolve_split_dirs(data_root, "test")

    train_ds = StemFruitDataset(train_img, train_mask, img_size=img_size, train=True)
    val_ds = StemFruitDataset(val_img, val_mask, img_size=img_size, train=False)
    test_ds = StemFruitDataset(test_img, test_mask, img_size=img_size, train=False)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    print(f"Dataset size: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader


# ---------------------------
# SegFormer-SFG modules
# ---------------------------
class FSAGate(nn.Module):
    """Foreground structure attention gate."""

    def __init__(self, channels: int, reduction: int = 4, use_dwconv: bool = True):
        super().__init__()
        mid = max(8, channels // reduction)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self._lambda_raw = nn.Parameter(torch.tensor(-0.85, dtype=torch.float32))
        self.use_dwconv = use_dwconv

        if use_dwconv:
            self.dwconv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

    @property
    def lam(self) -> torch.Tensor:
        return torch.sigmoid(self._lambda_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.attn(x))
        attn_dilated = F.max_pool2d(attn, kernel_size=3, stride=1, padding=1)
        attn_mix = (1.0 - self.lam) * attn + self.lam * attn_dilated
        x = x * (1.0 + self.alpha * attn_mix)
        if self.use_dwconv:
            x = x + self.dwconv(x)
        return x


class FSGM(nn.Module):
    """Foreground structure-guided modulation module."""

    def __init__(
        self,
        channels: int,
        num_classes: int = 3,
        fg_indices: List[int] = None,
        dilate_k: int = 5,
        strip_k: int = 7,
        coef_clip: float = 0.10,
        init_beta_pos: float = 0.04,
        init_beta_neg: float = 0.015,
    ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.fg_indices = fg_indices or [1, 2]
        self.coef_clip = coef_clip

        self.aux_head = nn.Conv2d(channels, num_classes, kernel_size=1, bias=True)

        dilate_k = dilate_k if dilate_k % 2 == 1 else dilate_k + 1
        strip_k = strip_k if strip_k % 2 == 1 else strip_k + 1
        self.pool = nn.MaxPool2d(kernel_size=dilate_k, stride=1, padding=dilate_k // 2)

        self.dw_h = nn.Conv2d(channels, channels, kernel_size=(1, strip_k), padding=(0, strip_k // 2), groups=channels, bias=False)
        self.dw_v = nn.Conv2d(channels, channels, kernel_size=(strip_k, 1), padding=(strip_k // 2, 0), groups=channels, bias=False)
        self.dw_3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        self.beta_pos = nn.Parameter(torch.tensor(init_beta_pos, dtype=torch.float32))
        self.beta_neg = nn.Parameter(torch.tensor(init_beta_neg, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aux_logits = self.aux_head(x)
        prob = torch.softmax(aux_logits - aux_logits.max(dim=1, keepdim=True).values, dim=1)

        gate = prob[:, self.fg_indices, :, :].sum(dim=1, keepdim=True)
        gate = self.pool(gate)
        gate = 0.5 * gate + 0.5 * self.pool(gate)

        delta = self.dw_h(x) + self.dw_v(x) + self.dw_3(x)
        delta = self.act(self.bn(self.pw(delta)))

        coef = self.beta_pos * gate - self.beta_neg * (1.0 - gate)
        coef = self.coef_clip * torch.tanh(coef / max(1e-6, self.coef_clip))
        return x + coef * delta


def _infer_decoder_hidden_size(model: nn.Module) -> int:
    hidden = getattr(getattr(model, "config", None), "decoder_hidden_size", None)
    if hidden is not None:
        return int(hidden)
    classifier = getattr(model.decode_head, "classifier", None)
    if isinstance(classifier, nn.Conv2d):
        return int(classifier.in_channels)
    if isinstance(classifier, nn.Sequential):
        for module in classifier.modules():
            if isinstance(module, nn.Conv2d):
                return int(module.in_channels)
    raise AttributeError("Cannot infer SegFormer decoder hidden size.")


def apply_fsa_fsgm(model: nn.Module, use_dwconv: bool = True) -> nn.Module:
    if not hasattr(model, "decode_head"):
        raise AttributeError("The model has no decode_head attribute.")

    hidden = _infer_decoder_hidden_size(model)
    original_classifier = model.decode_head.classifier

    fsa = FSAGate(hidden, use_dwconv=use_dwconv)
    fsgm = FSGM(hidden, num_classes=NUM_CLASSES, fg_indices=[1, 2])

    model.decode_head.fsa_gate = fsa
    model.decode_head.fsgm = fsgm
    model.decode_head.classifier = nn.Sequential(fsa, fsgm, original_classifier)
    return model


def build_model(backbone: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return apply_fsa_fsgm(model, use_dwconv=True)


# ---------------------------
# Loss
# ---------------------------
def dice_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, class_ids: List[int], smooth: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    losses = []
    for cls_id in class_ids:
        pred = probs[:, cls_id, :, :].reshape(logits.size(0), -1)
        target = (labels == cls_id).float().reshape(logits.size(0), -1)
        inter = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        losses.append(1.0 - dice.mean())
    return sum(losses) / len(losses)


def boundary_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, class_ids: List[int]) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=probs.dtype,
        device=probs.device,
    ).view(1, 1, 3, 3)

    losses = []
    for cls_id in class_ids:
        pred = probs[:, cls_id:cls_id + 1, :, :]
        target = (labels == cls_id).float().unsqueeze(1)
        pred_edge = torch.abs(F.conv2d(pred, kernel, padding=1)).clamp(0, 1)
        gt_edge = torch.abs(F.conv2d(target, kernel, padding=1)).clamp(0, 1)
        losses.append(F.l1_loss(pred_edge, gt_edge))
    return sum(losses) / len(losses)


def compute_seg_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ce_weight: float = 1.0,
    dice_weight: float = 0.5,
    boundary_weight: float = 0.15,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels)
    dice = dice_loss_from_logits(logits, labels, EVAL_CLASS_IDS)
    boundary = boundary_loss_from_logits(logits, labels, EVAL_CLASS_IDS)
    return ce_weight * ce + dice_weight * dice + boundary_weight * boundary


# ---------------------------
# Metrics
# ---------------------------
def confusion_for_class(preds: torch.Tensor, labels: torch.Tensor, cls_id: int) -> Dict[str, float]:
    pred_bin = preds == cls_id
    label_bin = labels == cls_id
    tp = (pred_bin & label_bin).sum().item()
    fp = (pred_bin & ~label_bin).sum().item()
    fn = (~pred_bin & label_bin).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn}


def metrics_from_confusion(cf: Dict[str, float]) -> Dict[str, float]:
    tp, fp, fn = cf["tp"], cf["fp"], cf["fn"]
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {"precision": precision, "recall": recall, "iou": iou, "dice": dice}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cfs = {cls_id: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for cls_id in EVAL_CLASS_IDS}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = compute_seg_loss(logits, labels)
        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        for cls_id in EVAL_CLASS_IDS:
            cf = confusion_for_class(preds, labels, cls_id)
            for key in total_cfs[cls_id]:
                total_cfs[cls_id][key] += cf[key]

    results = {"loss": total_loss / len(loader.dataset)}
    ious, dices = [], []
    for cls_id in EVAL_CLASS_IDS:
        name = ID2LABEL[cls_id]
        metrics = metrics_from_confusion(total_cfs[cls_id])
        results[f"{name}_iou"] = metrics["iou"]
        results[f"{name}_dice"] = metrics["dice"]
        results[f"{name}_precision"] = metrics["precision"]
        results[f"{name}_recall"] = metrics["recall"]
        ious.append(metrics["iou"])
        dices.append(metrics["dice"])
    results["miou"] = float(np.mean(ious))
    results["mdice"] = float(np.mean(dices))
    return results


def save_visual_examples(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: str, num_images: int = 3) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    saved = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)

            imgs = (images * std + mean).clamp(0, 1).cpu()
            preds = preds.cpu()
            labels = labels.cpu()

            for i in range(images.size(0)):
                if saved >= num_images:
                    return
                img_np = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gt = np.zeros_like(img_np)
                pr = np.zeros_like(img_np)
                gt[..., 0] = (labels[i].numpy() == 1) * 255
                gt[..., 1] = (labels[i].numpy() == 2) * 255
                pr[..., 0] = (preds[i].numpy() == 1) * 255
                pr[..., 1] = (preds[i].numpy() == 2) * 255
                canvas = np.concatenate([img_np, (0.7 * img_np + 0.3 * gt).astype(np.uint8), (0.7 * img_np + 0.3 * pr).astype(np.uint8)], axis=1)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_dir, f"vis_{saved + 1:02d}.png"), canvas)
                saved += 1


# ---------------------------
# Training
# ---------------------------
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    vis_dir = os.path.join(args.output_dir, "pred_vis")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(
        args.data_root,
        args.batch_size,
        args.num_workers,
        args.img_size,
        args.seed,
    )

    model = build_model(args.backbone).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_stem_iou = -1.0
    best_path = os.path.join(ckpt_dir, "segformer_sfg_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = model(pixel_values=images)
                logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = compute_seg_loss(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, device)
        stem_iou = val_metrics["stem_iou"]

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"mIoU={val_metrics['miou']:.4f} | mDice={val_metrics['mdice']:.4f} | "
            f"stem_IoU={val_metrics['stem_iou']:.4f} | fruit_IoU={val_metrics['fruit_iou']:.4f}"
        )

        if stem_iou > best_stem_iou:
            best_stem_iou = stem_iou
            torch.save({"model_state_dict": model.state_dict(), "best_stem_iou": best_stem_iou, "epoch": epoch}, best_path)
            print(f"Saved best checkpoint to {best_path}")

    print("Loading best checkpoint for testing...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"[Test] loss={test_metrics['loss']:.4f} | mIoU={test_metrics['miou']:.4f} | "
        f"mDice={test_metrics['mdice']:.4f} | stem_IoU={test_metrics['stem_iou']:.4f} | "
        f"fruit_IoU={test_metrics['fruit_iou']:.4f}"
    )

    if args.save_vis:
        save_visual_examples(model, test_loader, device, vis_dir, num_images=args.num_vis)
        print(f"Saved visual examples to {vis_dir}")


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public SegFormer-SFG training script")
    parser.add_argument("--data-root", type=str, default="./data/mask_aug")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--backbone", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision training")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--save-vis", action="store_true", help="Save a few prediction visualizations")
    parser.add_argument("--num-vis", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
