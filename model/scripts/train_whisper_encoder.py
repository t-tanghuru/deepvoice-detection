import argparse
import os
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from model.model import WhisperEncoderLCNN


DEFAULT_DATA_DIR = PROJECT_ROOT / "model" / "features"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model_tts_whisper_encoder_lcnn.pt"


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def optional_path(value):
    return str(value) if value else ""


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_min_dcf(labels, scores, p_target=0.5, c_miss=1.0, c_fa=1.0):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    idx = int(np.nanargmin(dcf))
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    normalized = float(dcf[idx] / default_cost) if default_cost > 0 else float(dcf[idx])
    return float(dcf[idx]), normalized, float(thresholds[idx])


class PTDataset(Dataset):
    def __init__(self, real_files, fake_files):
        self.files = [(path, 0) for path in real_files] + [(path, 1) for path in fake_files]
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        mel = torch.load(path, map_location="cpu")
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        if mel.dim() != 2:
            raise ValueError(f"Expected 2D mel tensor, got {tuple(mel.shape)} from {path}")
        return mel.float(), label


def whisper_mel_collate(batch, target_frames=3000):
    mels, labels = zip(*batch)
    n_mels = mels[0].shape[0]
    padded = torch.zeros(len(mels), n_mels, target_frames)
    for idx, mel in enumerate(mels):
        frames = min(mel.shape[-1], target_frames)
        padded[idx, :, :frames] = mel[:, :frames]
    return padded, torch.tensor(labels)


def collect_pt_files(data_dir, dirs):
    files = []
    for name in dirs:
        path = Path(data_dir) / name
        print(f"{name}: scanning...", flush=True)
        if not path.exists():
            print(f"{name}: 0 (missing folder)", flush=True)
            continue
        found = sorted(str(Path(entry.path)) for entry in os.scandir(path) if entry.is_file() and entry.name.endswith(".pt"))
        print(f"{name}: {len(found)}", flush=True)
        files.extend(found)
    return files


def load_pretrained(model, path, device):
    if not path or str(path).lower() in {"none", "null", "false"}:
        print("pretrained load skipped")
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"pretrained file not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"pretrained loaded: {path}")


def predict_scores(model, loader, device):
    model.eval()
    labels = []
    fake_scores = []
    with torch.no_grad():
        for mel, label in loader:
            logits = model(mel.to(device))
            prob = torch.softmax(logits, dim=1)[:, 1]
            fake_scores.extend(prob.cpu().numpy())
            labels.extend(label.numpy())
    return np.array(labels), np.array(fake_scores)


def best_threshold(labels, fake_scores):
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.arange(0.20, 0.81, 0.01):
        preds = (fake_scores >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
            }
    return best


def train(config):
    print(f"model_type: whisper_encoder_lcnn")
    print(f"whisper_size: {config['whisper_size']}")
    print(f"freeze_whisper: {config['freeze_whisper']}")
    print(f"data_dir: {config['data_dir']}")
    print(f"pretrained_path: {config['pretrained_path']}")
    print(f"output_path: {config['output_path']}")

    real_train = collect_pt_files(config["data_dir"], config["real_train_dirs"])
    real_val = collect_pt_files(config["data_dir"], config["real_val_dirs"])
    fake_train = collect_pt_files(config["data_dir"], config["fake_train_dirs"])
    fake_val = collect_pt_files(config["data_dir"], config["fake_val_dirs"])

    print(f"train - real: {len(real_train)}, fake: {len(fake_train)}")
    print(f"val   - real: {len(real_val)}, fake: {len(fake_val)}")
    if not real_train or not fake_train or not real_val or not fake_val:
        raise ValueError("One or more real/fake train/val folders are empty.")

    train_loader = DataLoader(
        PTDataset(real_train, fake_train),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=whisper_mel_collate,
    )
    val_loader = DataLoader(
        PTDataset(real_val, fake_val),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=whisper_mel_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperEncoderLCNN(
        whisper_size=config["whisper_size"],
        freeze_whisper=config["freeze_whisper"],
        dropout=config["dropout"],
    ).to(device)
    load_pretrained(model, config["pretrained_path"], device)

    class_counts = torch.tensor([len(real_train), len(fake_train)], dtype=torch.float)
    if config["balanced_loss"]:
        weights = class_counts.sum() / (2 * class_counts)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        print(f"class weights: real={weights[0]:.4f}, fake={weights[1]:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])

    wandb_run = None
    if config["wandb"]:
        import wandb

        wandb_run = wandb.init(project=config["wandb_project"], name=config["run_name"], config=config)

    best_f1 = 0.0
    best_info = None
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        if config["freeze_whisper"]:
            model.encoder.eval()
        train_loss = 0.0
        for step, (mel, label) in enumerate(train_loader):
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (step + 1) % config["log_every"] == 0:
                print(f"Epoch {epoch + 1} | Step {step + 1}/{len(train_loader)} | loss: {loss.item():.4f}", flush=True)

        labels, fake_scores = predict_scores(model, val_loader, device)
        fixed_preds = (fake_scores >= 0.5).astype(int)
        fixed_f1 = f1_score(labels, fixed_preds, zero_division=0)
        tuned = best_threshold(labels, fake_scores)
        eer, eer_threshold = compute_eer(labels, fake_scores)
        min_dcf, norm_min_dcf, dcf_threshold = compute_min_dcf(labels, fake_scores)

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"loss: {train_loss / len(train_loader):.4f} | "
            f"val_f1@0.50: {fixed_f1:.4f} | "
            f"best_f1: {tuned['f1']:.4f} @ {tuned['threshold']:.2f} | "
            f"EER: {eer * 100:.2f}% @ {eer_threshold:.4f} | "
            f"min-DCF: {norm_min_dcf:.4f}",
            flush=True,
        )

        if wandb_run:
            wandb_run.log(
                {
                    "train_loss": train_loss / len(train_loader),
                    "val_f1_050": fixed_f1,
                    "val_best_f1": tuned["f1"],
                    "val_best_threshold": tuned["threshold"],
                    "val_precision": tuned["precision"],
                    "val_recall": tuned["recall"],
                    "val_eer": eer,
                    "val_min_dcf": norm_min_dcf,
                    "epoch": epoch + 1,
                }
            )

        if tuned["f1"] > best_f1:
            best_f1 = tuned["f1"]
            best_info = tuned
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "threshold": tuned["threshold"],
                    "val_f1": tuned["f1"],
                    "config": config,
                    "model_type": "whisper_encoder_lcnn",
                },
                output_path,
            )
            print(f"  -> best saved: {output_path} (f1: {best_f1:.4f})", flush=True)

    if wandb_run:
        wandb_run.finish()

    print(f"training complete | best f1: {best_f1:.4f}")
    if best_info and best_info["f1"] < config["target_f1"]:
        print(f"Target F1 {config['target_f1']:.2f} not reached.")


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen Whisper encoder + LCNN-style classifier training")
    parser.add_argument("--deepvoice-dir", default="")
    parser.add_argument("--data-dir", default="", help=f"default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--pretrained-path", default="none", help="optional same-architecture checkpoint")
    parser.add_argument("--output-path", default="", help=f"default: {DEFAULT_OUTPUT_PATH}")
    parser.add_argument("--run-name", default="tts_whisper_encoder_lcnn")
    parser.add_argument("--whisper-size", default="base")
    parser.add_argument("--freeze-whisper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-f1", type=float, default=0.90)
    parser.add_argument("--real-train-dirs", default="real_sampled_aug_train_balanced")
    parser.add_argument("--real-val-dirs", default="real_val")
    parser.add_argument(
        "--fake-train-dirs",
        default=(
            "fake_train,elevenlabs_train,elevenlabs_train_add,elevenlabs_train_add2,"
            "elevenlabs_train_add3,elevenlabs_train_lily"
        ),
    )
    parser.add_argument("--fake-val-dirs", default="fake_val,elevenlabs_val,holdout,holdout_v2")
    parser.add_argument("--balanced-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="deepvoice")
    args = parser.parse_args()

    deepvoice = Path(args.deepvoice_dir).expanduser() if args.deepvoice_dir else None
    args.data_dir = optional_path(args.data_dir) or str(deepvoice / "whisper_features" if deepvoice else DEFAULT_DATA_DIR)
    args.output_path = optional_path(args.output_path) or str(
        deepvoice / "best_model_tts_whisper_encoder_lcnn.pt" if deepvoice else DEFAULT_OUTPUT_PATH
    )

    config = vars(args)
    config["real_train_dirs"] = comma_list(args.real_train_dirs)
    config["real_val_dirs"] = comma_list(args.real_val_dirs)
    config["fake_train_dirs"] = comma_list(args.fake_train_dirs)
    config["fake_val_dirs"] = comma_list(args.fake_val_dirs)
    return config


if __name__ == "__main__":
    try:
        train(parse_args())
    except Exception as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
