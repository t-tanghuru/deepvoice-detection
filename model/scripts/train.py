import argparse
import glob
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

from model.model import LCNN


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


DEFAULT_DATA_DIR = PROJECT_ROOT / "model" / "features"
DEFAULT_PRETRAINED_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model_tts_v8_thr020.pt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model_tts_strong.pt"


class PTDataset(Dataset):
    def __init__(self, real_files, fake_files):
        self.files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        mel = torch.load(path, map_location="cpu")
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        return mel, label


def pad_collate(batch):
    mels, labels = zip(*batch)
    max_len = max(m.shape[-1] for m in mels)
    padded = torch.zeros(len(mels), 1, mels[0].shape[1], max_len)
    for i, m in enumerate(mels):
        padded[i, 0, :, : m.shape[-1]] = m[0] if m.dim() == 3 else m
    return padded, torch.tensor(labels)


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def optional_path(value):
    return str(value) if value else ""


def collect_pt_files(data_dir, dirs):
    files = []
    for name in dirs:
        path = Path(data_dir) / name
        print(f"{name}: 스캔 중...", flush=True)
        if not path.exists():
            print(f"{name}: 0 (폴더 없음)", flush=True)
            continue
        found = sorted(str(Path(entry.path)) for entry in os.scandir(path) if entry.is_file() and entry.name.endswith(".pt"))
        print(f"{name}: {len(found)}", flush=True)
        files.extend(found)
    return files


def load_pretrained(model, path, device):
    if not path or str(path).lower() in {"none", "null", "false"}:
        print("pretrained 로드 생략")
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"pretrained 파일 없음: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"pretrained 로드: {path}")


def predict_scores(model, loader, device):
    model.eval()
    labels = []
    fake_scores = []
    with torch.no_grad():
        for mel, label in loader:
            out = model(mel.to(device))
            prob = torch.softmax(out, dim=1)[:, 1]
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
        raise ValueError("real/fake train/val 중 비어 있는 폴더가 있습니다. 위 폴더별 개수를 확인하세요.")

    train_loader = DataLoader(
        PTDataset(real_train, fake_train),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        PTDataset(real_val, fake_val),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=pad_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LCNN().to(device)
    load_pretrained(model, config["pretrained_path"], device)

    class_counts = torch.tensor([len(real_train), len(fake_train)], dtype=torch.float)
    if config["balanced_loss"]:
        weights = class_counts.sum() / (2 * class_counts)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        print(f"class weights: real={weights[0]:.4f}, fake={weights[1]:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    wandb_run = None
    if config["wandb"]:
        import wandb

        wandb_run = wandb.init(project=config["wandb_project"], name=config["run_name"])

    best_f1 = 0.0
    best_info = None
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for step, (mel, label) in enumerate(train_loader):
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(mel)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (step + 1) % config["log_every"] == 0:
                print(f"Epoch {epoch + 1} | Step {step + 1}/{len(train_loader)} | loss: {loss.item():.4f}")

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
            f"EER: {eer*100:.2f}% @ {eer_threshold:.4f} | "
            f"min-DCF: {norm_min_dcf:.4f}"
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
                },
                output_path,
            )
            print(f"  -> best 저장: {output_path} (f1: {best_f1:.4f})")

    if wandb_run:
        wandb_run.finish()

    print(f"학습 완료 | best f1: {best_f1:.4f}")
    if best_info and best_info["f1"] < config["target_f1"]:
        print(f"목표 F1 {config['target_f1']:.2f} 미달. hard TTS 데이터 추가/holdout 분리/threshold 재탐색이 필요합니다.")


def parse_args():
    parser = argparse.ArgumentParser(description="LCNN TTS 탐지 모델 fine-tuning")
    parser.add_argument(
        "--deepvoice-dir",
        default="",
        help=(
            "선택: deepvoice 루트 경로. 지정하면 기본 data/pretrained/output 경로가 "
            "<deepvoice-dir>/whisper_features, best_model_tts_v8.pt, best_model_tts_strong.pt로 설정됩니다."
        ),
    )
    parser.add_argument("--data-dir", default="", help=f"기본값: {DEFAULT_DATA_DIR}")
    parser.add_argument("--pretrained-path", default="", help=f"기본값: {DEFAULT_PRETRAINED_PATH}")
    parser.add_argument("--output-path", default="", help=f"기본값: {DEFAULT_OUTPUT_PATH}")
    parser.add_argument("--run-name", default="tts_lcnn_strong")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-f1", type=float, default=0.90)
    parser.add_argument("--real-train-dirs", default="real_train")
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
    args.pretrained_path = optional_path(args.pretrained_path) or str(
        deepvoice / "best_model_tts_v8.pt" if deepvoice else DEFAULT_PRETRAINED_PATH
    )
    args.output_path = optional_path(args.output_path) or str(
        deepvoice / "best_model_tts_strong.pt" if deepvoice else DEFAULT_OUTPUT_PATH
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
        print(f"에러: {exc}")
        raise SystemExit(1)
