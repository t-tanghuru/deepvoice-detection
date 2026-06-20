import argparse
import glob
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from model.model import LCNN


DEFAULT_LOCAL_DATA_DIR = PROJECT_ROOT / "model" / "features"
DEFAULT_LOCAL_MODEL_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model_tts_v8_thr020.pt"


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    threshold = 0.5
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        threshold = float(checkpoint.get("threshold", threshold))
    else:
        state_dict = checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model = LCNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, threshold


def load_mel(path):
    mel = torch.load(path, map_location="cpu")
    if mel.dim() == 2:
        mel = mel.unsqueeze(0).unsqueeze(0)
    elif mel.dim() == 3:
        mel = mel.unsqueeze(0)
    return mel


class PTFilesDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = torch.load(self.files[idx], map_location="cpu")
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        return mel


def pad_collate(mels):
    max_len = max(m.shape[-1] for m in mels)
    padded = torch.zeros(len(mels), 1, mels[0].shape[1], max_len)
    for i, mel in enumerate(mels):
        padded[i, 0, :, : mel.shape[-1]] = mel[0] if mel.dim() == 3 else mel
    return padded


def collect_files(base_dir, dirs):
    files = []
    for name in dirs:
        path = Path(base_dir) / name
        print(f"{name}: 스캔 중...", flush=True)
        if not path.exists():
            print(f"{name}: 0 (폴더 없음)", flush=True)
            continue
        found = sorted(str(entry) for entry in path.iterdir() if entry.is_file() and entry.name.endswith(".pt"))
        print(f"{name}: {len(found)}", flush=True)
        files.extend(found)
    return files


def predict_scores(model, files, device, batch_size=32, num_workers=0):
    scores = []
    loader = DataLoader(
        PTFilesDataset(files),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate,
    )
    total = len(loader)
    for step, mel in enumerate(loader, start=1):
        with torch.no_grad():
            out = model(mel.to(device))
            scores.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        if step % 10 == 0 or step == total:
            print(f"예측 진행: {step}/{total} batches", flush=True)
    return np.array(scores)


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx]), float(fpr[idx]), float(fnr[idx])


def compute_min_dcf(labels, scores, p_target=0.5, c_miss=1.0, c_fa=1.0):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    idx = int(np.nanargmin(dcf))
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    normalized = float(dcf[idx] / default_cost) if default_cost > 0 else float(dcf[idx])
    return float(dcf[idx]), normalized, float(thresholds[idx])


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_threshold = load_model(args.model_path, device)
    threshold = checkpoint_threshold if args.threshold is None else args.threshold
    print(f"모델: {args.model_path}")
    print(f"threshold: {threshold:.2f}")

    real_files = collect_files(args.data_dir, comma_list(args.real_dirs))
    fake_files = collect_files(args.data_dir, comma_list(args.fake_dirs))
    if not real_files or not fake_files:
        raise ValueError("평가할 real/fake .pt 파일이 부족합니다.")

    print("\nreal 예측 중...", flush=True)
    real_scores = predict_scores(model, real_files, device, args.batch_size, args.num_workers)
    print("fake 예측 중...", flush=True)
    fake_scores = predict_scores(model, fake_files, device, args.batch_size, args.num_workers)
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.array([0] * len(real_scores) + [1] * len(fake_scores))
    preds = (scores >= threshold).astype(int)

    print("\n=== 평가 결과 ===")
    print(classification_report(labels, preds, target_names=["real", "fake"], zero_division=0))
    print("confusion matrix [[real->real, real->fake], [fake->real, fake->fake]]")
    print(confusion_matrix(labels, preds))
    print(f"F1: {f1_score(labels, preds, zero_division=0):.4f}")

    eer, eer_threshold, eer_fpr, eer_fnr = compute_eer(labels, scores)
    min_dcf, norm_min_dcf, min_dcf_threshold = compute_min_dcf(
        labels,
        scores,
        p_target=args.dcf_p_target,
        c_miss=args.dcf_c_miss,
        c_fa=args.dcf_c_fa,
    )
    print("\n=== ASVspoof 계열 지표 ===")
    print(f"EER: {eer * 100:.2f}% (threshold={eer_threshold:.4f}, FPR={eer_fpr:.4f}, FNR={eer_fnr:.4f})")
    print(
        "min-DCF: "
        f"{min_dcf:.4f} / normalized {norm_min_dcf:.4f} "
        f"(threshold={min_dcf_threshold:.4f}, "
        f"Ptarget={args.dcf_p_target}, Cmiss={args.dcf_c_miss}, Cfa={args.dcf_c_fa})"
    )

    best = (0.5, 0.0)
    for candidate in np.arange(0.20, 0.81, 0.01):
        candidate_preds = (scores >= candidate).astype(int)
        candidate_f1 = f1_score(labels, candidate_preds, zero_division=0)
        if candidate_f1 > best[1]:
            best = (float(candidate), float(candidate_f1))
    print(f"best threshold scan: {best[0]:.2f}, F1: {best[1]:.4f}")

    if best[1] >= args.target_f1:
        print(f"목표 F1 {args.target_f1:.2f} 달성 가능")
    else:
        print(f"목표 F1 {args.target_f1:.2f} 미달")


def parse_args():
    parser = argparse.ArgumentParser(description="LCNN TTS 탐지 모델 평가")
    parser.add_argument("--data-dir", default=str(DEFAULT_LOCAL_DATA_DIR))
    parser.add_argument("--model-path", default=str(DEFAULT_LOCAL_MODEL_PATH))
    parser.add_argument("--real-dirs", default="real_val")
    parser.add_argument("--fake-dirs", default="fake_val,elevenlabs_val,holdout,holdout_v2")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-f1", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dcf-p-target", type=float, default=0.5)
    parser.add_argument("--dcf-c-miss", type=float, default=1.0)
    parser.add_argument("--dcf-c-fa", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        evaluate(parse_args())
    except Exception as exc:
        print(f"에러: {exc}")
        raise SystemExit(1)
