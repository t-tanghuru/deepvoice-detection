
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import whisper

from model.model import WhisperEncoderLCNN


def pad_or_trim_mel_local(mel, target_frames=3000):
    if mel.dim() == 4:
        mel = mel.squeeze(1)
    frames = mel.shape[-1]
    if frames < target_frames:
        mel = F.pad(mel, (0, target_frames - frames))
    elif frames > target_frames:
        mel = mel[..., :target_frames]
    return mel


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    threshold = float(checkpoint.get("threshold", 0.5))
    config = checkpoint.get("config", {}) or {}

    model = WhisperEncoderLCNN(
        whisper_size=config.get("whisper_size", "base"),
        freeze_whisper=bool(config.get("freeze_whisper", True)),
        dropout=float(config.get("dropout", 0.5)),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, threshold, config


def normalize_heatmap(x):
    x = x - x.min()
    if float(x.max()) > 0:
        x = x / x.max()
    return x


def smooth_heatmap(x, kernel_size=7):
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    return F.avg_pool2d(
        x.unsqueeze(0).unsqueeze(0),
        kernel_size=(3, kernel_size),
        stride=1,
        padding=(1, pad),
    ).squeeze(0).squeeze(0)


def save_png(mel, heatmap, output_path, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(float(meta["duration_sec"]), mel.shape[-1] / 100.0)
    extent = [0, duration, 0, mel.shape[0]]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].imshow(mel, aspect="auto", origin="lower", extent=extent, cmap="magma")
    axes[0].set_title("Whisper log-Mel spectrogram")
    axes[0].set_ylabel("Mel bin")

    axes[1].imshow(mel, aspect="auto", origin="lower", extent=extent, cmap="gray_r")
    axes[1].imshow(heatmap, aspect="auto", origin="lower", extent=extent, cmap="jet", alpha=0.55)
    axes[1].set_title(
        f"Log-Mel gradient evidence heatmap | result={meta['result']} | "
        f"fake_prob={meta['fake_prob']:.4f}"
    )
    axes[1].set_ylabel("Mel bin")

    im = axes[2].imshow(heatmap, aspect="auto", origin="lower", extent=extent, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Normalized log-Mel gradient heatmap")
    axes[2].set_ylabel("Mel bin")
    axes[2].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[2], fraction=0.025, pad=0.015)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate(audio_path, model_path, output_path, target_class="fake", threshold=None):
    audio_path = Path(audio_path)
    model_path = Path(model_path)
    output_path = Path(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_threshold, config = load_model(model_path, device)
    threshold = ckpt_threshold if threshold is None else float(threshold)

    # 학습 때는 encoder가 frozen/no_grad였지만,
    # 시각화에서는 입력 log-Mel에 대한 gradient가 필요하므로 no_grad 우회.
    model.freeze_whisper = False
    for p in model.parameters():
        p.requires_grad_(False)

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration_sec = len(audio) / sr
    raw_mel = whisper.log_mel_spectrogram(torch.from_numpy(audio).float())
    visible_frames = min(raw_mel.shape[-1], 3000)

    mel = pad_or_trim_mel_local(raw_mel.unsqueeze(0), 3000).detach().clone().to(device)
    mel.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(mel)
    prob = torch.softmax(logits, dim=1)[0]

    real_prob = float(prob[0].detach().cpu())
    fake_prob = float(prob[1].detach().cpu())
    pred = 1 if fake_prob >= threshold else 0
    result = "FAKE" if pred == 1 else "REAL"

    target_idx = 1 if target_class == "fake" else 0
    logits[0, target_idx].backward()

    grad = mel.grad.detach()[0].float().cpu()
    mel_cpu = mel.detach()[0].float().cpu()

    # log-Mel 입력값과 gradient의 곱을 사용한 판별 근거 heatmap
    heatmap = (grad * mel_cpu).abs()
    heatmap = heatmap[:, :visible_frames]
    mel_cpu = mel_cpu[:, :visible_frames]

    heatmap = smooth_heatmap(heatmap, 7)
    heatmap = normalize_heatmap(heatmap)

    meta = {
        "audio_path": str(audio_path),
        "model_path": str(model_path),
        "output_path": str(output_path),
        "model_type": "whisper_encoder_lcnn",
        "visualization": "log-Mel input gradient-based evidence heatmap",
        "threshold": threshold,
        "result": result,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "confidence": fake_prob if pred == 1 else real_prob,
        "target_class": target_class,
        "duration_sec": duration_sec,
        "visible_frames": int(visible_frames),
    }

    save_png(mel_cpu.numpy(), heatmap.numpy(), output_path, meta)

    output_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(output_path.with_suffix(".heatmap.npy"), heatmap.numpy())
    np.save(output_path.with_suffix(".mel.npy"), mel_cpu.numpy())

    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--target-class", choices=["fake", "real"], default="fake")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    generate(
        audio_path=args.audio_path,
        model_path=args.model_path,
        output_path=args.output_path,
        target_class=args.target_class,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
