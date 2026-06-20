import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import torch
import whisper

from model.model import WhisperEncoderLCNN


DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model_tts_whisper_encoder_lcnn.pt"


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    threshold = 0.5
    config = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        threshold = float(checkpoint.get("threshold", threshold))
        config = checkpoint.get("config", {}) or {}
    else:
        state_dict = checkpoint

    model = WhisperEncoderLCNN(
        whisper_size=config.get("whisper_size", "base"),
        freeze_whisper=bool(config.get("freeze_whisper", True)),
        dropout=float(config.get("dropout", 0.5)),
    ).to(device)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, threshold


def audio_to_mel(audio_path, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = whisper.log_mel_spectrogram(torch.from_numpy(audio).float())
    return mel.unsqueeze(0)


def predict(audio_path, model_path, threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_threshold = load_model(model_path, device)
    threshold = checkpoint_threshold if threshold is None else threshold
    mel = audio_to_mel(audio_path).to(device)

    with torch.no_grad():
        logits = model(mel)
        prob = torch.softmax(logits, dim=1)[0]
        real_prob = float(prob[0].item())
        fake_prob = float(prob[1].item())

    pred = 1 if fake_prob >= threshold else 0
    confidence = fake_prob if pred == 1 else real_prob
    label = "FAKE" if pred == 1 else "REAL"
    print(f"model: {model_path}")
    print(f"threshold: {threshold:.2f}")
    print(f"result: {label} | confidence: {confidence * 100:.2f}% | fake_prob: {fake_prob:.4f}")
    return pred, fake_prob


def parse_args():
    parser = argparse.ArgumentParser(description="Single-audio prediction with Whisper encoder + LCNN-style classifier")
    parser.add_argument("audio_path")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        predict(args.audio_path, args.model_path, args.threshold)
    except Exception as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
