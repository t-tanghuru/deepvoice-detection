import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.inference import predict_audio_file
from model.paths import DEFAULT_MODEL_PATH


def predict(audio_path, model_path, threshold=None):
    pred, confidence, fake_prob = predict_audio_file(audio_path, model_path=model_path, threshold=threshold)
    label = "FAKE" if pred == 1 else "REAL"
    print(f"결과: {label} | confidence: {confidence:.2f}% | fake_prob: {fake_prob:.4f}")
    return pred, fake_prob


def parse_args():
    parser = argparse.ArgumentParser(description="단일 오디오 TTS 탐지")
    parser.add_argument("audio_path")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.audio_path, args.model_path, args.threshold)
