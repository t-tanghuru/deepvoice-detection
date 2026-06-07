import os, glob, torch
import numpy as np
from sklearn.metrics import classification_report
from model import LCNN


def evaluate(model_path, real_val_dir, holdout_dir, holdout_speakers, device):
    model = LCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("모델 로드 완료")

    all_preds = []
    all_labels = []

    real_preds = []
    for f in glob.glob(f"{real_val_dir}/*.pt"):
        mel = torch.load(f)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0).unsqueeze(0)
        elif mel.dim() == 3:
            mel = mel.unsqueeze(0)
        with torch.no_grad():
            pred = torch.argmax(model(mel.to(device)), dim=1).item()
        real_preds.append(pred)

    real_acc = sum(1 for p in real_preds if p == 0) / len(real_preds)
    print(f"real ({len(real_preds)}개): accuracy {real_acc:.4f}")
    all_preds.extend(real_preds)
    all_labels.extend([0] * len(real_preds))

    for name in holdout_speakers:
        files = [f for f in glob.glob(f"{holdout_dir}/*.pt") if os.path.basename(f).startswith(name)]
        if not files:
            files = glob.glob(f"{holdout_dir}/{name}/*.pt")
        if not files:
            print(f"{name}: 파일 없음")
            continue

        preds = []
        for f in files:
            mel = torch.load(f)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0).unsqueeze(0)
            elif mel.dim() == 3:
                mel = mel.unsqueeze(0)
            with torch.no_grad():
                pred = torch.argmax(model(mel.to(device)), dim=1).item()
            preds.append(pred)

        acc = sum(1 for p in preds if p == 1) / len(preds)
        print(f"{name} ({len(preds)}개): fake 탐지율 {acc:.4f}")
        all_preds.extend(preds)
        all_labels.extend([1] * len(preds))

    print("\n=== 전체 holdout 결과 ===")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"]))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(
        model_path="/content/drive/MyDrive/deepvoice/best_model_tts_v6.pt",
        real_val_dir="/content/features/real_val",
        holdout_dir="/content/drive/MyDrive/deepvoice/whisper_features/holdout_v2",
        holdout_speakers=["Roger", "Will", "Callum"],
        device=device
    )
