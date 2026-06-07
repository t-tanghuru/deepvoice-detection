import torch
import whisper
import librosa
from model import LCNN


def predict(audio_path, model_path="best_model_tts_v6.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    whisper_model = whisper.load_model("base")

    audio, _ = librosa.load(audio_path, sr=16000)
    audio = torch.from_numpy(audio).float()
    mel = whisper.log_mel_spectrogram(audio)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0).unsqueeze(0)
    elif mel.dim() == 3:
        mel = mel.unsqueeze(0)

    with torch.no_grad():
        out = model(mel.to(device))
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(out, dim=1).item()

    label = "FAKE" if pred == 1 else "REAL"
    confidence = prob[0][pred].item()
    print(f"결과: {label} (confidence: {confidence:.4f})")
    return pred, confidence


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python predict.py <audio_path>")
    else:
        predict(sys.argv[1])
