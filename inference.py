import torch
import sys
from preprocess.feature_extract import extract_melspectrogram
from model.cnn_classifier import DeepVoiceCNN

def predict(file_path):
    # 모델 로드
    model = DeepVoiceCNN()
    model.load_state_dict(torch.load("model/best_model.pt", map_location="cpu"))
    model.eval()
    
    # 특징 추출
    mel = extract_melspectrogram(file_path)
    if mel is None:
        print("음성 파일 로드 실패")
        return
    
    mel = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)
    
    # 추론
    with torch.no_grad():
        output = model(mel)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
    
    real_prob = prob[0][0].item() * 100
    fake_prob = prob[0][1].item() * 100
    
    print(f"파일: {file_path}")
    print(f"실제 음성 확률: {real_prob:.1f}%")
    print(f"합성 음성 확률: {fake_prob:.1f}%")
    print(f"판별 결과: {'합성 음성' if pred == 1 else '실제 음성'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python inference.py 음성파일.mp3")
    else:
        predict(sys.argv[1])
