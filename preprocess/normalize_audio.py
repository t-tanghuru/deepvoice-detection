import librosa
import soundfile as sf
import os
from pathlib import Path

def normalize_audio(input_path, output_path, sr=16000):
    try:
        y, _ = librosa.load(input_path, sr=sr)
        sf.write(output_path, y, sr)
        return True
    except Exception as e:
        print(f"오류: {input_path} - {e}")
        return False

def normalize_folder(input_dir, output_dir, sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    files = list(Path(input_dir).glob("*"))
    files = [f for f in files if f.suffix in ['.wav', '.mp3']]
    
    success = 0
    for i, f in enumerate(files):
        output_path = os.path.join(output_dir, f.stem + ".wav")
        if normalize_audio(str(f), output_path, sr):
            success += 1
        if (i+1) % 100 == 0:
            print(f"{i+1}/{len(files)} 처리 중...")
    
    print(f"완료: {success}/{len(files)}개")

if __name__ == "__main__":
    print("실제 음성 정규화 중...")
    normalize_folder("data/real", "data/real_normalized")
    
    print("합성 음성 정규화 중...")
    normalize_folder("data/fake", "data/fake_normalized")
