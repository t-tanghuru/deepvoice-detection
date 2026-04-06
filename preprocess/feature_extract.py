import librosa
import numpy as np
import os

def extract_mfcc(file_path, n_mfcc=40, max_len=128):
    """음성 파일에서 MFCC 특징 추출"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # 길이 통일 (패딩 or 자르기)
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0,0),(0, max_len - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :max_len]
        
        return mfcc
    except Exception as e:
        print(f"오류: {file_path} - {e}")
        return None

def extract_melspectrogram(file_path, max_len=128):
    """음성 파일에서 Mel 스펙트로그램 추출"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        if mel_db.shape[1] < max_len:
            mel_db = np.pad(mel_db, ((0,0),(0, max_len - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :max_len]
        
        return mel_db
    except Exception as e:
        print(f"오류: {file_path} - {e}")
        return None

if __name__ == "__main__":
    # 테스트
    test_file = "data/fake/fake_0000.mp3"
    
    mfcc = extract_mfcc(test_file)
    mel = extract_melspectrogram(test_file)
    
    print(f"MFCC shape: {mfcc.shape}")
    print(f"Mel shape: {mel.shape}")
    print("특징 추출 완료")
