import os, glob, librosa, soundfile as sf, numpy as np
from pydub import AudioSegment


def save_mp3(y, sr, out_path):
    temp_wav = out_path.replace(".mp3", "_temp.wav")
    sf.write(temp_wav, y, sr)
    AudioSegment.from_wav(temp_wav).export(out_path, format="mp3")
    os.remove(temp_wav)


def normalize(y):
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.95
    return y


def change_speed(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)


def change_pitch(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y, noise_level=0.003):
    return y + noise_level * np.random.randn(len(y))


def telephone_effect(y, sr):
    y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
    return librosa.resample(y_8k, orig_sr=8000, target_sr=sr)


def augment_directory(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(f"{src_dir}/**/*.mp3", recursive=True)
    print(f"증강 시작: {len(files)}개")
    count = 0
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        y, sr = librosa.load(f, sr=None)

        save_mp3(normalize(y), sr, f"{dst_dir}/{base}_original.mp3")
        for rate in [0.9, 1.1]:
            save_mp3(normalize(change_speed(y, rate)), sr, f"{dst_dir}/{base}_speed_{rate}.mp3")
        for step in [-1, 1]:
            save_mp3(normalize(change_pitch(y, sr, step)), sr, f"{dst_dir}/{base}_pitch_{step}.mp3")
        save_mp3(normalize(add_noise(y)), sr, f"{dst_dir}/{base}_noise.mp3")
        save_mp3(normalize(telephone_effect(y, sr)), sr, f"{dst_dir}/{base}_telephone.mp3")

        count += 1
        if count % 100 == 0:
            print(f"{count}개 처리 완료")

    print(f"증강 완료: {count * 7}개")
