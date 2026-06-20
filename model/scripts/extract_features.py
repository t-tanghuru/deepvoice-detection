import argparse
import glob
import os

import librosa
import torch
import whisper


def extract_and_save(src_pattern, dst_dir, whisper_size="base", sr=16000, recursive=False):
    model = whisper.load_model(whisper_size)
    model.eval()

    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(src_pattern, recursive=recursive))
    print(f"{len(files)}개 처리 시작")

    for i, path in enumerate(files, start=1):
        try:
            out_path = os.path.join(dst_dir, os.path.splitext(os.path.basename(path))[0] + ".pt")
            if os.path.exists(out_path):
                continue
            audio, _ = librosa.load(path, sr=sr)
            mel = whisper.log_mel_spectrogram(torch.from_numpy(audio).float())
            torch.save(mel, out_path)
            if i % 500 == 0:
                print(f"{i}/{len(files)} 완료")
        except Exception as exc:
            print(f"에러: {path} - {exc}")

    print(f"완료: {dst_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="오디오 파일을 Whisper log-mel .pt 특징으로 변환합니다.")
    parser.add_argument("src_pattern", help="입력 오디오 glob 패턴. 예: data/fake_train/*.wav")
    parser.add_argument("dst_dir", help="출력 .pt 폴더. 예: features/fake_train")
    parser.add_argument("--whisper-size", default="base")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_and_save(
        args.src_pattern,
        args.dst_dir,
        whisper_size=args.whisper_size,
        sr=args.sr,
        recursive=args.recursive,
    )
