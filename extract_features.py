import os, glob, torch, whisper, librosa


def extract_and_save(src_pattern, dst_dir):
    model = whisper.load_model("base")
    model.eval()

    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(src_pattern)
    print(f"{len(files)}개 처리 시작")

    for i, f in enumerate(files):
        try:
            out_path = os.path.join(dst_dir, os.path.splitext(os.path.basename(f))[0] + ".pt")
            if os.path.exists(out_path):
                continue
            audio, _ = librosa.load(f, sr=16000)
            audio = torch.from_numpy(audio).float()
            mel = whisper.log_mel_spectrogram(audio)
            torch.save(mel, out_path)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(files)} 완료")
        except Exception as e:
            print(f"에러: {f} - {e}")

    print(f"완료: {dst_dir}")


if __name__ == "__main__":
    extract_and_save(
        "/content/drive/MyDrive/deepvoice/fake_raw/*.mp3",
        "/content/drive/MyDrive/deepvoice/whisper_features/fake_train"
    )
