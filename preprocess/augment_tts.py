from gtts import gTTS
from pydub import AudioSegment
import os
import random

# 한국어 문장들
sentences = [
    "안녕하세요, 저는 은행 직원입니다.",
    "고객님의 계좌에 문제가 발생했습니다.",
    "지금 바로 본인 확인이 필요합니다.",
    "개인정보를 알려주시면 도와드리겠습니다.",
    "저는 경찰청에서 연락드립니다.",
    "긴급 상황이 발생했습니다.",
    "계좌 이체를 즉시 진행해주세요.",
    "가족분이 사고가 났습니다.",
    "검찰청 수사관입니다.",
    "지금 바로 송금해주셔야 합니다.",
]

output_dir = "data/fake"
os.makedirs(output_dir, exist_ok=True)

count = 0
for i, sentence in enumerate(sentences):
    # 원본 gTTS 생성
    tts = gTTS(sentence, lang='ko')
    base_path = f"{output_dir}/fake_{count:04d}.mp3"
    tts.save(base_path)
    print(f"생성: {base_path}")
    count += 1

    # pydub으로 증강
    audio = AudioSegment.from_mp3(base_path)

    # 1. 피치 높이기 (빠르게)
    fast = audio.speedup(playback_speed=1.2)
    fast.export(f"{output_dir}/fake_{count:04d}.mp3", format="mp3")
    count += 1

    # 2. 피치 낮추기 (느리게)
    slow = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * 0.85)})
    slow.export(f"{output_dir}/fake_{count:04d}.mp3", format="mp3")
    count += 1

    # 3. 볼륨 조절
    quiet = audio - 5
    quiet.export(f"{output_dir}/fake_{count:04d}.mp3", format="mp3")
    count += 1

print(f"총 {count}개 생성 완료")
