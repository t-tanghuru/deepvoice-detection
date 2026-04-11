from gtts import gTTS
import os

# 한국어 문장 샘플
sentences = [
    "안녕하세요, 저는 은행 직원입니다.",
    "고객님의 계좌에 문제가 발생했습니다.",
    "지금 바로 본인 확인이 필요합니다.",
    "개인정보를 알려주시면 도와드리겠습니다.",
    "오늘 날씨가 정말 좋네요.",
    "내일 회의가 있으니 준비해주세요.",
    "가족들과 즐거운 시간 보내세요.",
    "점심으로 뭘 먹을까요?",
    "저는 경찰청에서 연락드립니다.",
    "긴급 상황이 발생했습니다.",
]

output_dir = "data/fake"
os.makedirs(output_dir, exist_ok=True)

for i, sentence in enumerate(sentences):
    tts = gTTS(sentence, lang='ko')
    filename = f"{output_dir}/fake_{i:04d}.mp3"
    tts.save(filename)
    print(f"생성 완료: {filename}")

print("전체 완료!")
