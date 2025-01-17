import json
import os
import tensorflow as tf
from transformers import AutoTokenizer, logging

# 디버깅 로그 숨기기
logging.set_verbosity_error()

# 저장된 TensorFlow 모델 경로 (디렉토리 경로)
base_dir = "/Users/birdbox/Library/CloudStorage/OneDrive-개인/SMU/S-Digital/Project/S-Digital_Project/Classification/"
original_model_path = os.path.join(base_dir, "saved_model_tf")
tokenizer_dir = os.path.join(original_model_path, "tokenizer")

# 모델 경로 확인
if not os.path.exists(original_model_path):
    raise FileNotFoundError(f"모델 디렉토리가 없습니다: {original_model_path}")

# TensorFlow SavedModel 로드
try:
    model = tf.saved_model.load(original_model_path)
    infer = model.signatures["serving_default"]
except Exception as e:
    raise RuntimeError(f"모델 로드 실패: {e}")

# Hugging Face 토크나이저 로드
if not os.path.exists(tokenizer_dir):
    raise FileNotFoundError(f"토크나이저 디렉토리가 없습니다: {tokenizer_dir}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# JSON 데이터 파일 경로
input_json_path = os.path.join(base_dir, "r_combined_emails_dataset.json")
output_json_path = os.path.join(base_dir, "classified_emails.json")

# JSON 파일 로드
if not os.path.exists(input_json_path):
    raise FileNotFoundError(f"{input_json_path} 파일이 존재하지 않습니다.")

with open(input_json_path, "r", encoding="utf-8") as file:
    emails = json.load(file)

# 이메일 본문 추출 및 분류
classified_emails = []
correct_predictions = 0

for email in emails:
    content = email.get("content", "")
    if not content:
        continue

    # 이메일 텍스트 토큰화
    inputs = tokenizer(
        content,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="tf"
    )

    # 예측 수행
    outputs = infer(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"]
    )

    # 예측 결과
    predicted_label = tf.argmax(outputs["logits"], axis=1).numpy()[0]

    # 예측된 레이블을 기반으로 이메일 분류
    email["predicted_label"] = "Phishing" if predicted_label == 1 else "Legitimate"

    # 채점: 실제 라벨과 예측된 라벨 비교
    if email["label"] == predicted_label:
        correct_predictions += 1

    classified_emails.append(email)

# 정확도 계산
accuracy = correct_predictions / len(emails) * 100

# 결과 JSON 파일로 저장
with open(output_json_path, "w", encoding="utf-8") as file:
    json.dump(classified_emails, file, indent=4, ensure_ascii=False)

print(f"Classification completed. Accuracy: {accuracy:.2f}%")
print(f"Results saved to {output_json_path}")