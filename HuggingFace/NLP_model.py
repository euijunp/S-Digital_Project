import nltk
import tensorflow as tf
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tqdm import tqdm
from nlpaug.augmenter.word import SynonymAug

# NLTK 리소스 다운로드
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

combined_dataset_path = "combined_emails_dataset.json"

# 데이터 로드 및 검증
if not os.path.exists(combined_dataset_path):
    raise FileNotFoundError(f"{combined_dataset_path} 파일이 존재하지 않습니다.")

with open(combined_dataset_path, "r", encoding="utf-8") as file:
    combined_emails = json.load(file)

# 텍스트와 레이블 추출
texts = [email["content"] for email in combined_emails]
labels = [email["label"] for email in combined_emails]

# 데이터 증강 (SynonymAug 사용)
aug = SynonymAug(aug_src="wordnet")
augmented_texts = [aug.augment(text) for text in texts]
augmented_labels = labels.copy()  # 증강된 텍스트에 대응하는 동일한 레이블 추가

# 증강된 텍스트와 레이블 합치기
texts.extend(augmented_texts)
labels.extend(augmented_labels)

# 입력 데이터 검증
def clean_texts_and_labels(text_list, label_list):
    cleaned_texts = []
    cleaned_labels = []
    for text, label in zip(text_list, label_list):
        if isinstance(text, str) and text.strip():
            cleaned_texts.append(text)
            cleaned_labels.append(label)
    return cleaned_texts, cleaned_labels

texts, labels = clean_texts_and_labels(texts, labels)

# 데이터셋 분리 (90% 학습, 10% 테스트)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# 모델 및 토크나이저 로드
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 텍스트 데이터 토큰화
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="tf"
)
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="tf"
)

# TensorFlow 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(len(train_texts)).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
)).batch(32)

# 모델 컴파일 (AdamW 사용)
optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 모델 학습
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # 학습 단계
    with tqdm(total=len(train_dataset), desc="Training", unit="batch") as pbar:
        for batch in train_dataset:
            model.train_on_batch(batch[0], batch[1])
            pbar.update(1)
    
    # 검증 단계
    with tqdm(total=len(test_dataset), desc="Validation", unit="batch") as pbar:
        for batch in test_dataset:
            model.test_on_batch(batch[0], batch[1])
            pbar.update(1)

# 모델 저장 및 압축
saved_model_dir = "./saved_model_tf"
tar_gz_path = "./saved_model.tar.gz"

os.makedirs(saved_model_dir, exist_ok=True)
model.save(saved_model_dir, save_format="tf")
tokenizer.save_pretrained(os.path.join(saved_model_dir, "tokenizer"))

import tarfile

def create_tar_gz(source_dir, output_file):
    try:
        with tarfile.open(output_file, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        print(f"Model saved and compressed to {output_file}")
    except Exception as e:
        print(f"Error while creating tar.gz: {e}")

create_tar_gz(saved_model_dir, tar_gz_path)

# 성능 평가
predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

# 성능 출력
print(classification_report(test_labels, predicted_labels, target_names=["Legitimate", "Phishing"]))