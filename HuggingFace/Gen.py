import json
import random
import string
import os
from concurrent.futures import ThreadPoolExecutor
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import time
import tensorflow as tf

# GPU 비활성화 설정 (CPU만 사용)
def configure_cpu_only():
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CUDA GPU 사용 안함
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # GPU growth 허용 안함
        tf.config.set_visible_devices([], 'GPU')  # GPU를 숨김
    except Exception as e:
        print(f"Error in configuring CPU only: {e}")

# 모델 로드
def load_model_and_tokenizer(model_name="distilgpt2"):
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# 랜덤 URL 생성
def generate_fake_url():
    return f"http://{''.join(random.choices(string.ascii_lowercase, k=5))}{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}.com"

# 랜덤 이메일 주소 생성
def generate_fake_email():
    return f"{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}@{''.join(random.choices(string.ascii_lowercase, k=5))}.com"

# 이메일 내용 생성
def generate_email_content(model, tokenizer, prompt, generation_count, max_length=300):
    try:
        inputs = tokenizer(prompt, return_tensors='tf', padding=True, truncation=True)
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=generation_count,
            temperature=0.7,
            top_k=10,
            top_p=0.85,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        return [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    except Exception as e:
        print(f"Error generating email content: {e}")
        return []

# JSON 파일 저장
def save_to_json(data, file_path):
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")

# 배치 처리 함수
def process_batch(model, tokenizer, prompt, batch_size):
    return generate_email_content(model, tokenizer, prompt, batch_size)

# 피싱 이메일 데이터셋 생성
def create_phishing_dataset(model, tokenizer, prompt, email_count, batch_size, file_path):
    phishing_emails = []
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        with tqdm(total=email_count, desc="Generating emails", unit="email") as pbar:
            while len(phishing_emails) < email_count:
                remaining = email_count - len(phishing_emails)
                batch_sizes = [min(batch_size, remaining)] * (remaining // batch_size + 1)
                future_batches = [
                    executor.submit(process_batch, model, tokenizer, prompt, batch)
                    for batch in batch_sizes
                ]
                for future in future_batches:
                    email_contents = future.result()
                    for content in email_contents:
                        phishing_url = generate_fake_url()
                        content = content.replace(prompt, "").strip()
                        content = content.replace(
                            "Include a fake website URL in the email content naturally.",
                            f"Click here to verify your account: {phishing_url}"
                        )
                        if len(set(content.split())) > 10 and "http" in content:
                            phishing_emails.append({
                                "sender": generate_fake_email(),
                                "content": content,
                                "phishing_url": phishing_url
                            })
                            pbar.update(1)  # 진행률 갱신
                            if len(phishing_emails) >= email_count:
                                break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Phishing dataset created with {len(phishing_emails)} emails in {elapsed_time:.2f} seconds.")
    save_to_json(phishing_emails, file_path)
    return phishing_emails

# 실행
if __name__ == "__main__":
    # CPU만 사용하도록 설정
    configure_cpu_only()

    model, tokenizer = load_model_and_tokenizer()
    prompt = (
        "Write a detailed and convincing email that appears to be from a legitimate source. "
        "The email should try to trick the recipient into clicking the link. "
        "Include a fake website URL in the email content naturally. "
        "Make sure the email content is coherent and does not repeat itself. "
        "Avoid using words like 'phishing', 'scam', 'fake', 'fraud', and 'spam'."
    )
    email_count = 10
    batch_size = 3

    base_path = os.path.dirname(os.path.abspath(__file__))
    phishing_file = os.path.join(base_path, "phishing_emails_dataset.json")

    phishing_emails = create_phishing_dataset(model, tokenizer, prompt, email_count, batch_size, phishing_file)