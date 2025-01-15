import os
import random
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import multiprocessing
import string

# 모델 로드
def load_model(model_name="gpt2"):
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# 랜덤 URL 생성 (피싱 특성 강화)
def generate_fake_url():
    return f"http://{''.join(random.choices(string.ascii_lowercase, k=5))}{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}.com"

# 랜덤 이메일 주소 생성 (다양한 피싱 시나리오 반영)
def generate_fake_email():
    domain = random.choice(['gmail', 'yahoo', 'hotmail', 'banking', 'support'])
    return f"{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}@{domain}.com"

# 프롬프트 스타일 및 랜덤 요소 정의
def generate_random_prompt():
    product_names = ["laptop", "smartphone", "headphones", "tablet", "watch"]
    random_product = random.choice(product_names)
    
    extensions = [
        "This offer expires soon, so don’t wait! Act now.",
        "Hurry, this offer won’t last forever. Claim it today.",
        "This is a limited-time offer, so take advantage of it while you can.",
        "You are one of the few lucky recipients of this exclusive offer."
    ]
    random_extension = random.choice(extensions)
    
    prompts = [
        f"You have a special offer waiting for you for a {random_product}. Please click the link below to claim your reward.",
        f"Congratulations! You’ve been selected to receive a limited-time offer for a {random_product}. Click here to claim it.",
        f"We are excited to inform you about a special discount on {random_product}. Don't miss out, click below to get started.",
        f"This is your opportunity to claim a fantastic reward on {random_product}. Just follow the link below to proceed.",
        f"Act now to take advantage of this exclusive offer on {random_product}. Click here to get started."
    ]
    
    # 랜덤으로 프롬프트와 확장된 텍스트 선택
    prompt = random.choice(prompts)
    return f"{prompt} {random_extension}"

# 캐시된 이메일 내용을 생성하는 함수
@lru_cache(maxsize=128)
def generate_email_content_cached(model_name, prompt, num_emails, max_length=300):
    model, tokenizer = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="tf", truncation=True, padding=True)
    generated = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_emails,
        no_repeat_ngram_size=3,  # n-gram 크기를 3으로 늘려서 반복을 줄임
        do_sample=True,  # 샘플링 활성화
        temperature=0.7,  # 샘플링에서 다양성 조정
        top_p=0.85,       # top-p 설정
        top_k=50,         # top-k 설정
    )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated]

# 이메일 생성 함수
def generate_email_batch(model_name, prompt, batch_size, max_length=300):
    return generate_email_content_cached(model_name, prompt, batch_size, max_length)

# 이메일 데이터를 JSON 파일로 저장하는 함수
def save_to_json(data, filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    existing_data.extend(data)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

# 배치 생성 함수
def generate_batch(args):
    model_name, prompt, batch_size = args
    return generate_email_batch(model_name, prompt, batch_size)

# 이메일 생성 및 저장 함수 (병렬 처리)
def process_email_batch_parallel(model_name, batch_size, total_batches, output_file):
    with ProcessPoolExecutor() as executor:
        # batch_size가 너무 크지 않도록 나누어 처리
        batch_sizes = [(model_name, generate_random_prompt(), batch_size) for _ in range(total_batches)]
        results = list(executor.map(generate_batch, batch_sizes))
    
    # 결과를 평탄화하여 한 번에 저장
    all_generated_emails = [item for sublist in results for item in sublist]
    
    # 추가적으로 phishing-related 단어를 필터링하여 저장
    filtered_emails = []
    for email in all_generated_emails:
        if not any(term in email.lower() for term in ['phishing', 'scam', 'fake', 'fraud', 'spam']):
            phishing_url = generate_fake_url()  # 랜덤 URL 생성
            sender_email = generate_fake_email()  # 랜덤 이메일 생성
            filtered_emails.append({
                "sender": sender_email,
                "content": email,
                "phishing_url": phishing_url
            })
    
    save_to_json(filtered_emails, output_file)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 이메일 생성 파라미터 및 출력 파일 설정
    batch_size = 10  # 한 번에 생성할 이메일 수
    total_batches = 10  # 생성할 배치 수
    output_file = "generated_emails.json"
    model_name = "gpt2"

    # 이메일 생성 및 저장
    process_email_batch_parallel(model_name, batch_size, total_batches, output_file)

    print(f"Generated emails saved to {output_file}")