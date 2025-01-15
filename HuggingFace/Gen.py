import json
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import random
import string

# 모델 및 토크나이저 로드
model_name = "gpt2"
model = TFGPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 임의의 피싱 웹사이트 주소 생성 함수
def generate_fake_url():
    random_string1 = ''.join(random.choices(string.ascii_lowercase, k=5))
    random_string2 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"http://{random_string1}{random_string2}.com"

# 임의의 발신자 이메일 주소 생성 함수
def generate_fake_email():
    random_string1 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    random_string2 = ''.join(random.choices(string.ascii_lowercase, k=5))
    return f"{random_string1}@{random_string2}.com"

# 피싱 이메일 시나리오 텍스트 프롬프트 정의
prompt = (
    "Write a detailed and convincing email that appears to be from a legitimate source. "
    "The email should try to trick the recipient into clicking the link. "
    "Include a fake website URL in the email content naturally. "
    "Make sure the email content is coherent and does not repeat itself. "
    "Avoid using words like 'phishing', 'scam', 'fake', 'fraud', and 'spam'."
)

# 피싱 이메일 데이터셋 생성
inputs = tokenizer.encode(prompt, return_tensors='tf')
attention_mask = [[1] * len(inputs[0])]
outputs = model.generate(
    inputs, 
    attention_mask=attention_mask, 
    max_length=300, 
    num_return_sequences=10,  # 한 번에 10개의 이메일 생성
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    do_sample=True  # 샘플링 모드 활성화
)

phishing_emails = []
for output in outputs:
    email_content = tokenizer.decode(output, skip_special_tokens=True)
    phishing_url = generate_fake_url()
    # 프롬프트 내용을 제거하고 실제 이메일 본문만 남김
    email_content = email_content.replace(prompt, "").strip()
    # 피싱 URL을 이메일 본문에 자연스럽게 포함
    email_content = email_content.replace("Include a fake website URL in the email content naturally.", f"Click here to verify your account: {phishing_url}")
    # 특정 키워드를 필터링하여 제거
    keywords_to_remove = ["phishing", "scam", "fake", "fraud", "spam"]
    for keyword in keywords_to_remove:
        email_content = email_content.replace(keyword, "")
    # 이메일 본문이 반복되지 않도록 필터링
    if len(set(email_content.split())) > 10 and "http" in email_content:  # 단어의 종류가 10개 이상이고 URL이 포함된 경우만 사용
        phishing_email = {
            "sender": generate_fake_email(),
            "content": email_content,
            "phishing_url": phishing_url
        }
        phishing_emails.append(phishing_email)
        print(phishing_email)

# 피싱 이메일을 JSON 파일에 저장
with open("phishing_emails_dataset.json", "w") as file:
    json.dump(phishing_emails, file, indent=4)