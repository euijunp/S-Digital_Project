import json
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# 모델 및 토크나이저 로드
model_name = "gpt2"
model = TFGPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 기존 피싱 이메일 데이터셋 로드
with open("phishing_emails_dataset.json", "r") as file:
    phishing_emails = json.load(file)

refined_phishing_emails = []

# 각 이메일 본문을 다시 자연스럽게 생성
for email in phishing_emails:
    prompt = (
        "Refine the following email content to make it more natural and convincing: "
        f"{email['content']}"
    )
    inputs = tokenizer.encode(prompt, return_tensors='tf')
    attention_mask = [[1] * len(inputs[0])]
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask, 
        max_length=300, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95,
        do_sample=True  # 샘플링 모드 활성화
    )
    refined_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
    refined_phishing_email = {
        "sender": email["sender"],
        "content": refined_content,
        "phishing_url": email["phishing_url"]
    }
    refined_phishing_emails.append(refined_phishing_email)
    print(refined_phishing_email)

# 수정된 피싱 이메일을 JSON 파일에 저장
with open("refined_phishing_emails_dataset.json", "w") as file:
    json.dump(refined_phishing_emails, file, indent=4)
