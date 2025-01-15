import json
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# 모델 및 토크나이저 로드
model_name = "gpt2"
model = TFGPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 피싱 이메일 시나리오 텍스트 프롬프트 정의
prompt = (
    "Generate a variety of convincing phishing email contents. "
    "Each email should include a fake phishing website URL, a fake sender email address, and be detailed, creative, and common. "
    "The email should look like it is from a legitimate source and should try to trick the recipient into clicking the link."
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

phishing_emails = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
for email in phishing_emails:
    print(email)

# 피싱 이메일을 JSON 파일에 저장
with open("phishing_emails_dataset.json", "w") as file:
    json.dump(phishing_emails, file, indent=4)