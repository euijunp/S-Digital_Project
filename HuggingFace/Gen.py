from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# 모델 및 토크나이저 로드
model_name = "gpt2"
model = TFGPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 피싱 이메일 시나리오 텍스트 프롬프트 정의
prompt = "Dear customer, your account has been compromised. Click the link below to secure it immediately."

# 텍스트 생성
inputs = tokenizer.encode(prompt, return_tensors='tf')
attention_mask = [[1] * len(inputs[0])]
outputs = model.generate(inputs, attention_mask=attention_mask, max_length=200, num_return_sequences=1)

# 생성된 텍스트 디코딩
phishing_email = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(phishing_email)

# 피싱 이메일을 파일에 저장
with open("phishing_email.txt", "w") as file:
    file.write(phishing_email)