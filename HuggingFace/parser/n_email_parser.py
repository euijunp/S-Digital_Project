import os
import re
import json
from email import message_from_string

# 파일 경로
file_path = 'n_dataset/0001*'
output_json_path = os.path.join(os.path.dirname(__file__), 'n_dataset', 'parsed_emails.json')

# 정규식: 본문 내 링크 추출
url_regex = re.compile(r'https?://[^\s]+')

# 결과 저장 리스트
parsed_emails = []

# 파일이 저장된 디렉토리에서 각 파일을 처리
directory = os.path.join(os.path.dirname(__file__), 'n_dataset')
for i in range(1, 11): # 마지막 파일 번호 +1
    prefix = f'{i:04d}'
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            file_full_path = os.path.join(directory, filename)
            if not os.path.isfile(file_full_path):
                continue

            with open(file_full_path, 'r', encoding='utf-8', errors='ignore') as file:
                raw_email = file.read()
                
                # 이메일 파싱
                email_message = message_from_string(raw_email)
                
                # 발신자
                sender = email_message.get('From', '')
                
                # 본문
                content = ''
                if email_message.is_multipart():
                    for part in email_message.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                else:
                    content = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
                
                # 링크 추출
                urls = url_regex.findall(content)
                
                # 결과 저장
                parsed_emails.append({
                    'sender': sender,
                    'content': content.strip(),
                    'urls': urls
                })
            break

# JSON 파일로 저장
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(parsed_emails, json_file, ensure_ascii=False, indent=4)

output_json_path