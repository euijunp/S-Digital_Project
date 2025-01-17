import random
import json

# JSON 파일 경로
base_dir = "/Users/birdbox/Library/CloudStorage/OneDrive-개인/SMU/S-Digital/Project/S-Digital_Project/Classification/"
input_json_path = base_dir + "combined_emails_dataset.json"
output_json_path = base_dir + "r_combined_emails_dataset.json"

# JSON 파일 로드
with open(input_json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 데이터 순서를 랜덤으로 섞기
random.shuffle(data)

# 결과를 새로운 JSON 파일로 저장
with open(output_json_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print(f"Shuffled data saved to {output_json_path}")