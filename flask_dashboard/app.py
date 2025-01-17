from flask import Flask, render_template
import boto3
import json
from bs4 import BeautifulSoup

# Flask 애플리케이션 생성
app = Flask(__name__)

# AWS S3 정보
AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
AWS_SECRET_KEY = "AWS_SECRET_KEY"
AWS_REGION = "AWS_REGION"  # 예: "us-east-1"
S3_BUCKET_NAME = "S3_BUCKET_NAME"
S3_FILE_KEY = "S3_FILE_KEY"  # S3에 저장된 파일 이름

# S3 클라이언트 설정
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def clean_html(content):
    """HTML 태그 제거"""
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text()

@app.route("/")
def index():
    # S3에서 JSON 파일 가져오기
    try:
        s3_response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_KEY)
        emails = json.loads(s3_response['Body'].read().decode('utf-8'))
    except Exception as e:
        return f"Error retrieving data from S3: {e}"

    # HTML 태그 제거
    for email in emails:
        email["content"] = clean_html(email["content"])

    return render_template("index.html", emails=emails)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12345, debug=True)
