# S-Digital_Project
## AI 기반 피싱 이메일 탐지 및 데이터 생성 시스템
**AI-based Phishing Email Detection and Analysis System**

---

## 개요

본 프로젝트는 AI를 활용하여 피싱 이메일 탐지 및 데이터를 자동으로 생성하는 시스템을 구축하는 것을 목표로 합니다.  
이 시스템은 피싱 시나리오를 반영한 데이터 생성과 탐지 모델 학습을 통해 보안 위협을 효과적으로 완화할 수 있습니다.

---

## 주요 기능

1. **피싱 이메일 탐지**  
   - 이메일 본문, URL, 헤더 등을 분석하여 피싱 여부를 판별.

2. **피싱 이메일 데이터 생성**  
   - 랜덤 URL, 이메일 주소, 프롬프트를 기반으로 AI를 활용해 자동화된 피싱 데이터를 생성.

3. **데이터 저장 및 분석**  
   - 생성된 데이터를 JSON 형식으로 저장하고 Amazon S3를 통해 관리.

4. **관리자 대시보드**  
   - Flask 기반 대시보드에서 탐지 결과 및 데이터를 시각화.

---

## 기술 스택

- **Backend**: Python, Flask, Amazon Lambda, Amazon S3
- **AI 모델**: GPT-2 (Hugging Face Transformers)
- **병렬 처리**: `multiprocessing`, `concurrent.futures`
- **데이터 저장**: JSON 파일, Amazon S3
- **프론트엔드**: Flask 대시보드

---
