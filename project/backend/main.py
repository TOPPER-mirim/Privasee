# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io
from PIL import Image
import re
import numpy as np

# 이미지 처리/AI 라이브러리는 주석 아래에 실제 사용 예시 추가
# import cv2
# import mediapipe as mp
# import easyocr
# from transformers import pipeline

app = FastAPI(title="Privacy Risk Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- 간단한 정규식 패턴들 ---
PHONE_RE = re.compile(r"\b01[0-9][-\s]?\d{3,4}[-\s]?\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RRN_RE = re.compile(r"\b\d{6}[-\s]?\d{7}\b")  # 주민등록번호 패턴(형태만)
# 주소 패턴은 매우 복잡하므로 단순화 예시
ADDRESS_RE = re.compile(r"(시|도|군|구|읍|면|동)\s?.{2,40}")

class TextAnalyzeRequest(BaseModel):
    text: str

@app.post("/analyze/text")
async def analyze_text(payload: TextAnalyzeRequest):
    text = payload.text
    findings = []
    score = 0

    # 전화번호
    phones = PHONE_RE.findall(text)
    if phones:
        findings.append({"type": "phone", "matches": phones})
        score += 15

    # 이메일
    emails = EMAIL_RE.findall(text)
    if emails:
        findings.append({"type": "email", "matches": emails})
        score += 15

    # 주민등록번호
    rrns = RRN_RE.findall(text)
    if rrns:
        findings.append({"type": "rrn", "matches": rrns})
        score += 15

    # 주소 (단순히 키워드/패턴으로 탐지)
    addresses = ADDRESS_RE.findall(text)
    if addresses:
        findings.append({"type": "address", "matches": addresses})
        score += 10

    # 학력/학교/회사 키워드 예시
    school_keywords = ["고등학교", "중학교", "초등학교", "대학교", "학교", "학원"]
    for kw in school_keywords:
        if kw in text:
            findings.append({"type": "school_keyword", "matches": [kw]})
            score += 10
            break

    # 이름 감지는 간단한 고유명사/문맥 기반으로 추가 개선 필요
    # (여기에 KoBERT 기반 NER을 연결하면 인물명 추출 가능)
    result = {
        "module": "text",
        "score": min(50, score),
        "findings": findings
    }
    return result

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    w, h = image.size

    findings = []
    score = 0

    # -- 예시: 여기서 Mediapipe로 얼굴 감지, EasyOCR로 텍스트 인식 --
    # 실제 코드(외부 라이브러리)가 필요하므로 아래는 흐름 설명 형태
    #
    # 1) Mediapipe로 얼굴 박스 찾기 -> 얼굴이 보이면 score += 20
    # 2) EasyOCR로 이미지에서 텍스트 추출 -> "OO고등학교" 같은 텍스트가 있으면 score += 20
    # 3) 배경의 지리적 표지(예: 지도, 건물명) 감지는 키워드/문맥 검사 -> score += 10
    #
    # 예시(더미):
    findings.append({"type":"face", "confidence":0.9})
    score += 20
    findings.append({"type":"badge_text", "text":"예시고등학교"})
    score += 20

    return {"module":"image", "score": min(50, score), "findings": findings}

@app.post("/analyze/multi")
async def analyze_multi(text: str = Form(None), file: UploadFile = File(None)):
    # 통합 분석 예시
    total = 0
    detail = {}
    if text:
        t = await analyze_text(TextAnalyzeRequest(text=text))
        total += t["score"]
        detail["text"] = t
    if file:
        i = await analyze_image(file)
        total += i["score"]
        detail["image"] = i
    total = min(100, total)
    # 간단한 권장 조치 메시지
    advice = []
    if total >= 70:
        advice.append("매우 높은 위험: 게시를 재고하세요. 식별 가능한 정보가 많습니다.")
    elif total >= 40:
        advice.append("중간 위험: 일부 민감정보가 보입니다. 편집을 권장합니다.")
    else:
        advice.append("낮음: 위험 요소가 적지만 최종 확인 권장.")
    return {"total_score": total, "detail": detail, "advice": advice}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
