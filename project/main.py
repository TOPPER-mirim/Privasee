from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import re
import base64
import cv2
import numpy as np
import mediapipe as mp
import easyocr
from typing import List, Dict, Optional
import io
from PIL import Image
import logging
import openai
import os
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="개인정보 위험 자가 진단 서비스")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
pose_detection = mp_pose.Pose(min_detection_confidence=0.5)

# EasyOCR 초기화 (한국어, 영어)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# 요청 모델
class TextAnalysisRequest(BaseModel):
    text: str
    user_context: Optional[Dict] = None  # 사용자 컨텍스트 (나이, 직업, 활동 유형 등)

class AnalysisResponse(BaseModel):
    risk_score: int
    detected_items: List[Dict]
    combination_risks: List[Dict]
    recommendations: List[str]
    personalized_feedback: str
    risk_level: str

# 개인정보 패턴 정의
PATTERNS = {
    'phone': r'(\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4})|(\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})',
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'rrn': r'\d{6}[-\s]?[1-4]\d{6}',  # 주민등록번호
    'address': r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)[\s]?[\w\s]+[시군구][\s]?[\w\s]+[동읍면리]',
    'school': r'[\w]+(?:초등학교|중학교|고등학교|대학교|대학|학교)',
    'name': r'[가-힣]{2,4}(?:님|씨|학생|선생|교수)',
    'card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
    'account': r'\d{3,6}[-\s]?\d{2,6}[-\s]?\d{6,}',
    'workplace': r'[\w]+(?:회사|기업|병원|은행|대학|공사|그룹)',
    'birth_date': r'(\d{4})[년\.\-/](\d{1,2})[월\.\-/](\d{1,2})[일]?',
    'age': r'(\d{1,2})[세살]|나이\s*(\d{1,2})',
}

# 위험도 가중치
RISK_WEIGHTS = {
    'phone': 25,
    'email': 15,
    'rrn': 40,
    'address': 20,
    'school': 10,
    'name': 10,
    'card': 35,
    'account': 30,
    'face': 15,
    'body': 10,
    'text_in_image': 5,
    'workplace': 15,
    'birth_date': 25,
    'age': 10,
}

# 조합 위험 패턴 정의
COMBINATION_RISKS = [
    {
        'name': '신원 특정 위험',
        'pattern': ['name', 'school', 'workplace'],
        'min_count': 2,
        'risk_multiplier': 1.5,
        'description': '이름과 학교/직장 정보로 개인 신원이 특정될 수 있습니다'
    },
    {
        'name': '연락처 추적 위험',
        'pattern': ['name', 'phone', 'address'],
        'min_count': 2,
        'risk_multiplier': 2.0,
        'description': '이름, 연락처, 주소 조합으로 개인 추적이 가능합니다'
    },
    {
        'name': '금융 사기 위험',
        'pattern': ['name', 'birth_date', 'phone', 'card', 'account'],
        'min_count': 3,
        'risk_multiplier': 2.5,
        'description': '개인정보와 금융정보 조합으로 금융 사기에 악용될 수 있습니다'
    },
    {
        'name': '개인정보 도용 위험',
        'pattern': ['name', 'rrn', 'phone'],
        'min_count': 2,
        'risk_multiplier': 3.0,
        'description': '주민등록번호와 개인정보 조합으로 신분 도용이 가능합니다'
    },
    {
        'name': '스토킹/괴롭힘 위험',
        'pattern': ['name', 'address', 'school', 'workplace'],
        'min_count': 2,
        'risk_multiplier': 1.8,
        'description': '개인 활동 장소 조합으로 스토킹이나 괴롭힘에 노출될 수 있습니다'
    }
]

def analyze_text(text: str) -> Dict:
    """텍스트 분석 함수"""
    detected_items = []
    total_risk = 0
    
    for pattern_name, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            count = len(matches)
            risk = RISK_WEIGHTS.get(pattern_name, 10) * min(count, 3)
            total_risk += risk
            
            detected_items.append({
                'type': pattern_name,
                'count': count,
                'risk_contribution': risk,
                'examples': matches[:2] if len(matches) > 0 else []
            })
    
    return {
        'detected_items': detected_items,
        'total_risk': min(total_risk, 100)
    }

def analyze_combination_risks(detected_items: List[Dict]) -> List[Dict]:
    """조합 위험 분석"""
    combination_risks = []
    detected_types = set([item['type'] for item in detected_items])
    
    for combo_risk in COMBINATION_RISKS:
        pattern_match_count = sum(1 for pattern_type in combo_risk['pattern'] 
                                if pattern_type in detected_types)
        
        if pattern_match_count >= combo_risk['min_count']:
            matched_types = [t for t in combo_risk['pattern'] if t in detected_types]
            
            combination_risks.append({
                'name': combo_risk['name'],
                'matched_types': matched_types,
                'risk_multiplier': combo_risk['risk_multiplier'],
                'description': combo_risk['description'],
                'severity': 'high' if combo_risk['risk_multiplier'] >= 2.0 else 'medium'
            })
    
    return combination_risks

def analyze_image(image_bytes: bytes) -> Dict:
    """이미지 분석 함수"""
    detected_items = []
    total_risk = 0
    
    try:
        # 이미지 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. 얼굴 탐지
        face_results = face_detection.process(rgb_image)
        if face_results.detections:
            face_count = len(face_results.detections)
            risk = RISK_WEIGHTS['face'] * min(face_count, 3)
            total_risk += risk
            detected_items.append({
                'type': 'face',
                'count': face_count,
                'risk_contribution': risk,
                'description': f'{face_count}개의 얼굴이 감지되었습니다'
            })
        
        # 2. 신체 탐지
        pose_results = pose_detection.process(rgb_image)
        if pose_results.pose_landmarks:
            total_risk += RISK_WEIGHTS['body']
            detected_items.append({
                'type': 'body',
                'count': 1,
                'risk_contribution': RISK_WEIGHTS['body'],
                'description': '신체 부위가 감지되었습니다'
            })
        
        # 3. OCR을 통한 텍스트 추출
        ocr_results = reader.readtext(image_bytes)
        extracted_text = ' '.join([text[1] for text in ocr_results])
        
        if extracted_text:
            # 추출된 텍스트에서 개인정보 패턴 검색
            text_analysis = analyze_text(extracted_text)
            if text_analysis['detected_items']:
                for item in text_analysis['detected_items']:
                    item['source'] = 'image_text'
                    detected_items.extend(text_analysis['detected_items'])
                    total_risk += item['risk_contribution']
            
            # 이미지에서 텍스트가 발견된 것 자체도 위험 요소
            total_risk += RISK_WEIGHTS['text_in_image']
            detected_items.append({
                'type': 'text_in_image',
                'count': len(ocr_results),
                'risk_contribution': RISK_WEIGHTS['text_in_image'],
                'description': f'이미지에서 {len(ocr_results)}개의 텍스트 영역이 감지되었습니다'
            })
    
    except Exception as e:
        logger.error(f"이미지 분석 중 오류 발생: {str(e)}")
        return {'detected_items': [], 'total_risk': 0}
    
    return {
        'detected_items': detected_items,
        'total_risk': min(total_risk, 100)
    }

async def generate_personalized_feedback(detected_items: List[Dict], 
                                       combination_risks: List[Dict],
                                       user_context: Optional[Dict] = None) -> str:
    """AI 기반 개인 맞춤형 피드백 생성"""
    try:
        # 사용자 컨텍스트 기본값 설정
        if not user_context:
            user_context = {'age_group': 'general', 'activity_type': 'general'}
        
        # 탐지된 항목 요약
        detected_summary = ", ".join([f"{item['type']} ({item['count']}개)" 
                                    for item in detected_items])
        
        # 조합 위험 요약
        combo_summary = ", ".join([risk['name'] for risk in combination_risks])
        
        # GPT 프롬프트 구성
        prompt = f"""
개인정보 보호 전문가로서 다음 분석 결과를 바탕으로 개인 맞춤형 피드백을 작성해주세요.

분석 결과:
- 탐지된 개인정보: {detected_summary if detected_summary else '없음'}
- 조합 위험: {combo_summary if combo_summary else '없음'}
- 사용자 정보: {user_context}

피드백 요구사항:
1. 사용자의 상황에 맞는 구체적이고 실용적인 조언
2. 왜 이런 위험이 발생하는지 쉽게 설명
3. 개선 방법을 단계별로 제시
4. 친근하고 이해하기 쉬운 톤
5. 200자 내외로 간결하게

피드백:
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 개인정보 보호 전문가입니다. 사용자에게 친근하고 실용적인 조언을 제공합니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"AI 피드백 생성 오류: {str(e)}")
        # 기본 피드백 반환
        if combination_risks:
            return "여러 개인정보가 조합되어 위험도가 높습니다. 필요한 정보만 선별적으로 공개하고, 민감한 정보는 삭제하는 것을 권장합니다."
        elif detected_items:
            return "일부 개인정보가 노출되어 있습니다. 개인 식별이 가능한 정보는 가리거나 삭제하여 안전하게 공유하세요."
        else:
            return "개인정보 노출 위험이 낮습니다. 하지만 항상 주의하여 정보를 공유하세요."

def get_risk_level(score: int) -> str:
    """위험도 레벨 판정"""
    if score >= 70:
        return "매우 위험"
    elif score >= 50:
        return "위험"
    elif score >= 30:
        return "주의"
    elif score >= 10:
        return "양호"
    else:
        return "안전"

def generate_recommendations(detected_items: List[Dict], combination_risks: List[Dict]) -> List[str]:
    """개선 권고사항 생성 (조합 위험 포함)"""
    recommendations = []
    
    type_messages = {
        'phone': '전화번호가 노출되어 있습니다. 부분적으로 가리거나 삭제를 권장합니다.',
        'email': '이메일 주소가 노출되어 있습니다. 스팸 메일의 위험이 있으니 주의하세요.',
        'rrn': '주민등록번호는 절대 공개하지 마세요. 즉시 삭제를 권장합니다.',
        'address': '상세 주소가 노출되면 위치가 특정될 수 있습니다. 동 단위까지만 공개하세요.',
        'school': '학교명이 노출되어 있습니다. 신원 파악의 단서가 될 수 있습니다.',
        'name': '실명이 노출되어 있습니다. 닉네임 사용을 권장합니다.',
        'card': '카드번호가 노출되어 있습니다. 금융 사기의 위험이 있으니 즉시 삭제하세요.',
        'account': '계좌번호가 노출되어 있습니다. 금융 정보는 절대 공개하지 마세요.',
        'face': '얼굴이 노출되어 있습니다. 모자이크 처리나 스티커로 가리는 것을 권장합니다.',
        'body': '신체가 노출되어 있습니다. 개인 식별이 가능할 수 있으니 주의하세요.',
        'text_in_image': '이미지에 텍스트가 포함되어 있습니다. 민감한 정보가 없는지 확인하세요.',
        'workplace': '직장 정보가 노출되어 있습니다. 개인 신원 파악에 활용될 수 있습니다.',
        'birth_date': '생년월일이 노출되어 있습니다. 신원 도용에 악용될 수 있습니다.',
        'age': '나이 정보가 노출되어 있습니다. 다른 정보와 조합하여 신원 추정이 가능합니다.'
    }
    
    # 기본 권고사항
    for item in detected_items:
        if item['type'] in type_messages:
            recommendations.append(type_messages[item['type']])
    
    # 조합 위험 권고사항
    for combo_risk in combination_risks:
        if combo_risk['severity'] == 'high':
            recommendations.append(f"⚠️ {combo_risk['description']} - 일부 정보를 삭제하거나 가려주세요.")
        else:
            recommendations.append(f"⚡ {combo_risk['description']} - 주의가 필요합니다.")
    
    # 일반 권고사항 추가
    if len(detected_items) > 3:
        recommendations.append('여러 개인정보가 동시에 노출되어 있습니다. 전반적인 검토가 필요합니다.')
    
    if not recommendations:
        recommendations.append('개인정보 노출 위험이 낮습니다. 하지만 항상 주의하세요.')
    
    return recommendations

@app.get("/")
async def root():
    return {"message": "개인정보 위험 자가 진단 서비스 API (AI 피드백 지원)"}

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text_endpoint(request: TextAnalysisRequest):
    """텍스트 분석 엔드포인트 (AI 피드백 포함)"""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다")
        
        analysis = analyze_text(request.text)
        combination_risks = analyze_combination_risks(analysis['detected_items'])
        
        # 조합 위험으로 인한 추가 점수
        combo_bonus = sum(risk['risk_multiplier'] * 10 for risk in combination_risks)
        final_risk = min(analysis['total_risk'] + combo_bonus, 100)
        
        recommendations = generate_recommendations(analysis['detected_items'], combination_risks)
        risk_level = get_risk_level(final_risk)
        
        # AI 기반 개인 맞춤형 피드백 생성
        personalized_feedback = await generate_personalized_feedback(
            analysis['detected_items'], 
            combination_risks, 
            request.user_context
        )
        
        return AnalysisResponse(
            risk_score=int(final_risk),
            detected_items=analysis['detected_items'],
            combination_risks=combination_risks,
            recommendations=recommendations,
            personalized_feedback=personalized_feedback,
            risk_level=risk_level
        )
    
    except Exception as e:
        logger.error(f"텍스트 분석 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image_endpoint(file: UploadFile = File(...), user_context: str = None):
    """이미지 분석 엔드포인트 (AI 피드백 포함)"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
        
        contents = await file.read()
        analysis = analyze_image(contents)
        combination_risks = analyze_combination_risks(analysis['detected_items'])
        
        # 조합 위험으로 인한 추가 점수
        combo_bonus = sum(risk['risk_multiplier'] * 10 for risk in combination_risks)
        final_risk = min(analysis['total_risk'] + combo_bonus, 100)
        
        recommendations = generate_recommendations(analysis['detected_items'], combination_risks)
        risk_level = get_risk_level(final_risk)
        
        # 사용자 컨텍스트 파싱
        context_dict = {}
        if user_context:
            try:
                import json
                context_dict = json.loads(user_context)
            except:
                context_dict = {'activity_type': user_context}
        
        # AI 기반 개인 맞춤형 피드백 생성
        personalized_feedback = await generate_personalized_feedback(
            analysis['detected_items'], 
            combination_risks, 
            context_dict
        )
        
        return AnalysisResponse(
            risk_score=int(final_risk),
            detected_items=analysis['detected_items'],
            combination_risks=combination_risks,
            recommendations=recommendations,
            personalized_feedback=personalized_feedback,
            risk_level=risk_level
        )
    
    except Exception as e:
        logger.error(f"이미지 분석 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/combined")
async def analyze_combined_endpoint(
    text: Optional[str] = None,
    file: Optional[UploadFile] = File(None),
    user_context: Optional[str] = None
):
    """텍스트와 이미지 통합 분석 엔드포인트 (AI 피드백 포함)"""
    try:
        total_risk = 0
        all_detected_items = []
        
        # 텍스트 분석
        if text:
            text_analysis = analyze_text(text)
            total_risk += text_analysis['total_risk']
            all_detected_items.extend(text_analysis['detected_items'])
        
        # 이미지 분석
        if file:
            contents = await file.read()
            image_analysis = analyze_image(contents)
            total_risk += image_analysis['total_risk']
            all_detected_items.extend(image_analysis['detected_items'])
        
        # 조합 위험 분석
        combination_risks = analyze_combination_risks(all_detected_items)
        combo_bonus = sum(risk['risk_multiplier'] * 10 for risk in combination_risks)
        final_risk = min(total_risk + combo_bonus, 100)
        
        recommendations = generate_recommendations(all_detected_items, combination_risks)
        risk_level = get_risk_level(final_risk)
        
        # 사용자 컨텍스트 파싱
        context_dict = {}
        if user_context:
            try:
                import json
                context_dict = json.loads(user_context)
            except:
                context_dict = {'activity_type': user_context}
        
        # AI 기반 개인 맞춤형 피드백 생성
        personalized_feedback = await generate_personalized_feedback(
            all_detected_items, 
            combination_risks, 
            context_dict
        )
        
        return AnalysisResponse(
            risk_score=int(final_risk),
            detected_items=all_detected_items,
            combination_risks=combination_risks,
            recommendations=recommendations,
            personalized_feedback=personalized_feedback,
            risk_level=risk_level
        )
    
    except Exception as e:
        logger.error(f"통합 분석 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)