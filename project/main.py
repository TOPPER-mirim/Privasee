from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import re
import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter
import os
import base64
import json
import google.generativeai as genai
import google.generativeai as genai
from PIL import Image


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API 설정
GEMINI_API_KEY = os.getenv("AIzaSyBHDNQa_5rVWZwLJzGafR9EUtp4ZX1oKBA", "AIzaSyBHDNQa_5rVWZwLJzGafR9EUtp4ZX1oKBA")
genai.configure(api_key=GEMINI_API_KEY)

# FastAPI 앱 초기화
app = FastAPI(title="Gemini 기반 개인정보 위험 자가 진단 서비스")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class TextAnalysisRequest(BaseModel):
    text: str
    user_context: Optional[Dict] = None

class AnalysisResponse(BaseModel):
    risk_score: int
    detected_items: List[Dict]
    combination_risks: List[Dict]
    recommendations: List[str]
    personalized_feedback: str
    risk_level: str
    detailed_analysis: Optional[Dict] = None

# 개인정보 패턴 정의 (정규식)
PATTERNS = {
    'phone': r'(?:\b0(?:1[016789]|2|[3-6]\d|70)[-.\s]?\d{3,4}[-.\s]?\d{4}\b)',
    'phone_international': r'(?:\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4})',
    'email': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    'rrn': r'\b(?:19|20)?\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[-\s]?[1-4]\d{6}\b',
    'passport': r'\b[A-Z]\d{7,8}\b',
    'passport_mrz': r'\b[A-Z0-9<]{20,}\b',
    'driver_license': r'\b[0-9]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{6}[-\s]?[0-9]{2}\b',
    'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    'account': r'\b\d{2,6}[-\s]?\d{2,6}[-\s]?\d{4,}\b',
    'address': r'\b(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)(?:특별시|광역시|도)?\s*[^\n,;]{1,60}(?:시|군|구|읍|면|동|리)\b',
    'detailed_address': r'\b\d{1,5}(?:[-]\d{1,4})?\s*(?:번지|호|층)?\b',
    'school': r'\b[A-Za-z가-힣0-9\s\-]{2,40}(?:초등학교|중학교|고등학교|대학교|대학)\b',
    'workplace': r'\b[A-Za-z가-힣0-9\s\-]{2,40}(?:회사|기업|병원|은행)\b',
    'name': r'(?<![A-Za-z0-9가-힣])[가-힣]{2,4}(?:님|씨)?(?![A-Za-z0-9가-힣])',
    'birth_date': r'\b(?:19|20)\d{2}[년\.\-/]\s?\d{1,2}[월\.\-/]\s?\d{1,2}일?\b',
    'car_number': r'\b\d{2,3}[가-힣]\s*\d{4}\b',
    'ip_address': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
}

# 위험도 가중치 (100점 만점)
RISK_WEIGHTS = {
    # 정규식으로 탐지 가능한 항목
    'phone': 25,
    'email': 15,
    'rrn': 50,  # 주민등록번호 최고 위험
    'address': 20,
    'detailed_address': 30,
    'school': 12,
    'name': 10,
    'credit_card': 45,
    'account': 35,
    'birth_date': 25,
    'car_number': 20,
    'passport': 45,
    'driver_license': 40,
    'workplace': 15,
    'ip_address': 18,
    
    # Gemini로 탐지하는 항목
    'face_clear': 30,  # 선명한 얼굴
    'face': 20,  # 일반 얼굴
    'student_id': 40,  # 학생증
    'pharmacy_bag': 38,  # 약봉투 (질병정보)
    'delivery_label': 35,  # 운송장 (주소+전화번호)
    'wedding_invitation': 25,  # 청첩장 (이름+주소+전화번호)
    'id_card': 50,  # 신분증
    'body_identifiable': 15,  # 식별 가능한 신체
    'background_info': 12,  # 배경의 개인정보
    'handwriting': 10,  # 필적
    'fingerprint': 35,  # 지문
    'medical_info': 40,  # 의료정보
    'location_landmark': 22,  # 위치 특정 가능한 랜드마크
}

# 조합 위험 패턴
COMBINATION_RISKS = [
    {
        'name': '신원 특정 위험',
        'pattern': ['name', 'school', 'workplace', 'address'],
        'min_count': 2,
        'risk_multiplier': 1.5,
        'description': '이름과 소속 정보로 개인 신원이 특정될 수 있습니다'
    },
    {
        'name': '연락처 추적 위험',
        'pattern': ['name', 'phone', 'address', 'delivery_label'],
        'min_count': 2,
        'risk_multiplier': 2.0,
        'description': '이름, 연락처, 주소 조합으로 개인 추적이 가능합니다'
    },
    {
        'name': '금융 사기 위험',
        'pattern': ['name', 'birth_date', 'phone', 'credit_card', 'account'],
        'min_count': 3,
        'risk_multiplier': 2.5,
        'description': '개인정보와 금융정보 조합으로 금융 사기에 악용될 수 있습니다'
    },
    {
        'name': '신분 도용 위험',
        'pattern': ['name', 'rrn', 'phone', 'birth_date', 'id_card', 'student_id'],
        'min_count': 2,
        'risk_multiplier': 3.0,
        'description': '신분증과 개인정보 조합으로 신분 도용이 가능합니다'
    },
    {
        'name': '의료정보 유출 위험',
        'pattern': ['name', 'pharmacy_bag', 'medical_info', 'phone', 'address'],
        'min_count': 2,
        'risk_multiplier': 2.2,
        'description': '질병정보와 개인정보가 결합되어 민감한 의료정보가 유출될 수 있습니다'
    },
    {
        'name': '위치 추적 위험',
        'pattern': ['face_clear', 'address', 'location_landmark', 'car_number'],
        'min_count': 2,
        'risk_multiplier': 1.8,
        'description': '얼굴과 위치 정보로 실시간 추적이 가능합니다'
    },
    {
        'name': '생체정보 유출 위험',
        'pattern': ['face_clear', 'fingerprint', 'name'],
        'min_count': 2,
        'risk_multiplier': 2.3,
        'description': '생체정보가 노출되어 생체인증 시스템 악용 가능성이 있습니다'
    },
]

# Gemini 프롬프트
GEMINI_ANALYSIS_PROMPT = """
당신은 개인정보 보호 전문가입니다. 이미지를 분석하여 개인정보 노출 위험을 평가해주세요.

다음 항목들을 찾아서 JSON 형식으로 반환해주세요:

1. **얼굴 (face)**
   - face_clear: 선명하게 식별 가능한 얼굴 (개수)
   - face: 흐릿하거나 부분적인 얼굴 (개수)

2. **신분증/문서 (documents)**
   - id_card: 주민등록증, 운전면허증, 여권 등 정부 발급 신분증
   - student_id: 학생증 (학교명, 이름, 사진 포함)
   - pharmacy_bag: 약봉투 (약국명, 환자명, 처방내역)
   - delivery_label: 운송장/택배 라벨 (이름, 주소, 전화번호)
   - wedding_invitation: 청첩장 (신랑신부 이름, 연락처, 장소)
   - medical_document: 의료 관련 문서 (진단서, 처방전 등)

3. **텍스트 정보 (text_in_image)**
   - 이미지에서 추출 가능한 모든 텍스트
   - 특히 이름, 전화번호, 주소, 이메일 등

4. **생체정보 (biometric)**
   - fingerprint: 선명한 지문
   - handwriting: 필적 (서명 포함)

5. **위치/배경 정보 (location)**
   - location_landmark: 특정 장소를 식별할 수 있는 랜드마크, 간판, 건물명
   - background_info: 배경에 노출된 개인정보 (포스터, 명함, 서류 등)

6. **기타**
   - body_identifiable: 문신, 흉터 등 식별 가능한 신체 특징
   - car_number: 차량 번호판 (텍스트로도 확인)

**응답 형식 (JSON):**
```json
{
  "detected_items": [
    {
      "type": "face_clear",
      "count": 2,
      "confidence": 0.95,
      "description": "선명한 얼굴 2개 감지"
    },
    {
      "type": "student_id",
      "count": 1,
      "confidence": 0.88,
      "description": "학생증 감지 - 대학교명, 이름, 사진 포함",
      "details": "XX대학교 학생증"
    }
  ],
  "extracted_text": "이미지에서 추출된 모든 텍스트",
  "risk_assessment": "이미지의 전반적인 개인정보 노출 위험 평가",
  "sensitive_areas": ["얼굴 영역", "신분증 영역"]
}
```

**중요 사항:**
- 각 항목은 명확하게 확인되는 경우에만 포함
- confidence는 0~1 사이의 값
- 의심스럽거나 불확실한 경우 confidence를 낮게 설정
- 한국어로 된 문서나 텍스트도 정확히 인식
"""

def analyze_text_with_regex(text: str) -> Dict:
    """정규식을 사용한 텍스트 분석"""
    detected_items = []
    total_risk = 0
    
    logger.info(f"텍스트 정규식 분석 시작: {len(text)} 글자")
    
    for pattern_name, pattern in PATTERNS.items():
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                count = len(matches)
                risk = RISK_WEIGHTS.get(pattern_name, 10) * min(count, 3)
                total_risk += risk
                
                # 마스킹 처리
                masked_examples = []
                for match in matches[:2]:
                    if isinstance(match, tuple):
                        match = ''.join(match)
                    
                    # 민감정보 강력 마스킹
                    if pattern_name in ['rrn', 'credit_card', 'account', 'passport', 'driver_license']:
                        if len(str(match)) > 6:
                            masked = str(match)[:2] + '*' * (len(str(match)) - 4) + str(match)[-2:]
                        else:
                            masked = '*' * len(str(match))
                    else:
                        masked = str(match)[:2] + '*' * max(0, len(str(match)) - 2)
                    
                    masked_examples.append(masked)
                
                detected_items.append({
                    'type': pattern_name,
                    'count': count,
                    'risk_contribution': risk,
                    'examples': masked_examples,
                    'source': 'text'
                })
                
                logger.info(f"패턴 발견: {pattern_name} - {count}개")
        
        except Exception as e:
            logger.error(f"패턴 {pattern_name} 매칭 오류: {str(e)}")
    
    return {
        'detected_items': detected_items,
        'total_risk': min(total_risk, 100)
    }

async def analyze_image_with_gemini(image_bytes: bytes) -> Dict:
    """Gemini API를 사용한 이미지 분석"""
    try:
        logger.info("Gemini API 이미지 분석 시작")
        
        # API Key 확인
        if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
            logger.warning("⚠️ Gemini API Key가 설정되지 않음 - 기본 분석만 수행")
            # API 없이 텍스트만 추출 (fallback)
            return {
                'detected_items': [{
                    'type': 'face',
                    'count': 1,
                    'risk_contribution': 20,
                    'description': '이미지가 업로드되었습니다 (Gemini API 미설정)',
                    'source': 'image'
                }],
                'total_risk': 20,
                'detailed_analysis': {'error': 'Gemini API Key not configured'}
            }
        
        # Gemini 모델 초기화
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # 이미지를 PIL로 변환
        from PIL import Image as PILImage
        import io
        image = PILImage.open(io.BytesIO(image_bytes))
        
        # 프롬프트와 함께 분석 요청
        response = model.generate_content([
            GEMINI_ANALYSIS_PROMPT,
            image
        ])
        
        logger.info(f"Gemini API 응답 받음: {len(response.text)} 글자")
        logger.info(f"응답 샘플: {response.text[:300]}...")
        
        # JSON 파싱
        response_text = response.text
        
        # JSON 블록 추출 (```json ... ``` 형식)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 블록이 없으면 전체 텍스트에서 JSON 찾기
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("JSON 형식을 찾을 수 없음, 원본 응답 확인")
                logger.warning(response_text)
                # 기본값 반환
                return {
                    'detected_items': [{
                        'type': 'face',
                        'count': 1,
                        'risk_contribution': 20,
                        'description': '이미지 분석 완료 (JSON 파싱 실패)',
                        'source': 'image'
                    }],
                    'total_risk': 20,
                    'detailed_analysis': {'raw_response': response_text[:500]}
                }
        
        gemini_result = json.loads(json_str)
        logger.info(f"JSON 파싱 성공: {len(gemini_result.get('detected_items', []))}개 항목")
        
        # 결과 변환
        detected_items = []
        total_risk = 0
        
        for item in gemini_result.get('detected_items', []):
            item_type = item.get('type')
            count = item.get('count', 1)
            confidence = item.get('confidence', 0.8)
            
            logger.info(f"처리 중: {item_type}, count={count}, confidence={confidence}")
            
            # 신뢰도가 0.5 이상인 경우만 포함 (임계값 낮춤)
            if confidence >= 0.5 and item_type in RISK_WEIGHTS:
                risk = RISK_WEIGHTS[item_type] * min(count, 3) * confidence
                total_risk += risk
                
                detected_items.append({
                    'type': item_type,
                    'count': count,
                    'risk_contribution': risk,
                    'confidence': confidence,
                    'description': item.get('description', ''),
                    'details': item.get('details', ''),
                    'source': 'image'
                })
                
                logger.info(f"✅ Gemini 탐지: {item_type} - {count}개 (위험도: {risk:.1f}점)")
            else:
                if item_type not in RISK_WEIGHTS:
                    logger.warning(f"⚠️ 알 수 없는 타입: {item_type}")
                else:
                    logger.info(f"❌ 신뢰도 낮음: {item_type} (confidence={confidence})")
        
        # 추출된 텍스트도 정규식으로 분석
        extracted_text = gemini_result.get('extracted_text', '')
        if extracted_text:
            logger.info(f"추출된 텍스트: {len(extracted_text)} 글자")
            text_analysis = analyze_text_with_regex(extracted_text)
            for item in text_analysis['detected_items']:
                item['source'] = 'image_text'
                detected_items.append(item)
                total_risk += item['risk_contribution']
            logger.info(f"텍스트 분석 추가: {len(text_analysis['detected_items'])}개 항목")
        
        logger.info(f"✅ 이미지 분석 완료: 총 위험도 {total_risk:.1f}점, {len(detected_items)}개 항목")
        
        return {
            'detected_items': detected_items,
            'total_risk': min(total_risk, 100),
            'detailed_analysis': {
                'gemini_raw': gemini_result,
                'extracted_text': extracted_text,
                'risk_assessment': gemini_result.get('risk_assessment', ''),
                'sensitive_areas': gemini_result.get('sensitive_areas', [])
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Gemini API 분석 오류: {str(e)}", exc_info=True)
        # 오류 시에도 기본 위험도 반환
        return {
            'detected_items': [{
                'type': 'face',
                'count': 1,
                'risk_contribution': 15,
                'description': f'이미지 분석 중 오류 발생: {str(e)[:100]}',
                'source': 'image'
            }],
            'total_risk': 15,
            'detailed_analysis': {'error': str(e)}
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
            
            logger.info(f"조합 위험 발견: {combo_risk['name']}")
    
    return combination_risks

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
    """개선 권고사항 생성"""
    recommendations = []
    
    type_messages = {
        'phone': '📱 전화번호: 뒷자리를 가리거나 삭제하세요.',
        'email': '📧 이메일 주소: 스팸 위험이 있습니다.',
        'rrn': '⚠️ 주민등록번호: 절대 공개하지 마세요! 즉시 삭제하세요.',
        'address': '📍 주소: 동 단위까지만 공개하세요.',
        'detailed_address': '🏠 상세 주소: 번지수/호수를 삭제하세요.',
        'school': '🏫 학교명: 신원 파악의 단서가 될 수 있습니다.',
        'name': '👤 실명: 닉네임 사용을 권장합니다.',
        'credit_card': '💳 카드번호: 즉시 삭제하세요!',
        'account': '🏦 계좌번호: 금융 정보는 절대 공개하지 마세요.',
        'face': '😊 얼굴: 모자이크나 스티커로 가리세요.',
        'face_clear': '⚠️ 선명한 얼굴: 얼굴 인식 가능, 반드시 가리세요.',
        'workplace': '🏢 직장 정보: 신원 파악에 활용될 수 있습니다.',
        'birth_date': '📅 생년월일: 신원 도용에 악용될 수 있습니다.',
        'car_number': '🚗 차량번호: 가려주세요.',
        'passport': '✈️ 여권: 즉시 삭제하세요.',
        'driver_license': '🪪 운전면허: 신분증 정보는 절대 공개하지 마세요.',
        'id_card': '⚠️ 신분증: 절대 공개하지 마세요!',
        'student_id': '🎓 학생증: 이름, 사진, 학교 정보가 노출됩니다.',
        'pharmacy_bag': '💊 약봉투: 질병 정보가 노출됩니다. 환자명과 처방내역을 가리세요.',
        'delivery_label': '📦 운송장: 이름, 주소, 전화번호가 모두 노출됩니다.',
        'wedding_invitation': '💒 청첩장: 연락처와 장소 정보를 가리세요.',
        'ip_address': '🌐 IP 주소: 위치 추적에 악용될 수 있습니다.',
        'fingerprint': '👆 지문: 생체인증 시스템 악용 가능, 가려주세요.',
        'handwriting': '✍️ 필적: 서명이나 필체를 가려주세요.',
        'medical_info': '🏥 의료정보: 민감한 건강정보가 노출됩니다.',
        'location_landmark': '🗺️ 위치정보: 랜드마크나 간판을 가려주세요.',
    }
    
    detected_types = set([item['type'] for item in detected_items])
    for item_type in detected_types:
        if item_type in type_messages:
            recommendations.append(type_messages[item_type])
    
    # 조합 위험 권고
    for combo_risk in combination_risks:
        if combo_risk['severity'] == 'high':
            recommendations.append(f"⚠️ {combo_risk['description']}")
        else:
            recommendations.append(f"💡 {combo_risk['description']}")
    
    if len(detected_items) > 5:
        recommendations.append('⚠️ 다수의 개인정보 노출: 전반적인 재검토가 필요합니다.')
    
    if not recommendations:
        recommendations.append('✅ 개인정보 노출 위험이 낮습니다. 계속 주의하세요.')
    
    return recommendations

def generate_personalized_feedback(detected_items: List[Dict], 
                                   combination_risks: List[Dict],
                                   user_context: Optional[Dict] = None) -> str:
    """개인 맞춤형 피드백 생성"""
    risk_types = Counter([item['type'] for item in detected_items])
    total_risk = sum([item['risk_contribution'] for item in detected_items])
    
    feedback_parts = []
    
    # 전반적 평가
    if total_risk >= 70:
        feedback_parts.append("⚠️ 매우 위험한 수준의 개인정보가 노출되어 있습니다. 즉시 조치하세요.")
    elif total_risk >= 50:
        feedback_parts.append("⚡ 주의가 필요한 수준의 개인정보가 감지되었습니다.")
    elif total_risk >= 30:
        feedback_parts.append("💡 일부 개인정보가 노출되어 주의가 필요합니다.")
    else:
        feedback_parts.append("✅ 개인정보 노출 위험이 비교적 낮습니다.")
    
    # 주요 위험 강조
    high_risk_items = [
        ('id_card', '신분증'),
        ('rrn', '주민등록번호'),
        ('credit_card', '카드번호'),
        ('face_clear', '선명한 얼굴'),
        ('pharmacy_bag', '약봉투'),
        ('student_id', '학생증'),
        ('delivery_label', '운송장'),
    ]
    
    critical_items = [name for type_key, name in high_risk_items if type_key in risk_types]
    if critical_items:
        feedback_parts.append(f"가장 심각한 위험: **{critical_items[0]}** 노출입니다.")
    
    # 조합 위험
    if combination_risks:
        high_severity = [r for r in combination_risks if r.get('severity') == 'high']
        if high_severity:
            feedback_parts.append(f"❌ {high_severity[0]['description']}")
    
    # 구체적 조언
    if 'face_clear' in risk_types or 'id_card' in risk_types:
        feedback_parts.append("얼굴과 신분증은 반드시 모자이크 처리하세요.")
    
    if any(key in risk_types for key in ['pharmacy_bag', 'medical_info']):
        feedback_parts.append("의료정보는 매우 민감한 개인정보입니다. 노출을 피하세요.")
    
    return " ".join(feedback_parts)

# API 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "Gemini 기반 개인정보 위험 자가 진단 서비스 API",
        "version": "4.0 (Gemini AI 통합)",
        "features": [
            "Google Gemini AI 이미지 분석",
            "정규식 기반 텍스트 분석",
            "신분증/학생증/약봉투/운송장 감지",
            "얼굴/생체정보 탐지",
            "조합 위험 분석",
            "개인 맞춤 피드백"
        ],
        "supported_items": list(RISK_WEIGHTS.keys())
    }

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text_endpoint(request: TextAnalysisRequest):
    """텍스트 분석 엔드포인트"""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다")
        
        logger.info(f"텍스트 분석 요청: {len(request.text)} 글자")
        
        analysis = analyze_text_with_regex(request.text)
        combination_risks = analyze_combination_risks(analysis['detected_items'])
        
        combo_bonus = sum(risk['risk_multiplier'] * 10 for risk in combination_risks)
        final_risk = min(analysis['total_risk'] + combo_bonus, 100)
        
        recommendations = generate_recommendations(analysis['detected_items'], combination_risks)
        risk_level = get_risk_level(final_risk)
        personalized_feedback = generate_personalized_feedback(
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
        logger.error(f"텍스트 분석 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image_endpoint(file: UploadFile = File(...), user_context: str = None):
    """이미지 분석 엔드포인트 (Gemini AI 사용)"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
        
        logger.info(f"이미지 분석 요청: {file.filename}")
        
        contents = await file.read()
        analysis = await analyze_image_with_gemini(contents)
        combination_risks = analyze_combination_risks(analysis['detected_items'])
        
        combo_bonus = sum(risk['risk_multiplier'] * 10 for risk in combination_risks)
        final_risk = min(analysis['total_risk'] + combo_bonus, 100)
        
        recommendations = generate_recommendations(analysis['detected_items'], combination_risks)
        risk_level = get_risk_level(final_risk)
        
        context_dict = {}
        if user_context:
            try:
                context_dict = json.loads(user_context)
            except:
                context_dict = {'activity_type': user_context}
        
        personalized_feedback = generate_personalized_feedback(
            analysis['detected_items'], 
            combination_risks, 
            context_dict
        )
        
        logger.info(f"분석 완료: 위험도 {final_risk}, 탐지 항목 {len(analysis['detected_items'])}개")
        
        return AnalysisResponse(
            risk_score=int(final_risk),
            detected_items=analysis['detected_items'],
            combination_risks=combination_risks,
            recommendations=recommendations,
            personalized_feedback=personalized_feedback,
            risk_level=risk_level,
            detailed_analysis=analysis.get('detailed_analysis')
        )
    
    except Exception as e:
        logger.error(f"이미지 분석 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/combined")
async def analyze_combined_endpoint(
    text: Optional[str] = None,
    file: Optional[UploadFile] = File(None),
    user_context: Optional[str] = None
):
    """텍스트와 이미지 통합 분석 (Gemini + 정규식)"""
    try:
        logger.info("통합 분석 요청")
        
        total_risk = 0
        all_detected_items = []
        detailed_analysis = {}
        
        # 텍스트 분석 (정규식)
        if text:
            logger.info(f"텍스트 분석: {len(text)} 글자")
            text_analysis = analyze_text_with_regex(text)
            total_risk += text_analysis['total_risk']
            all_detected_items.extend(text_analysis['detected_items'])
        
        # 이미지 분석 (Gemini AI)
        if file:
            logger.info(f"이미지 분석: {file.filename}")
            contents = await file.read()
            image_analysis = await analyze_image_with_gemini(contents)
            total_risk += image_analysis['total_risk']
            all_detected_items.extend(image_analysis['detected_items'])
            detailed_analysis = image_analysis.get('detailed_analysis', {})
        
        # 조합 위험 분석
        combination_risks = analyze_combination_risks(all_detected_items)
        combo_bonus = sum(risk['risk_multiplier'] * 10 for risk in combination_risks)
        final_risk = min(total_risk + combo_bonus, 100)
        
        recommendations = generate_recommendations(all_detected_items, combination_risks)
        risk_level = get_risk_level(final_risk)
        
        context_dict = {}
        if user_context:
            try:
                context_dict = json.loads(user_context)
            except:
                context_dict = {'activity_type': user_context}
        
        personalized_feedback = generate_personalized_feedback(
            all_detected_items, 
            combination_risks, 
            context_dict
        )
        
        logger.info(f"통합 분석 완료: 위험도 {final_risk}, 항목 {len(all_detected_items)}개")
        
        return AnalysisResponse(
            risk_score=int(final_risk),
            detected_items=all_detected_items,
            combination_risks=combination_risks,
            recommendations=recommendations,
            personalized_feedback=personalized_feedback,
            risk_level=risk_level,
            detailed_analysis=detailed_analysis
        )
    
    except Exception as e:
        logger.error(f"통합 분석 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크"""
    gemini_status = "active" if GEMINI_API_KEY != "YOUR_API_KEY_HERE" else "not_configured"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gemini_ai": gemini_status,
            "text_analysis": "active",
            "combination_analysis": "active"
        },
        "supported_patterns": len(PATTERNS),
        "supported_risk_items": len(RISK_WEIGHTS)
    }

@app.get("/api/info")
async def api_info():
    """API 상세 정보"""
    return {
        "version": "4.0",
        "description": "Gemini AI 기반 개인정보 위험 자가 진단 서비스",
        "text_patterns": list(PATTERNS.keys()),
        "image_detection": list(set(RISK_WEIGHTS.keys()) - set(PATTERNS.keys())),
        "combination_risks": [r['name'] for r in COMBINATION_RISKS],
        "risk_weights": RISK_WEIGHTS,
        "endpoints": {
            "POST /analyze/text": "텍스트 분석 (정규식)",
            "POST /analyze/image": "이미지 분석 (Gemini AI)",
            "POST /analyze/combined": "텍스트 + 이미지 통합 분석",
            "POST /test/analyze": "테스트용 분석 (상세 로그)",
            "GET /health": "서버 상태 확인",
            "GET /api/info": "API 정보"
        }
    }

@app.post("/test/analyze")
async def test_analyze(
    text: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
):
    """테스트용 상세 분석 엔드포인트 (디버깅용)"""
    try:
        logger.info("=" * 50)
        logger.info("🔍 테스트 분석 시작")
        logger.info("=" * 50)
        
        result = {
            "text_analysis": None,
            "image_analysis": None,
            "final_result": None
        }
        
        # 텍스트 분석
        if text:
            logger.info(f"📝 텍스트 길이: {len(text)} 글자")
            logger.info(f"📝 텍스트 내용: {text[:100]}...")
            text_result = analyze_text_with_regex(text)
            result["text_analysis"] = text_result
            logger.info(f"✅ 텍스트 분석 완료: {len(text_result['detected_items'])}개 항목, 위험도 {text_result['total_risk']}")
        
        # 이미지 분석
        if file:
            logger.info(f"🖼️ 이미지 파일: {file.filename}")
            contents = await file.read()
            logger.info(f"🖼️ 이미지 크기: {len(contents)} bytes")
            image_result = await analyze_image_with_gemini(contents)
            result["image_analysis"] = image_result
            logger.info(f"✅ 이미지 분석 완료: {len(image_result['detected_items'])}개 항목, 위험도 {image_result['total_risk']}")
        
        # 통합 결과
        all_items = []
        total_risk = 0
        
        if result["text_analysis"]:
            all_items.extend(result["text_analysis"]["detected_items"])
            total_risk += result["text_analysis"]["total_risk"]
        
        if result["image_analysis"]:
            all_items.extend(result["image_analysis"]["detected_items"])
            total_risk += result["image_analysis"]["total_risk"]
        
        combination_risks = analyze_combination_risks(all_items)
        combo_bonus = sum(risk['risk_multiplier'] * 10 for risk in combination_risks)
        final_risk = min(total_risk + combo_bonus, 100)
        
        result["final_result"] = {
            "total_items": len(all_items),
            "base_risk": total_risk,
            "combo_bonus": combo_bonus,
            "final_risk": final_risk,
            "risk_level": get_risk_level(final_risk),
            "combination_risks_count": len(combination_risks)
        }
        
        logger.info("=" * 50)
        logger.info(f"🎯 최종 결과: {final_risk}점 ({result['final_result']['risk_level']})")
        logger.info(f"   - 기본 위험: {total_risk}점")
        logger.info(f"   - 조합 보너스: {combo_bonus}점")
        logger.info(f"   - 탐지 항목: {len(all_items)}개")
        logger.info("=" * 50)
        
        return result
    
    except Exception as e:
        logger.error(f"❌ 테스트 분석 오류: {str(e)}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("🚀 Gemini 기반 개인정보 분석 서버 시작...")
    logger.info(f"📊 지원 패턴: {len(PATTERNS)}개")
    logger.info(f"🔍 위험 항목: {len(RISK_WEIGHTS)}개")
    logger.info(f"⚠️ 조합 위험: {len(COMBINATION_RISKS)}개")
    
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("⚠️ Gemini API Key가 설정되지 않았습니다!")
        logger.warning("환경변수 GEMINI_API_KEY를 설정해주세요.")
    else:
        logger.info("✅ Gemini API 설정 완료")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)