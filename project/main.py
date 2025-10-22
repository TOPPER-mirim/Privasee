from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import re
import cv2
import numpy as np
import mediapipe as mp
import easyocr
from typing import List, Dict, Optional
import logging
from datetime import datetime
from collections import Counter
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="개인정보 위험 자가 진단 서비스 (개선됨)")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
pose_detection = mp_pose.Pose(min_detection_confidence=0.5)
hands_detection = mp_hands.Hands(min_detection_confidence=0.5)

# EasyOCR 초기화 (한국어, 영어)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# Tesseract OCR도 함께 사용 (더 정확한 인식을 위해)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract를 사용할 수 없습니다. EasyOCR만 사용합니다.")

# 요청 모델
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

# 개인정보 패턴 정의 (확장됨)
PATTERNS = {
    'phone': r'(\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})',
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'rrn': r'\d{6}[-\s]?[1-4]\d{6}',
    'address': r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)[\s]?[\w\s]+[시군구][\s]?[\w\s]+[동읍면리]',
    'detailed_address': r'\d+[-]?\d*\s*(?:번지|호)',
    'school': r'[\w]+(?:초등학교|중학교|고등학교|대학교|대학|학교)',
    'name': r'[가-힣]{2,4}(?:님|씨|학생|선생|교수|군|양)',
    'card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
    'account': r'\d{3,6}[-\s]?\d{2,6}[-\s]?\d{6,}',
    'workplace': r'[\w]+(?:회사|기업|병원|은행|대학|공사|그룹|연구소|재단)',
    'birth_date': r'(\d{4})[년\.\-/](\d{1,2})[월\.\-/](\d{1,2})[일]?',
    'age': r'(\d{1,2})[세살]|나이\s*(\d{1,2})',
    'car_number': r'\d{2,3}[가-힣]\d{4}',
    'passport': r'[A-Z]\d{8}',
    # 운전면허 패턴 강화 (지역 코드 2자리 + 2자리 + 6자리 + 2자리)
    'driver_license': r'(?:\d{2}[-\s]?[0-9]{2}|[가-힣]{2}[-\s]?[0-9]{2})[-\s]?\d{6}[-\s]?\d{2}', 
    'sns_id': r'@[a-zA-Z0-9_]{3,}',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'medical_info': r'(?:진단서|처방전|병명|질환|치료|환자|복용|투약)',
    'financial_info': r'(?:연봉|월급|급여|소득|자산|대출)',
    'id_card_keywords': r'(?:주민등록증|운전면허증|여권|신분증|등록증|주민번호|면허번호)',
    'pharmacy_keywords': r'(?:약국|조제|처방|복용|투약|용법|용량|mg|정)',
    # 여권 MRZ 패턴 (고급 패턴)
    'passport_mrz': r'[A-Z0-9<]{30,}'
}

# 위험도 가중치 (확장됨)
RISK_WEIGHTS = {
    'phone': 25, 'email': 15, 'rrn': 45, 'address': 20, 'detailed_address': 30,
    'school': 12, 'name': 10, 'card': 40, 'account': 35, 'face': 18,
    'face_clear': 25, 'body': 10, 'hands': 8, 'text_in_image': 5,
    'workplace': 15, 'birth_date': 25, 'age': 10, 'car_number': 20,
    'passport': 40, 'driver_license': 35, 'sns_id': 12, 'ip_address': 15,
    'medical_info': 30, 'financial_info': 25, 'metadata': 10,
    'location_exif': 25, 'background_info': 15, 'id_card': 45,
    'pharmacy_bag': 35, 'passport_mrz': 30
}

# 조합 위험 패턴
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
        'pattern': ['name', 'rrn', 'phone', 'birth_date'],
        'min_count': 2,
        'risk_multiplier': 3.0,
        'description': '주민등록번호와 개인정보 조합으로 신분 도용이 가능합니다'
    },
    {
        'name': '스토킹/괴롭힘 위험',
        'pattern': ['name', 'address', 'school', 'workplace', 'face'],
        'min_count': 2,
        'risk_multiplier': 1.8,
        'description': '개인 활동 장소 조합으로 스토킹이나 괴롭힘에 노출될 수 있습니다'
    },
    {
        'name': '위치 추적 위험',
        'pattern': ['location_exif', 'address', 'face', 'background_info'],
        'min_count': 2,
        'risk_multiplier': 2.2,
        'description': '위치 정보와 개인 식별 정보로 실시간 추적이 가능합니다'
    },
]

def preprocess_for_ocr(image: np.ndarray) -> List[np.ndarray]:
    """OCR 정확도 향상을 위한 다양한 전처리"""
    processed_images = []
    
    # 1. 원본
    processed_images.append(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 그레이스케일 + 이진화 (OTSU)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    
    # 3. 적응형 이진화 (신분증의 그림자 제거에 유리)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 15, 5) # 블록 크기 및 C값 조정
    processed_images.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
    
    # 4. 노이즈 제거 + 이진화
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, binary_denoised = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(cv2.cvtColor(binary_denoised, cv2.COLOR_GRAY2BGR))
    
    # 5. 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    processed_images.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
    
    return processed_images

def extract_text_enhanced(image_bytes: bytes) -> Dict[str, str]:
    """향상된 텍스트 추출 (다중 전처리 + 다중 OCR 엔진)"""
    try:
        # 이미지 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        easyocr_texts = []
        tesseract_texts = []
        
        # 다양한 전처리 적용
        processed_images = preprocess_for_ocr(image)
        
        # 각 전처리된 이미지에서 OCR 수행
        for proc_img in processed_images:
            # EasyOCR
            _, encoded_img = cv2.imencode('.jpg', proc_img)
            ocr_results = reader.readtext(encoded_img.tobytes())
            texts = [text[1] for text in ocr_results]
            easyocr_texts.extend(texts)
            
            # Tesseract OCR (사용 가능한 경우)
            if TESSERACT_AVAILABLE:
                try:
                    # 한글 + 영어 인식
                    custom_config = r'--oem 3 --psm 6 -l kor+eng'
                    tess_text = pytesseract.image_to_string(proc_img, config=custom_config)
                    if tess_text.strip():
                        tesseract_texts.append(tess_text)
                except Exception as e:
                    logger.debug(f"Tesseract OCR 오류: {str(e)}")
        
        # 중복 제거 및 결합
        unique_easyocr_texts = list(set(easyocr_texts))
        unique_tesseract_texts = list(set(tesseract_texts))
        
        combined_text = ' '.join(unique_easyocr_texts + unique_tesseract_texts)
        
        return {
            'combined_text': combined_text,
            'easyocr_text': ' '.join(unique_easyocr_texts),
            'tesseract_text': ' '.join(unique_tesseract_texts)
        }
    
    except Exception as e:
        logger.error(f"텍스트 추출 오류: {str(e)}")
        return {'combined_text': '', 'easyocr_text': '', 'tesseract_text': ''}


def detect_id_card(image: np.ndarray, all_ocr_texts: Dict[str, str], face_results) -> Dict:
    """신분증 감지 (주민등록증, 운전면허증, 여권 등) - 점수 기반 로직 강화"""
    id_card_info = {
        'detected': False,
        'type': None,
        'confidence': 0,
        'risk': 0,
        'features_found': [],
        'detection_score': 0
    }
    
    extracted_text = all_ocr_texts['combined_text']
    h, w = image.shape[:2]
    
    # --- 1. 패턴 및 키워드 매칭 (가장 높은 점수) ---
    
    # 1.1. 민감 정보 패턴 매칭 (가중치 높음)
    if re.search(PATTERNS['rrn'], extracted_text):
        id_card_info['features_found'].append('주민등록번호 패턴')
        id_card_info['detection_score'] += 35
        id_card_info['type'] = id_card_info['type'] or '주민등록증' # 초기 타입 지정
    
    if re.search(PATTERNS['driver_license'], extracted_text):
        id_card_info['features_found'].append('운전면허번호 패턴')
        id_card_info['detection_score'] += 30
        id_card_info['type'] = id_card_info['type'] or '운전면허증'
        
    if re.search(PATTERNS['passport_mrz'], extracted_text):
        id_card_info['features_found'].append('여권 MRZ 패턴')
        id_card_info['detection_score'] += 25
        id_card_info['type'] = id_card_info['type'] or '여권'

    # 1.2. 교차 검증 (EasyOCR/Tesseract 모두에서 패턴 발견 시 보너스)
    if TESSERACT_AVAILABLE:
        rrn_easy = re.search(PATTERNS['rrn'], all_ocr_texts['easyocr_text'])
        rrn_tess = re.search(PATTERNS['rrn'], all_ocr_texts['tesseract_text'])
        if rrn_easy and rrn_tess:
             id_card_info['features_found'].append('RRN 교차 검증')
             id_card_info['detection_score'] += 10 # 신뢰도 보너스

    # 1.3. 신분증 키워드 매칭
    id_keywords = {
        '주민등록증': ['주민등록증', '주민', '발급'],
        '운전면허증': ['운전면허증', '면허', '운전', '도로교통'],
        '여권': ['PASSPORT', 'REPUBLIC OF KOREA', '여권'],
    }
    
    for card_type, keywords in id_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in extracted_text)
        if matches > 0:
            id_card_info['features_found'].append(f'{card_type} 키워드')
            id_card_info['detection_score'] += (matches * 5)
            id_card_info['type'] = id_card_info['type'] or card_type # 초기 타입 지정

    # --- 2. 형태 분석 ---

    # 2.1. 카드 형태 비율 (대략 1.4:1 ~ 1.8:1)
    aspect_ratio = w / h
    if 1.4 <= aspect_ratio <= 1.8:
        id_card_info['features_found'].append('카드 형태 비율')
        id_card_info['detection_score'] += 10
    
    # --- 3. 얼굴 구성 분석 (증명사진 특징) ---
    
    if face_results.detections:
        # 2.2. 단일 얼굴 감지 (증명사진은 보통 하나)
        if len(face_results.detections) == 1:
            id_card_info['features_found'].append('단일 증명사진')
            id_card_info['detection_score'] += 10
            
            # 2.3. 얼굴이 이미지의 작은 비율을 차지하는지 (셀카가 아닌 증명사진)
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            face_ratio = bbox.width * bbox.height
            if 0.01 <= face_ratio <= 0.08: # 이미지의 1%~8% 정도
                id_card_info['features_found'].append('작은 얼굴 크기')
                id_card_info['detection_score'] += 10

    # --- 최종 판정 ---
    
    # 70점 이상일 때만 신분증으로 확정
    if id_card_info['detection_score'] >= 60:
        id_card_info['detected'] = True
        id_card_info['risk'] = RISK_WEIGHTS['id_card']
        # 신뢰도는 점수를 100점으로 정규화
        id_card_info['confidence'] = min(id_card_info['detection_score'] / 100.0, 1.0)
    
    return id_card_info

# (이하 나머지 함수는 변경 없음)

def detect_face_quality(image: np.ndarray, face_locations: list) -> Dict:
    """얼굴 선명도 및 크기 분석"""
    # ... (기존 코드 유지) ...
    quality_info = {
        'clear_faces': 0,
        'large_faces': 0,
        'total_faces': len(face_locations)
    }
    
    if not face_locations:
        return quality_info
    
    h, w = image.shape[:2]
    
    for detection in face_locations:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # 얼굴 영역 추출
        face_roi = image[max(0, y):min(h, y+height), max(0, x):min(w, x+width)]
        
        if face_roi.size > 0:
            # 선명도 측정 (라플라시안 분산)
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var > 100:  # 선명한 얼굴
                quality_info['clear_faces'] += 1
            
            # 얼굴 크기 비율 (이미지 대비)
            face_ratio = (width * height) / (w * h)
            if face_ratio > 0.05:  # 이미지의 5% 이상
                quality_info['large_faces'] += 1
    
    return quality_info

def extract_exif_data(image_bytes: bytes) -> Dict:
    """EXIF 메타데이터 추출"""
    metadata = {
        'has_gps': False,
        'has_datetime': False,
        'camera_info': False,
        'location_risk': 0
    }
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                
                if tag == 'GPSInfo':
                    metadata['has_gps'] = True
                    metadata['location_risk'] = RISK_WEIGHTS['location_exif']
                
                if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                    metadata['has_datetime'] = True
                
                if tag in ['Make', 'Model']:
                    metadata['camera_info'] = True
    
    except Exception as e:
        logger.debug(f"EXIF 데이터 추출 실패: {str(e)}")
    
    return metadata

def analyze_text(text: str) -> Dict:
    """텍스트 분석 함수 (확장됨)"""
    detected_items = []
    total_risk = 0
    
    for pattern_name, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            count = len(matches)
            risk = RISK_WEIGHTS.get(pattern_name, 10) * min(count, 3)
            total_risk += risk
            
            # 예제를 마스킹 처리
            masked_examples = []
            for match in matches[:2]:
                if isinstance(match, tuple):
                    match = ''.join(match)
                if pattern_name in ['phone', 'email', 'card', 'account', 'rrn', 'passport', 'driver_license']:
                    # 더 강력한 마스킹
                    if len(match) > 6:
                        masked = match[:3] + '*' * (len(match) - 6) + match[-3:]
                    else:
                        masked = '*' * len(match)
                else:
                    masked = match[:2] + '*' * (len(match) - 2)
                masked_examples.append(masked)
            
            detected_items.append({
                'type': pattern_name,
                'count': count,
                'risk_contribution': risk,
                'examples': masked_examples
            })
    
    return {
        'detected_items': detected_items,
        'total_risk': min(total_risk, 100)
    }

def detect_pharmacy_bag(image: np.ndarray, extracted_text: str) -> Dict:
    """약봉투/처방전 감지"""
    pharmacy_info = {
        'detected': False,
        'type': None,
        'risk': 0,
        'features_found': []
    }
    
    # 약국 관련 키워드
    pharmacy_keywords = [
        '약국', '조제', '처방', '복용', '투약', '용법', '용량',
        'pharmacy', '정', '캡슐', '알', 'mg', 'ml',
        '환자명', '처방의', '조제일', '약사'
    ]
    
    keyword_matches = sum(1 for keyword in pharmacy_keywords if keyword in extracted_text)
    
    if keyword_matches >= 2:
        pharmacy_info['detected'] = True
        pharmacy_info['type'] = '약봉투/처방전'
        pharmacy_info['risk'] = RISK_WEIGHTS['pharmacy_bag']
        pharmacy_info['features_found'].append(f'{keyword_matches}개 약국 키워드')
    
    # 날짜 패턴 (조제일자)
    date_patterns = [
        r'\d{4}[년\.\-/]\d{1,2}[월\.\-/]\d{1,2}',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}'
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, extracted_text):
            pharmacy_info['features_found'].append('조제 날짜')
            break
    
    # 용량 표시 (mg, ml 등)
    dosage_pattern = r'\d+\s*(mg|ml|정|캡슐|알|회)'
    if re.search(dosage_pattern, extracted_text):
        pharmacy_info['features_found'].append('약물 용량 정보')
    
    # 이름 패턴
    name_pattern = r'[가-힣]{2,4}(?:님|씨|환자)?'
    if re.search(name_pattern, extracted_text) and pharmacy_info['detected']:
        pharmacy_info['features_found'].append('환자명')
    
    # 충분한 특징이 발견되면 확정
    if len(pharmacy_info['features_found']) >= 2 and pharmacy_info['detected']:
        if not pharmacy_info['risk']:
            pharmacy_info['risk'] = RISK_WEIGHTS['pharmacy_bag']
    else:
        pharmacy_info['detected'] = False
    
    return pharmacy_info

def detect_background_info(image: np.ndarray, ocr_results: list) -> Dict:
    """배경 정보 분석 (간판, 표지판 등)"""
    background_risks = {
        'detected': False,
        'types': [],
        'risk': 0
    }
    
    # OCR 결과에서 배경 정보 키워드 검색
    background_keywords = [
        '간판', '병원', '학교', '은행', '마트', '아파트', 
        '빌딩', '역', '정류장', 'Hospital', 'School', 'Bank'
    ]
    
    extracted_text = ' '.join([text[1] for text in ocr_results])
    
    for keyword in background_keywords:
        if keyword in extracted_text:
            background_risks['detected'] = True
            background_risks['types'].append(keyword)
    
    if background_risks['detected']:
        background_risks['risk'] = RISK_WEIGHTS['background_info']
    
    return background_risks

def analyze_image_composition(image: np.ndarray) -> Dict:
    """이미지 구도 분석"""
    composition = {
        'has_people': False,
        'crowd_level': 'none',
        'indoor_outdoor': 'unknown',
        'brightness': 0
    }
    
    # 밝기 분석
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    composition['brightness'] = np.mean(gray)
    
    # 색상 분석으로 실내/외 추정
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])
    
    if avg_saturation > 80:
        composition['indoor_outdoor'] = 'outdoor'
    else:
        composition['indoor_outdoor'] = 'indoor'
    
    return composition

def analyze_image(image_bytes: bytes) -> Dict:
    """이미지 분석 함수 (대폭 확장됨)"""
    detected_items = []
    total_risk = 0
    detailed_analysis = {}
    
    try:
        # 이미지 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
             raise ValueError("이미지 디코딩 실패")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. EXIF 메타데이터 분석
        metadata = extract_exif_data(image_bytes)
        if metadata['has_gps']:
            total_risk += metadata['location_risk']
            detected_items.append({
                'type': 'location_exif',
                'count': 1,
                'risk_contribution': metadata['location_risk'],
                'description': 'GPS 위치 정보가 이미지에 포함되어 있습니다'
            })
        
        if metadata['has_datetime'] or metadata['camera_info']:
            risk = RISK_WEIGHTS['metadata']
            total_risk += risk
            detected_items.append({
                'type': 'metadata',
                'count': 1,
                'risk_contribution': risk,
                'description': '카메라 정보 및 촬영 시간이 포함되어 있습니다'
            })
        
        detailed_analysis['metadata'] = metadata
        
        # 2. 얼굴 탐지 (개선)
        face_results = face_detection.process(rgb_image)
        if face_results.detections:
            face_quality = detect_face_quality(image, face_results.detections)
            face_count = face_quality['total_faces']
            
            # 선명한 얼굴에 대한 높은 위험도
            if face_quality['clear_faces'] > 0:
                risk = RISK_WEIGHTS['face_clear'] * min(face_quality['clear_faces'], 3)
                total_risk += risk
                detected_items.append({
                    'type': 'face_clear',
                    'count': face_quality['clear_faces'],
                    'risk_contribution': risk,
                    'description': f'{face_quality["clear_faces"]}개의 선명한 얼굴이 감지되었습니다'
                })
            
            # 일반 얼굴 위험도
            remaining_faces = face_count - face_quality['clear_faces']
            if remaining_faces > 0:
                risk = RISK_WEIGHTS['face'] * min(remaining_faces, 3)
                total_risk += risk
                detected_items.append({
                    'type': 'face',
                    'count': remaining_faces,
                    'risk_contribution': risk,
                    'description': f'{remaining_faces}개의 얼굴이 감지되었습니다'
                })
            
            detailed_analysis['face_quality'] = face_quality
        
        # 3. 얼굴 랜드마크 분석 (정밀도 향상)
        face_mesh_results = face_mesh.process(rgb_image)
        if face_mesh_results.multi_face_landmarks:
            detailed_analysis['face_landmarks_detected'] = len(face_mesh_results.multi_face_landmarks)
        
        # 4. 신체 탐지
        pose_results = pose_detection.process(rgb_image)
        if pose_results.pose_landmarks:
            risk = RISK_WEIGHTS['body']
            total_risk += risk
            detected_items.append({
                'type': 'body',
                'count': 1,
                'risk_contribution': risk,
                'description': '신체 부위가 명확하게 감지되었습니다'
            })
            detailed_analysis['body_detected'] = True
        
        # 5. 손 탐지
        hands_results = hands_detection.process(rgb_image)
        if hands_results.multi_hand_landmarks:
            hand_count = len(hands_results.multi_hand_landmarks)
            risk = RISK_WEIGHTS['hands'] * min(hand_count, 2)
            total_risk += risk
            detected_items.append({
                'type': 'hands',
                'count': hand_count,
                'risk_contribution': risk,
                'description': f'{hand_count}개의 손이 감지되었습니다 (지문 노출 가능)'
            })
        
        # 6. OCR을 통한 텍스트 추출 (향상된 방식)
        all_ocr_texts = extract_text_enhanced(image_bytes)
        extracted_text = all_ocr_texts['combined_text']
        
        # 7. 신분증 감지 (강화된 로직)
        id_card_result = detect_id_card(image, all_ocr_texts, face_results)
        if id_card_result['detected']:
            total_risk += id_card_result['risk']
            detected_items.append({
                'type': 'id_card',
                'count': 1,
                'risk_contribution': id_card_result['risk'],
                'description': f"⚠️ {id_card_result['type'] or '신분증'}이 감지되었습니다 (신뢰도: {id_card_result['confidence']:.0%})",
                'features': id_card_result['features_found']
            })
            detailed_analysis['id_card'] = id_card_result
        
        # 8. 약봉투/처방전 감지
        pharmacy_result = detect_pharmacy_bag(image, extracted_text)
        if pharmacy_result['detected']:
            total_risk += pharmacy_result['risk']
            detected_items.append({
                'type': 'pharmacy_bag',
                'count': 1,
                'risk_contribution': pharmacy_result['risk'],
                'description': f"⚠️ {pharmacy_result['type']}이 감지되었습니다 (민감한 의료정보 포함)",
                'features': pharmacy_result['features_found']
            })
            detailed_analysis['pharmacy'] = pharmacy_result
        
        # 9. 추출된 텍스트에서 개인정보 패턴 검색
        if extracted_text:
            # 추출된 텍스트에서 개인정보 패턴 검색
            text_analysis = analyze_text(extracted_text)
            if text_analysis['detected_items']:
                for item in text_analysis['detected_items']:
                    item['source'] = 'image_text'
                    # 신분증 감지에서 이미 위험 점수가 반영된 경우, 중복 반영 방지
                    if item['type'] not in ['rrn', 'driver_license', 'passport_mrz', 'id_card_keywords']:
                         detected_items.append(item)
                         total_risk += item['risk_contribution']
                    elif not id_card_result['detected']:
                         # 신분증으로 확정되지 않은 경우에만 개별 패턴 점수 반영
                         detected_items.append(item)
                         total_risk += item['risk_contribution']
            
            # 이미지에서 텍스트가 발견된 것 자체도 위험 요소
            risk = RISK_WEIGHTS['text_in_image']
            total_risk += risk
            
            # 추출된 텍스트 샘플 저장
            text_sample = extracted_text[:100].replace('\n', ' ') + '...' if len(extracted_text) > 100 else extracted_text.replace('\n', ' ')
            
            detected_items.append({
                'type': 'text_in_image',
                'count': len(extracted_text.split()),
                'risk_contribution': risk,
                'description': f'이미지에서 텍스트가 추출되었습니다',
                'extracted_sample': text_sample
            })
            
            detailed_analysis['extracted_text'] = {
                'full_text': extracted_text,
                'length': len(extracted_text)
            }
        
        # 10. 배경 정보 분석 (EasyOCR 결과를 다시 사용)
        initial_ocr_results = reader.readtext(image_bytes) # 정확한 bbox를 위해 초기 OCR 결과를 사용
        background_info = detect_background_info(image, initial_ocr_results)
        if background_info['detected']:
            total_risk += background_info['risk']
            detected_items.append({
                'type': 'background_info',
                'count': len(background_info['types']),
                'risk_contribution': background_info['risk'],
                'description': f'배경에서 위치 특정 가능한 정보 발견: {", ".join(background_info["types"][:3])}'
            })
        
        detailed_analysis['background_info'] = background_info
        
        # 11. 이미지 구도 분석
        composition = analyze_image_composition(image)
        detailed_analysis['composition'] = composition
        
        # 12. 이미지 품질 및 해상도 분석
        h, w = image.shape[:2]
        detailed_analysis['resolution'] = {'width': w, 'height': h}
        detailed_analysis['high_resolution'] = w > 1920 or h > 1080
        
        if detailed_analysis['high_resolution']:
            total_risk += 5  # 고해상도는 더 많은 정보 노출
    
    except Exception as e:
        logger.error(f"이미지 분석 중 오류 발생: {str(e)}")
        return {'detected_items': [], 'total_risk': 0, 'detailed_analysis': {}}
    
    return {
        'detected_items': detected_items,
        'total_risk': min(total_risk, 100),
        'detailed_analysis': detailed_analysis
    }

def generate_personalized_feedback(detected_items: List[Dict], 
                                   combination_risks: List[Dict],
                                   user_context: Optional[Dict] = None) -> str:
    """규칙 기반 개인 맞춤형 피드백 생성"""
    
    # 위험 유형별 카운트
    risk_types = Counter([item['type'] for item in detected_items])
    total_risk = sum([item['risk_contribution'] for item in detected_items])
    
    feedback_parts = []
    
    # 1. 전반적인 위험도 평가
    if total_risk >= 70:
        feedback_parts.append("⚠️ 매우 위험한 수준의 개인정보가 노출되어 있습니다. 즉시 조치하세요.")
    elif total_risk >= 50:
        feedback_parts.append("⚡ 주의가 필요한 수준의 개인정보가 감지되었습니다. 민감 정보를 가려주세요.")
    elif total_risk >= 30:
        feedback_parts.append("💡 일부 개인정보가 노출되어 있어 주의가 필요합니다.")
    else:
        feedback_parts.append("✅ 개인정보 노출 위험이 비교적 낮습니다. 계속 주의하세요.")
    
    # 2. 주요 위험 요소 강조
    high_risk_items = [
        ('id_card', '신분증 (주민등록증/면허증/여권)'),
        ('rrn', '주민등록번호'),
        ('card', '카드번호/계좌번호'),
        ('face_clear', '선명한 얼굴'),
        ('location_exif', 'GPS 위치'),
    ]
    
    critical_items = [name for type_key, name in high_risk_items if type_key in risk_types]
    if critical_items:
        feedback_parts.append(f"가장 심각한 위험은 **{critical_items[0]}** 노출입니다. **절대 공개해서는 안 됩니다.**")
    
    # 3. 조합 위험 강조
    if combination_risks:
        high_severity = [r for r in combination_risks if r.get('severity') == 'high']
        if high_severity:
            feedback_parts.append(f"❌ {high_severity[0]['description']} 위험이 감지되었습니다. 여러 정보가 합쳐져 위험도가 극대화됩니다.")
    
    # 4. 사용자 컨텍스트 기반 조언
    if user_context:
        age_group = user_context.get('age_group')
        activity_type = user_context.get('activity_type')
        
        if age_group in ['youth', 'teenager']:
            feedback_parts.append("청소년의 경우 개인정보가 악용될 위험이 더 높으니, 온라인 공유에 더욱 신중해야 합니다.")
        
        if activity_type == 'SNS':
             feedback_parts.append("SNS는 전파 속도가 빠릅니다. 공유하기 전에 반드시 민감 정보를 모자이크 처리하세요.")
    
    # 5. 구체적인 개선 방법 제안
    if 'face_clear' in risk_types or 'id_card' in risk_types:
        feedback_parts.append("얼굴과 신분증의 모든 민감 정보는 모자이크 또는 검은색 마스킹이 필수입니다.")
    
    if any(key in risk_types for key in ['phone', 'address', 'workplace']):
        feedback_parts.append("연락처, 주소, 직장 등의 정보는 최소한 부분적으로 가려야 합니다.")
    
    # 피드백 조합
    return " ".join(feedback_parts)

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
        'phone': '전화번호가 노출되어 있습니다. 뒷자리를 가리거나 삭제를 권장합니다.',
        'email': '이메일 주소가 노출되어 있습니다. 스팸 메일의 위험이 있으니 주의하세요.',
        'rrn': '⚠️ 주민등록번호는 절대 공개하지 마세요. 즉시 삭제를 권장합니다.',
        'address': '상세 주소가 노출되면 위치가 특정될 수 있습니다. 동 단위까지만 공개하세요.',
        'detailed_address': '번지수와 호수가 노출되어 있습니다. 정확한 위치 특정이 가능하므로 삭제하세요.',
        'school': '학교명이 노출되어 있습니다. 신원 파악의 단서가 될 수 있습니다.',
        'name': '실명이 노출되어 있습니다. 닉네임 사용을 권장합니다.',
        'card': '⚠️ 카드번호가 노출되어 있습니다. 금융 사기의 위험이 있으니 즉시 삭제하세요.',
        'account': '⚠️ 계좌번호가 노출되어 있습니다. 금융 정보는 절대 공개하지 마세요.',
        'face': '얼굴이 노출되어 있습니다. 모자이크 처리나 스티커로 가리는 것을 권장합니다.',
        'face_clear': '⚠️ 선명한 얼굴이 노출되어 얼굴 인식이 가능합니다. 반드시 가려주세요.',
        'body': '신체가 노출되어 있습니다. 개인 식별이 가능할 수 있으니 주의하세요.',
        'hands': '손이 노출되어 있습니다. 지문이나 특징적인 부분은 가려주세요.',
        'text_in_image': '이미지에 텍스트가 포함되어 있습니다. 민감한 정보가 없는지 확인하세요.',
        'workplace': '직장 정보가 노출되어 있습니다. 개인 신원 파악에 활용될 수 있습니다.',
        'birth_date': '생년월일이 노출되어 있습니다. 신원 도용에 악용될 수 있습니다.',
        'age': '나이 정보가 노출되어 있습니다. 다른 정보와 조합하여 신원 추정이 가능합니다.',
        'car_number': '차량 번호가 노출되어 있습니다. 개인 추적에 악용될 수 있으니 가려주세요.',
        'passport': '⚠️ 여권 번호가 노출되어 있습니다. 즉시 삭제하세요.',
        'driver_license': '⚠️ 운전면허 번호가 노출되어 있습니다. 신분증 정보는 절대 공개하지 마세요.',
        'sns_id': 'SNS 계정이 노출되어 있습니다. 타 플랫폼 추적이 가능하니 주의하세요.',
        'ip_address': 'IP 주소가 노출되어 있습니다. 위치 추적에 악용될 수 있습니다.',
        'medical_info': '⚠️ 의료 정보가 노출되어 있습니다. 매우 민감한 정보이므로 삭제하세요.',
        'financial_info': '금융 정보가 노출되어 있습니다. 소득 정보는 공개하지 마세요.',
        'metadata': '이미지 메타데이터가 포함되어 있습니다. 촬영 기기와 시간 정보를 삭제하세요.',
        'location_exif': '⚠️ GPS 위치 정보가 이미지에 포함되어 있습니다. 정확한 위치가 노출됩니다. 메타데이터를 제거하세요.',
        'background_info': '배경에서 위치를 특정할 수 있는 정보가 발견되었습니다. 간판이나 표지판을 가려주세요.',
        'id_card': '⚠️ 신분증이 감지되었습니다. 주민등록증, 면허증 등 신분증은 절대 공개하지 마세요.',
        'pharmacy_bag': '⚠️ 약봉투/처방전이 감지되었습니다. 환자명, 병명, 약물 정보는 민감한 의료정보입니다. 즉시 삭제하세요.',
    }
    
    # 기본 권고사항
    detected_types = set([item['type'] for item in detected_items])
    for item_type in detected_types:
        if item_type in type_messages:
            recommendations.append(type_messages[item_type])
    
    # 조합 위험 권고사항
    for combo_risk in combination_risks:
        if combo_risk['severity'] == 'high':
            recommendations.append(f"⚠️ {combo_risk['description']} - 일부 정보를 삭제하거나 가려주세요.")
        else:
            recommendations.append(f"⚡ {combo_risk['description']} - 주의가 필요합니다.")
    
    # 일반 권고사항 추가
    if len(detected_items) > 5:
        recommendations.append('⚠️ 다수의 개인정보가 동시에 노출되어 있습니다. 전반적인 재검토가 필요합니다.')
    
    if not recommendations:
        recommendations.append('✅ 개인정보 노출 위험이 낮습니다. 하지만 항상 주의하세요.')
    
    return recommendations

@app.get("/")
async def root():
    return {
        "message": "개인정보 위험 자가 진단 서비스 API (OpenCV 기반)",
        "version": "2.1 (신분증 인식률 개선)",
        "features": [
            "고급 얼굴 감지 및 품질 분석",
            "EXIF GPS 위치 정보 추출",
            "배경 정보 분석",
            "손 및 신체 부위 감지",
            "확장된 개인정보 패턴 인식",
            "조합 위험 분석",
            "신분증 감지 로직 강화 (점수 기반, 교차 검증)"
        ]
    }

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text_endpoint(request: TextAnalysisRequest):
    """텍스트 분석 엔드포인트"""
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
        
        # 규칙 기반 개인 맞춤형 피드백 생성
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
        logger.error(f"텍스트 분석 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image_endpoint(file: UploadFile = File(...), user_context: str = None):
    """이미지 분석 엔드포인트"""
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
        
        # 규칙 기반 개인 맞춤형 피드백 생성
        personalized_feedback = generate_personalized_feedback(
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
            risk_level=risk_level,
            detailed_analysis=analysis.get('detailed_analysis')
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
    """텍스트와 이미지 통합 분석 엔드포인트"""
    try:
        total_risk = 0
        all_detected_items = []
        detailed_analysis = {}
        
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
            detailed_analysis = image_analysis.get('detailed_analysis', {})
        
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
        
        # 규칙 기반 개인 맞춤형 피드백 생성
        personalized_feedback = generate_personalized_feedback(
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
            risk_level=risk_level,
            detailed_analysis=detailed_analysis
        )
    
    except Exception as e:
        logger.error(f"통합 분석 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "face_detection": "active",
            "ocr": "active",
            "text_analysis": "active"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)