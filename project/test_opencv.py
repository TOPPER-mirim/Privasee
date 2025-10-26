# ============================================
# 이미지 분석 기초 튜토리얼
# 1. 얼굴 감지하기
# 2. 텍스트 추출하기
# 3. 결과 출력하기
# ============================================

import cv2
import numpy as np
import mediapipe as mp
import easyocr
from PIL import Image

print("=" * 50)
print("이미지 분석 튜토리얼 시작!")
print("=" * 50)

# ============================================
# Step 1: 이미지 불러오기
# ============================================
print("\n[Step 1] 이미지 불러오기")

# 방법 1: 파일에서 이미지 읽기
image_path = "C:/Users/leeni/GitHub\Privasee/testimg_1.jpg"  # 여기에 분석할 이미지 경로 입력

try:
    # OpenCV로 이미지 읽기
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ 이미지를 찾을 수 없습니다: {image_path}")
        print("💡 이미지 파일을 같은 폴더에 넣고 파일명을 확인하세요!")
        exit()
    
    # 이미지 정보 출력
    height, width, channels = image.shape
    print(f"✅ 이미지 로드 성공!")
    print(f"   - 크기: {width} x {height} 픽셀")
    print(f"   - 채널: {channels} (BGR)")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    exit()

print("\n[Step 3] 텍스트 추출 시작...")
print("   (처음 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")

# EasyOCR 초기화 (한국어, 영어)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# OCR 실행
print("   텍스트 인식 중...")
ocr_results = reader.readtext(image)

# 결과 확인
if ocr_results:
    print(f"✅ 텍스트 {len(ocr_results)}개 발견!")
    
    extracted_texts = []
    
    for idx, detection in enumerate(ocr_results, 1):
        # detection = (좌표, 텍스트, 신뢰도)
        bbox, text, confidence = detection
        
        # 신뢰도 30% 이상만 사용
        if confidence > 0.3:
            extracted_texts.append(text)
            
            print(f"\n   텍스트 #{idx}:")
            print(f"   - 내용: '{text}'")
            print(f"   - 신뢰도: {confidence * 100:.1f}%")
            
            # 이미지에 텍스트 영역 표시
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    
    # 전체 텍스트 합치기
    full_text = " ".join(extracted_texts)
    print(f"\n   추출된 전체 텍스트:")
    print(f"   '{full_text}'")
    
else:
    print("❌ 텍스트를 찾을 수 없습니다.")
    full_text = ""
