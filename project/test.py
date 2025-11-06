import cv2
import easyocr
import numpy as np
from PIL import Image

# EasyOCR Reader 초기화 (한글, 영어)
# GPU 사용 가능하면 gpu=True 설정
reader = easyocr.Reader(['ko', 'en'], gpu=False)


def extract_text_easyocr(image_path, detail=0):
    """
    EasyOCR을 사용한 텍스트 추출
    
    Args:
        image_path: 이미지 파일 경로
        detail: 0=텍스트만, 1=신뢰도 포함
    
    Returns:
        추출된 텍스트 또는 상세 정보
    """
    # 이미지 읽기
    img = cv2.imread(image_path)
    
    if img is None:
        return "이미지를 불러올 수 없습니다."
    
    # OCR 수행
    results = reader.readtext(image_path, detail=detail)
    
    if detail == 0:
        # 텍스트만 반환
        return '\n'.join(results)
    else:
        # 상세 정보 반환 (위치, 텍스트, 신뢰도)
        return results


def extract_text_with_preprocessing(image_path, detail=1):
    """
    전처리를 포함한 고정확도 텍스트 추출
    
    Args:
        image_path: 이미지 파일 경로
        detail: 0=텍스트만, 1=상세정보
    
    Returns:
        추출된 텍스트 정보
    """
    # 이미지 읽기
    img = cv2.imread(image_path)
    
    if img is None:
        return "이미지를 불러올 수 없습니다."
    
    # 전처리 1: 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 전처리 2: 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 전처리 3: 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # 전처리 4: 선명도 향상
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # 임시 파일로 저장 후 OCR 수행
    temp_path = "temp_processed.jpg"
    cv2.imwrite(temp_path, sharpened)
    
    # OCR 수행
    results = reader.readtext(temp_path, detail=detail)
    
    if detail == 0:
        return '\n'.join(results)
    else:
        return results


def extract_text_with_boxes(image_path, confidence_threshold=0.5):
    """
    바운딩 박스와 함께 텍스트 추출
    
    Args:
        image_path: 이미지 파일 경로
        confidence_threshold: 신뢰도 임계값 (0.0~1.0)
    
    Returns:
        바운딩 박스가 그려진 이미지와 텍스트 정보
    """
    img = cv2.imread(image_path)
    
    if img is None:
        return None, []
    
    # OCR 수행
    results = reader.readtext(image_path)
    
    text_info = []
    
    for (bbox, text, confidence) in results:
        # 신뢰도 필터링
        if confidence >= confidence_threshold:
            # 바운딩 박스 좌표
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            # 사각형 그리기
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            
            # 텍스트 표시
            cv2.putText(img, text, top_left, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # 정보 저장
            text_info.append({
                'text': text,
                'bbox': bbox,
                'confidence': confidence
            })
    
    return img, text_info


def extract_text_optimized(image_path, paragraph=False):
    """
    최적화된 설정으로 텍스트 추출
    
    Args:
        image_path: 이미지 파일 경로
        paragraph: True면 문단 단위, False면 줄 단위
    
    Returns:
        추출된 텍스트
    """
    # OCR 수행 (파라미터 최적화)
    results = reader.readtext(
        image_path,
        detail=0,
        paragraph=paragraph,
        decoder='beamsearch',  # 더 정확한 디코딩
        beamWidth=5,           # 빔 서치 너비
        batch_size=1,
        contrast_ths=0.1,      # 대비 임계값
        adjust_contrast=0.5,   # 대비 조정
        text_threshold=0.7,    # 텍스트 감지 임계값
        low_text=0.4,          # 낮은 신뢰도 텍스트 임계값
        link_threshold=0.4     # 텍스트 연결 임계값
    )
    
    if paragraph:
        return '\n\n'.join(results)
    else:
        return '\n'.join(results)


def compare_methods(image_path):
    """
    여러 방법을 비교하여 최적의 결과 찾기
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        각 방법의 결과
    """
    print("=== 방법 1: 기본 EasyOCR ===")
    result1 = extract_text_easyocr(image_path, detail=0)
    print(result1)
    print()
    
    print("=== 방법 2: 전처리 + EasyOCR ===")
    result2 = extract_text_with_preprocessing(image_path, detail=0)
    print(result2)
    print()
    
    print("=== 방법 3: 최적화된 설정 ===")
    result3 = extract_text_optimized(image_path, paragraph=False)
    print(result3)
    print()
    
    return result1, result2, result3


# 사용 예제
if __name__ == "__main__":
    # 이미지 경로 설정
    image_path = r"C:/Users/leeni/GitHub/Privasee/testimg_2.jpg"
    
    print("=" * 50)
    print("EasyOCR 텍스트 추출 시작")
    print("=" * 50)
    print()
    
    # 예제 1: 기본 텍스트 추출
    print("=== 기본 텍스트 추출 ===")
    text = extract_text_easyocr(image_path, detail=0)
    print(text)
    print()
    
    # 예제 2: 상세 정보와 함께 추출
    print("=== 상세 정보 추출 (신뢰도 포함) ===")
    detailed_results = extract_text_easyocr(image_path, detail=1)
    for bbox, text, confidence in detailed_results:
        print(f"텍스트: {text}")
        print(f"신뢰도: {confidence:.2%}")
        print(f"위치: {bbox}")
        print("-" * 30)
    print()
    
    # 예제 3: 바운딩 박스 시각화
    print("=== 바운딩 박스 추출 ===")
    result_img, text_info = extract_text_with_boxes(image_path, confidence_threshold=0.5)
    
    if result_img is not None:
        # 결과 이미지 저장
        cv2.imwrite("easyocr_result_with_boxes.jpg", result_img)
        print(f"총 {len(text_info)}개의 텍스트 영역 감지")
        print("바운딩 박스 이미지 저장: easyocr_result_with_boxes.jpg")
        
        # 감지된 텍스트 출력
        for info in text_info:
            print(f"- {info['text']} (신뢰도: {info['confidence']:.2%})")
    print()
    
    # 예제 4: 최적화된 추출
    print("=== 최적화된 텍스트 추출 ===")
    optimized_text = extract_text_optimized(image_path, paragraph=True)
    print(optimized_text)
    print()
    
    print("=" * 50)
    print("추출 완료!")
    print("=" * 50)