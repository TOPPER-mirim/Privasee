from google import genai
from PIL import Image
import os
import io

YOUR_API_KEY = "AIzaSyBHDNQa_5rVWZwLJzGafR9EUtp4ZX1oKBA"
IMAGE_PATH = "C:/Users/leeni/GitHub/Privasee/testimg_2.jpg"

# 멀티모달(텍스트+이미지) 처리에 최적화된 모델
MODEL_NAME = "gemini-2.5-flash" 

# 이미지를 인식하고 텍스트만 추출하도록 지시하는 프롬프트
PROMPT = "이미지에서 모든 텍스트를 정확하게 추출해줘. 추출한 텍스트만 출력해야 해."


def extract_text_from_image(api_key: str, image_path: str, prompt: str, model_name: str):
    """
    주어진 이미지 경로에서 텍스트를 추출하고 결과를 반환합니다.
    """
    # 2-1. 파일 존재 여부 확인
    if not os.path.exists(image_path):
        return f"🚨 오류: 이미지 파일이 경로에 존재하지 않습니다: {image_path}"
    
    # 2-2. 클라이언트 초기화 (API 키 사용)
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return f"🚨 오류: Gemini 클라이언트 초기화 실패. API 키를 확인하세요. ({e})"

    # 2-3. 이미지 로드 및 API 호출
    try:
        # 이미지를 PIL.Image 객체로 로드합니다.
        img = Image.open(image_path)
        
        print("🔍 이미지 처리 및 텍스트 추출 중...")

        # 이미지와 프롬프트를 함께 모델에 전달하여 내용 생성 요청
        response = client.models.generate_content(
            model=model_name,
            contents=[img, prompt]
        )
        
        # 모델의 응답에서 텍스트 부분만 반환
        return response.text

    except Exception as e:
        return f"🚨 Gemini API 호출 중 오류가 발생했습니다: {e}"

# ==========================================================
# 3. 메인 실행 블록
# ==========================================================

# 함수 실행 및 결과 출력
extracted_text = extract_text_from_image(YOUR_API_KEY, IMAGE_PATH, PROMPT, MODEL_NAME)

print("-" * 50)
print(f"사용된 모델: {MODEL_NAME}")
print(f"사용된 프롬프트: '{PROMPT}'")
print("-" * 50)
print("추출된 최종 텍스트 결과")
print(extracted_text)
print("-" * 50)