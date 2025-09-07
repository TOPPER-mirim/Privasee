import cv2
import mediapipe as mp
import easyocr

mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
reader = easyocr.Reader(['ko','en'])  # 한국어 + 영어

img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if results.detections:
    for det in results.detections:
        # bounding box 계산
        score += 20  # 얼굴 노출 보상

# OCR
ocr_result = reader.readtext(img)
for bbox, text, conf in ocr_result:
    if "고등학교" in text or "초등학교" in text:
        score += 20
        findings.append({"type":"school_text", "text": text})