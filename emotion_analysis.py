from deepface import DeepFace

# 감정의 순서를 정의 (부정적 -> 긍정적)
emotion_order = ['angry', 'disgust', 'fear', 'sad', 'neutral', 'happy', 'surprise']

# 부정적 감정 목록
negative_emotions = ['angry', 'disgust', 'fear', 'sad']

def analyze_emotion(frame):
    result = DeepFace.analyze(img_path=frame, actions=['emotion'],
                              enforce_detection=False,
                              detector_backend="ssd",
                              align=True,
                              silent=True)[0]  # silent=True로 경고 출력 방지

    dominant_emotion = result['dominant_emotion']
    neutral_value = result['emotion'].get('neutral', 100)
    texts = [
        f"Dominant Emotion: {dominant_emotion} {round(result['emotion'][dominant_emotion], 1)}%",
    ]

    # Determine border color based on emotion
    if (dominant_emotion == "neutral" and neutral_value < 30) or dominant_emotion in negative_emotions:
        border_color = (0, 0, 255)  # Red border for negative or low neutral emotion
    else:
        border_color = (0, 255, 0)  # Green border for positive or high neutral emotion

    return result, texts, border_color