import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import cv2
import time
import logging

# 감정의 순서를 정의 (부정적 -> 긍정적)
emotion_order = ['angry', 'disgust', 'fear', 'sad', 'neutral', 'happy', 'surprise']

# 1분 동안 265 프레임을 수용하기 위한 프레임당 시간 간격 계산
frame_interval = 60 / 300  # 약 0.2264초

def calculate_y_position(emotion, value):
    idx = emotion_order.index(emotion)
    if idx == len(emotion_order) - 1:
        return idx  # 마지막 감정인 경우, 바로 위치

    # 감정 간의 위치를 점수에 따라 계산
    return idx + (value / 100.0)

def plot_emotion_history(height, x_start, emotion_history, time_history, emotion_values_history, start_timestamp):
    # 플롯의 크기를 설정
    plt.figure(figsize=(24, 12))

    # X, Y 좌표 배열 초기화
    x_values = []
    y_values = []
    
    # 1분(60초) 단위로 플롯을 초기화하는 로직 추가
    plot_duration = 60  # 1분 동안 데이터를 그린 후 초기화
    current_timestamp = time.time()
    elapsed_time = current_timestamp - start_timestamp
    logging.info(f"Elapsed time: {elapsed_time} seconds, x_start={x_start}")

    if elapsed_time >= plot_duration:
        start_timestamp = current_timestamp
        
        # x_start = 0  # x축 시작을 초기화
        x_start += plot_duration
        # logging.info("Resetting plot after 60 seconds.")

        # 1분이 지나면 감정 및 시간 기록을 초기화
        time_history.clear()
        emotion_values_history.clear()    
    
    # 최소 길이에 맞춰 프레임 수만큼 처리
    min_length = min(len(emotion_history), len(emotion_values_history))

    # X, Y 값 계산
    for i in range(min_length):
        emotion = emotion_history[i]
        y_pos = calculate_y_position(emotion, emotion_values_history[i][emotion])

        # 각 프레임마다 일정 시간 간격으로 배치 (x_start + frame_interval * i)
        x_value = x_start + (frame_interval * i)
        x_values.append(x_value)
        y_values.append(y_pos)

    # 선 그래프 그리기 (항상 파란색, 선 굵기 4.0)
    plt.plot(x_values, y_values, color='blue', linewidth=4.0)

    # 글씨 크기 및 스타일 조정
    plt.yticks(ticks=range(len(emotion_order)), labels=emotion_order, fontsize=24)
    plt.xticks(fontsize=24)  # x축 눈금의 글씨 크기 설정
    plt.ylim(-0.5, len(emotion_order) - 0.5)
    plt.xlim(x_start, x_start + plot_duration)  # x축의 범위 설정 (0 ~ 60초)
    plt.xlabel('Time (seconds)', fontsize=32)  # X축 라벨 크기 설정
    plt.ylabel('Emotion', fontsize=28)  # Y축 라벨 크기 설정
    plt.title('Emotion Over Time', fontsize=32)  # 타이틀 크기 설정
    plt.grid(True)

    # 플롯 이미지를 바이트 배열로 변환
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # 이미지를 OpenCV 형식으로 변환
    plot_img = Image.open(buf)
    plot_img = np.array(plot_img)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    return plot_img, x_start, start_timestamp