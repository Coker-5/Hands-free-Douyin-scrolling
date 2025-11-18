import time
from loguru import logger
import keyboard
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_pen = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles



GESTURE_STATU=""

SWIPE_UP = False  # 上滑
pre_up_point = None
prev_up_time = 0
cold_up_time = 1.0  # 上滑冷却时间
up_distance = 0.1  # 上滑额定阈值

SWIPE_DOWN = False  # 下滑
pre_down_point = None
prev_down_time = 0
cold_down_time = 4  # 下滑冷却时间
down_distance = 0.1  # 下滑额定阈值

LIKE_STATU=False # 点赞
pre_like_time=0
cold_like_time=5 # 点赞冷却时间




# 初始化手部模型
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6,
                       min_tracking_confidence=0.5)

# 开启摄像头
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# 检测上滑动作
def detect_swipe_up(hands_landmarks):
    global SWIPE_UP, pre_up_point, prev_up_time
    now_point = hands_landmarks.landmark[8].y
    bottom_index_finger=hands_landmarks.landmark[5].y
    distance = pre_up_point - now_point

    # 判断滑动距离
    if distance > up_distance:
        current_time = time.time()
        # 判断滑动时间
        if current_time - prev_up_time > cold_up_time:
            # 判断食指关键点位置
            if now_point<bottom_index_finger:
                SWIPE_UP = True
                prev_up_time = current_time


# 检测下滑动作
def detect_swipe_down(hands_landmarks):
    global SWIPE_DOWN, pre_down_point, prev_down_time
    now_point = hands_landmarks.landmark[8].y
    distance = now_point - pre_down_point

    # 判断滑动距离
    if distance > down_distance:
        # 判断滑动时间
        current_time = time.time()
        if current_time - prev_down_time > cold_down_time:
                SWIPE_DOWN = True
                prev_down_time = current_time


# 检测点赞动作
def detect_like(hands_landmarks):
    global LIKE_STATU, pre_like_time, cold_like_time
    thumb_top_point=hands_landmarks.landmark[4].y
    thumb_bottom_point=hands_landmarks.landmark[1].y

    center_point=hands_landmarks.landmark[0].y
    little_finger_bottom_point=hands_landmarks.landmark[17].y

    index_finger_top_point=hands_landmarks.landmark[8].y
    index_finger_bottom_point=hands_landmarks.landmark[5].y

    # 判断大拇指，食指，小拇指关键点位置
    if thumb_top_point<thumb_bottom_point and center_point<little_finger_bottom_point and index_finger_bottom_point<index_finger_top_point:
        current_time = time.time()
        if current_time - pre_like_time > cold_like_time:
            LIKE_STATU = True
            pre_like_time = current_time

def main():
    global pre_up_point, prev_up_time, pre_down_point, prev_down_time, SWIPE_UP, SWIPE_DOWN,GESTURE_STATU,LIKE_STATU
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print('Failed to capture image.')
                break

            frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (240, 180))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 开始检测
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:


                    mp_pen.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style()
                                          , mp_styles.get_default_hand_connections_style())
                    cv2.putText(frame, f"GESTURE: {GESTURE_STATU}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                    # 检测上滑动作
                    if pre_up_point is not None:
                        detect_swipe_up(hand_landmarks)

                    pre_up_point = hand_landmarks.landmark[8].y
                    if SWIPE_UP:
                        logger.info("SWIPE UP")
                        GESTURE_STATU="SWIPE UP"
                        keyboard.press_and_release('down')
                        SWIPE_UP = False


                    # 检测下滑动作
                    if pre_down_point is not None:
                        detect_swipe_down(hand_landmarks)

                    pre_down_point = hand_landmarks.landmark[8].y
                    if SWIPE_DOWN:
                        logger.info("SWIPE DOWN")
                        GESTURE_STATU = "SWIPE DOWN"
                        keyboard.press_and_release('up')
                        SWIPE_DOWN = False


                    # 检测点赞动作
                    detect_like(hand_landmarks)

                    if LIKE_STATU:
                        logger.info("LIKE")
                        GESTURE_STATU = "LIKE"
                        for _ in range(4):
                            keyboard.press_and_release('z')
                        LIKE_STATU = False




            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
