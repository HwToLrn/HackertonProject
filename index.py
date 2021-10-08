from typing import List, Tuple, Union, Optional, Mapping

import numpy as np
import mediapipe as mp
import cv2

import dataclasses
import math
from mediapipe.framework.formats import landmark_pb2

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

CUSTOMIZED_POSE_CONNETIONS: List[Tuple[int, int]] = [
    (11, 12), (11, 13), (11, 23), (12, 14), (12, 24),
    (13, 15), (14, 16), (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22), (17, 19), (18, 20),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32), (29, 31), (30, 32)
]


@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> \
Union[None, Tuple[int, int]]:
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        # Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def customized_draw_landmarks(image: np.ndarray,
                              landmark_list: landmark_pb2.NormalizedLandmarkList,
                              connections: Optional[List[Tuple[int, int]]] = None,
                              landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(
                                  color=RED_COLOR),
                              connection_drawing_spec: Union[
                                  DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):
    if not landmark_list:
        return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    # revised_list = landmark_list.landmark[11:] # 0~10번까지 얼굴 좌표
    for idx, landmark in zip(range(11, len(landmark_list.landmark)), landmark_list.landmark[11:]):
        if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    #             print('idx_to_coordinates[idx] = landmark_px', idx, landmark_px[0], landmark_px[1])
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], drawing_spec.color,
                         drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
            # White circle border
            circle_border_radius = max(drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2))
            cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR, drawing_spec.thickness)
            # Fill color into the circle
            cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)


# coor == coordinates: 좌표
def calculate_angle(coor_fst: List[float], coor_scd: List[float], coor_trd: List[float]) -> float:
    coor_fst = np.array(coor_fst)
    coor_scd = np.array(coor_scd)
    coor_trd = np.array(coor_trd)

    radius = np.arctan2(coor_trd[1] - coor_scd[1], coor_trd[0] - coor_scd[0]) - np.arctan2(coor_fst[1] - coor_scd[1], coor_fst[0] - coor_scd[0])
    angle = np.abs(radius * 180 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # count variables
    left_counter: int = 0
    right_counter: int = 0
    counter: int = 0
    stage: str = ''  # down or up

    # 사용자가 미리 정해 놓은 루틴 / 한 운동의 반복횟수 / 한 운동의 세트횟수
    ROUTE: List[str] = ['아령들기', '스쿼트', '팔굽혀펴기']
    REPEATS: int = 5
    SET: int = 2
    current_route: int = 0  # ROUTE[0] .. ROUTE[2] 순으로 사용
    current_set: int = 0

    video = cv2.VideoCapture(0)

    video_width = int(video.get(3))
    video_height = int(video.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 코덱 정보
    # 비디오 저장을 위한 객체를 생성
    output = cv2.VideoWriter(filename='SavedVideo/SavedVideo1.avi',
                             fourcc=fourcc, fps=20.0,
                             frameSize=(video_width, video_height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while video.isOpened():
            checkframe, frame = video.read()
            frame = cv2.flip(src=frame, flipCode=1) # 좌우 또는 상하 반전

            # Recolor 'frame' to RGB
            image = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make Detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # 루틴별로 연산 필요 -------------------------------------------------------
                # Get coordinates & Calculate a angle & Counter Logic
                if ROUTE[current_route] == '팔굽혀펴기':
                    # 어깨, 팔꿈치, 손목
                    coor_left_shoulder: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    coor_left_elbow: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    coor_left_wrist: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    coor_right_shoulder: List[float] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    coor_right_elbow: List[float] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    coor_right_wrist: List[float] = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_angle: float = calculate_angle(coor_left_shoulder, coor_left_elbow, coor_left_wrist)
                    right_angle: float = calculate_angle(coor_right_shoulder, coor_right_elbow, coor_right_wrist)

                    # 양쪽이 안보일 때는 어떻게 카운팅되는지 확인해보기
                    if left_angle >= 170 and right_angle >= 170 and stage == 'DOWN':
                        stage = 'UP'
                        counter += 1
                    if left_angle <= 100 and right_angle <= 100:
                        stage = 'DOWN'

                    if counter == REPEATS:
                        current_set += 1
                        counter = 0
                        if current_set == SET:  # 다음 운동 / 혹은 종료(구현 필요)
                            current_set = 0
                            current_route += 1

                    cv2.putText(img=image, text=str(int(left_angle)), org=(1000, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=3, color=BLUE_COLOR, thickness=2, lineType=cv2.LINE_AA)
                elif ROUTE[current_route] == '아령들기':
                    # 어깨, 팔꿈치, 손목
                    coor_left_shoulder: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    coor_left_elbow: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    coor_left_wrist: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    coor_right_shoulder: List[float] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    coor_right_elbow: List[float] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    coor_right_wrist: List[float] = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_angle: float = calculate_angle(coor_left_shoulder, coor_left_elbow, coor_left_wrist)
                    right_angle: float = calculate_angle(coor_right_shoulder, coor_right_elbow, coor_right_wrist)

                    if left_angle > 160:
                        stage = 'DOWN'
                    if left_angle < 30 and stage == 'DOWN':
                        stage = 'UP'
                        left_counter += 1

                    if right_angle > 160:
                        stage = 'DOWN'
                    if right_angle < 30 and stage == 'DOWN':
                        stage = 'UP'
                        right_counter += 1

                    if left_counter >= 1 and right_counter >= 1:
                        counter += 1
                        left_counter = 0
                        right_counter = 0
                        if counter == REPEATS:
                            current_set += 1
                            counter = 0
                            if current_set == SET:  # 다음 운동 / 혹은 종료(구현 필요)
                                current_set = 0
                                current_route += 1
                elif ROUTE[current_route] == '스쿼트':
                    # 골반(hip), 무릎(knee), 발목(ankle)
                    coor_left_hip: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    coor_left_knee: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    coor_left_ankle: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    coor_right_hip: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    coor_right_knee: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    coor_right_ankle: List[float] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    left_angle: float = calculate_angle(coor_left_hip, coor_left_knee, coor_left_ankle)
                    right_angle: float = calculate_angle(coor_right_hip, coor_right_knee, coor_right_ankle)

                    if left_angle >= 170 and right_angle >= 170 and stage == 'DOWN':
                        stage = 'UP'
                        counter += 1
                    if left_angle <= 120 and right_angle <= 120:
                        stage = 'DOWN'

                    if counter == REPEATS:
                        current_set += 1
                        counter = 0
                        if current_set == SET:  # 다음 운동 / 혹은 종료(구현 필요)
                            current_set = 0
                            current_route += 1

                    cv2.putText(img=image, text=str(int(left_angle)), org=(1000, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=3, color=BLUE_COLOR, thickness=2, lineType=cv2.LINE_AA)
                elif ROUTE[current_route] == 'something_1':
                    pass
                elif ROUTE[current_route] == 'something_2':
                    pass

                # ------------------------------------------------------------------------

                # Render curl counter
                # Setup status box
                cv2.rectangle(img=image, pt1=(0, 0), pt2=(400, 70), color=(245, 117, 16), thickness=-1)

                # Repeat data
                cv2.putText(img=image, text='REPS', org=(11, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img=image, text=str(counter), org=(10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                # Stage data
                cv2.putText(img=image, text='STAGE', org=(100, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img=image, text=stage, org=(100, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                # Set data
                cv2.putText(img=image, text='SET', org=(290, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img=image, text=str(current_set), org=(290, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            except:
                pass

            # Render detections
            # Customized things - Remove coordinates and line of face
            customized_draw_landmarks(image=image,
                                      landmark_list=results.pose_landmarks,  # same type : landmarks[32]
                                      connections=CUSTOMIZED_POSE_CONNETIONS)  # mp_pose.POSE_CONNECTIONS

            if checkframe:
                cv2.imshow(winname='MyWindow', mat=image)
                output.write(image=image)

            # ESC key for closing the window
            if cv2.waitKey(1) & 0xFF == 27:
                break

    video.release()
    output.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
