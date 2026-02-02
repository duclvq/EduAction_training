# hand and body keypoint detector by using mediapipe

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
import cv2
import matplotlib.pyplot as plt 
import mediapipe as mp
import numpy as np
import time
# from ddnet import data_generator, DDNet, C
# define the video path and model path
video_path = "..\video\s21_HoangSon\s21_HoangSon\s21_HoangSon_1.mp4"
model_path = ".\models\pose_landmarker_full.task"
hand_model_path = ".\models\hand_landmarker.task"
# Import mediapipe and its dependencies
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)
hand_option = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO, num_hands=2)
landmarker = PoseLandmarker.create_from_options(options)
hand_landmarker = HandLandmarker.create_from_options(hand_option)
video_reader = cv2.VideoCapture(video_path)

def extract_hand(result):
    rh = lh = np.zeros((1, 3))
    
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        assert len(handedness) == 1, "Unexpected Error in handedness"
        
        if handedness[0].category_name == 'Left':
            lh = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
        if handedness[0].category_name == 'Right':
            rh = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    hand_pose = np.concatenate((rh, lh), axis=0)
    return hand_pose


def extract_arm(result):
    extracted_keypoints = [
        solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
        solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
        solutions.pose.PoseLandmark.LEFT_ELBOW.value,
        solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
        solutions.pose.PoseLandmark.LEFT_WRIST.value,
        solutions.pose.PoseLandmark.RIGHT_WRIST.value
    ]
    n_keypoints = len(extracted_keypoints)
    arm_pose = np.zeros((n_keypoints, 3))
    
    for landmarks in result.pose_landmarks:
        extracted_lms = [landmarks[i] for i in extracted_keypoints]
        for i in range(n_keypoints):
            lm = extracted_lms[i]
            arm_pose[i] = (lm.x, lm.y, lm.z)
            
    return arm_pose    

def predict(pose_list):
    pose_list = np.array(pose_list).squeeze()
    sample = {
        'pose': pose_list,
        'label': -1
    }
    X_0, X_1 = data_generator(sample, C)
    Y = DDNet.predict([X_0, X_1])
    return Y 

# create a queue to store pose_list
import queue
infer_len = 32
pose_list = queue.Queue(maxsize=infer_len)
predict_every = 8
#####
frame_count = 0 
while True:
    ret, image = video_reader.read()
    if not ret:
        break
    frame_count += 1
    # Process the image
    # with PoseLandmarker.create_from_options(options) as landmarker:
    numpy_frame_from_opencv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
    frame_timestamp_ms = int(video_reader.get(cv2.CAP_PROP_POS_MSEC))
    # print(frame_timestamp_ms)
    extract_time = time.time()  
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    print('extract time:', time.time() - extract_time)
        # print(pose_landmarker_result)
    # get width and height of the image
    height, width, _ = image.shape
    pose = extract_arm(pose_landmarker_result)
    hand_pose = extract_hand(hand_landmarker_result)
    
    # concatenate pose and hand_pose
    concat_pose = np.concatenate(( hand_pose, pose), axis=0)
    if pose_list.full():
            pose_list.get_nowait()  # Remove the oldest item from the queue
    pose_list.put_nowait(concat_pose)
    print('pose_list:', pose_list.qsize())
    # print('ypred', predict(pose_list))
    # if frame_count % predict_every == 0:
    #     label = predict(pose_list)
    #     print(label)
    # print('frame:', frame_count)
        
    # print(pose.shape)
    # print(hand_pose.shape)
    # draw keypoints by cv2 and pose_list
    
    
    # draw keypoints
    # if pose_landmarker_result.pose_landmarks:
        
    #     for pose in pose_list:
    #         cv2.circle(image, pose, 3, (0, 255, 0), -1)
    
    # Display the image
    # cv2.imshow('MediaPipe Pose Landmarker', image)
    # # plot whole body by using matplotlib
    # # create a new figure that has a specific size like the image
    
    # cv2.imshow('MediaPipe Pose Landmarker', image)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break
video_reader.release()

    