import numpy as np
import math
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import glob
import gc

from tqdm import tqdm
import pickle
import os 

def overlap_score(p_s, p_e, g_s, g_e):
    inter = max(min(g_e, p_e) - max(g_s, p_s), 0.0)
    union = max(g_e, p_e) - min(g_s, p_s)
    if union <= 0:
        return 0.0
    return inter / union

file_path = 'test_person.txt'
POSE_PATH = r"/content/drive/MyDrive/backup/data/data/pose_new_v2"

def load_text_file(file_path, POSE_PATH):
    pose_filename_list = os.listdir(POSE_PATH)
    test_filename_list = []
    
    with open(file_path, "r") as f:
        for line in f:
            for p in pose_filename_list:
                
                if p.split("_")[0] == line.split("_")[0]:   
                    test_filename_list.append(p)

    return test_filename_list
video_test_list = load_text_file(file_path, POSE_PATH)

def eval():
    total = true  = 0 
    POSE_PATH = r"/content/drive/MyDrive/COBOT/CTR-GCN/CTR-GCN/mgsan_predictions/mgsan_online"
    video_test_list = load_text_file(file_path, POSE_PATH)
    s = 0
    saving_folder = "csv_results"
    gt_saving_folder = "gt_csv_results"
    for video in tqdm(video_test_list):
        if video.endswith(".txt"):
            continue
        subject = video[:video.rfind("_")]
        video_dir = f"/content/drive/MyDrive/backup/ddnet/DD-Net-Pytorch/Annotation_v4/{subject}/{video[:-4]}.csv"
        ann_csv = pd.read_csv(video_dir)
        pred_label_list = np.load(os.path.join(POSE_PATH, video), allow_pickle=True)
        #print(pred_label_list.shape)
        # ws = 30 
        pred_label_list  = np.concatenate((np.zeros((30,1)), pred_label_list.reshape(-1,1)), axis=0)
        # for i in range (pred_label_list.shape[0]-ws):
        #     if i%3!=0: continue
        #     sub_arr = pred_label_list[i:i+ws]
        #     if sum(sub_arr[0:1]) == sum(sub_arr [ :]):
        #         for j in range(1, ws ):
        #             if sub_arr[j]!= sub_arr[0]:
        #                 sub_arr[j] = sub_arr[0]
        #     pred_label_list[i:i+ws] = sub_arr    
        #pred_label_list = np.concatenate((np.zeros(30), pred_label_list))
        # convert pred_label_list to  a csv file that can be read by the annotation reader with the same format as the ground truth csv file that has
        # the following columns: start, stop, ID
        start_list = list()
        stop_list = list()
        label_list = list()
        start_list.append(0)

        for i in range(1, len(pred_label_list)):
            if pred_label_list[i] != pred_label_list[i-1]:
                start_list.append(i)
                stop_list.append(i)
                label_list.append(int(pred_label_list[i-1 ][0]))
                print(pred_label_list[i] , pred_label_list[i-1])
        stop_list.append(len(pred_label_list))
        label_list.append(int(pred_label_list[-1][0]))
        pred_csv = pd.DataFrame({'start': start_list, 'stop': stop_list, 'ID': label_list})
        pred_csv.to_csv(os.path.join(saving_folder, f"{video[:-4]}.csv"), index=False)
        
        gt_label_list = list()
        gt_label_list.extend([0 for i in range(ann_csv['start'][0])])
        for idx, label in enumerate(ann_csv['ID']):
                
                try: 
                    start_frame = ann_csv['stop'][idx-1]; stop_frame = ann_csv['start'][idx]
                    label_int = 0 
                    gt_label_list.extend([label_int for i in range(start_frame, stop_frame)])
                except:
                    pass

                start_frame = ann_csv['start'][idx]; stop_frame = ann_csv['stop'][idx]
                label_int = int(label) + 1
                gt_label_list.extend([label_int for i in range(start_frame, stop_frame)])
        gt_label_list = np.array(gt_label_list)
        
        gt_start_list = list()
        gt_stop_list = list()
        gt_label_list_ = list()
        gt_start_list.append(0)
        for i in range(1, len(gt_label_list)):
            if gt_label_list[i] != gt_label_list[i-1]:
                gt_start_list.append(i)
                gt_stop_list.append(i)
                gt_label_list_.append(int(gt_label_list[i-1]))
                
        gt_stop_list.append(len(gt_label_list))
        gt_label_list_.append(int(gt_label_list[-1]))
        
        
        gt_csv = pd.DataFrame({'start': gt_start_list, 'stop': gt_stop_list, 'ID': gt_label_list_})
        gt_csv.to_csv(os.path.join(gt_saving_folder, f"{video[:-4]}.csv"), index=False)
        
        true+= sum(a == b for a, b in zip(gt_label_list, pred_label_list[:gt_label_list.shape[0]])) 
        total+=len(gt_label_list)
        #print(total)
        
        
        
    print(f"Frame-wise Similarity: {true/total}")   
    
eval()
                
               
