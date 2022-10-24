#!/usr/bin/python

import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
import pickle
import pandas as pd
from math import acos, degrees
import Exercises.UpcSystemCost as UpcSystemCost

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose    

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 


def start(sets, reps, secs, df_trainer_coords, df_trainers_costs):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    sets_counter = 0 
    stframe = st.empty()
    start = 0
    counter = 0
    resultados_acum = []
    frames_sec = 15
    df_results_coords_total = pd.DataFrame()
    up = False
    down = False

    while sets_counter < sets:
        # Squats reps_counter variables
        reps_counter = 0
        stage = ""
         # Load Model.

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            cap.isOpened()
            while reps_counter < reps:
                ret, frame = cap.read()
                height, width, _ = frame.shape
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                    # Concate rows
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = LoadModel().predict(X)[0]
                    body_language_prob = LoadModel().predict_proba(X)[0]
                    body_language_prob1 = body_language_prob*100
                    body_language_prob1=round(body_language_prob1[np.argmax(body_language_prob1)],2)


                    x1 = int(landmarks[24].x * width)
                    y1 = int(landmarks[24].y * height)
                    x2 = int(landmarks[26].x * width)
                    y2 = int(landmarks[26].y * height)
                    x3 = int(landmarks[28].x * width)
                    y3 = int(landmarks[28].y * height)
                    p1 = np.array([x1, y1])
                    p2 = np.array([x2, y2])
                    p3 = np.array([x3, y3])
                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)
                        

                    # Calculate angle
                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    print(f'angle: {angle}')

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                    if body_language_prob1 > 60:    
                        if angle >= 160:
                                    up = True
                                    stage = "up"
                        if up == True and down == False and angle <= 70:
                                    down = True
                                    stage = "down"
                        if up == True and down == True and angle >= 160:
                                    # #funcion de Costos()
                        #         # df_results_coords_total = UpcSystemCost.process(frame_rgb,mp_drawing,mp_pose,results,
                        #         #                                                 counter,start,frames_sec,df_trainer_coords,
                        #         #                                                 df_trainers_costs,df_results_coords_total,
                        #         #                                                 sets_counter,reps_counter)
                        #         # counter +=1
                        #         # start +=1
                        #         #inicio,c,results,resultados_acum=start_cost(inicio,c,results,resultados_acum)
                                    print(f'Paso')
                                    reps_counter += 1
                                    up = False
                                    down = False
                                    stage = "up"
                    else:               
                         stage = ""
                    # Setup status box
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    
                    # Set data
                    cv2.putText(image, 'SET', (15,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(sets_counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                    # Rep data
                    cv2.putText(image, 'REPS', (65,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(reps_counter), 
                                (60,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                    
                    # Stage data
                    cv2.putText(image, 'STAGE', (115,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (110,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                    
                    # Setup status box
                    # cv2.rectangle(image, (0, 480), (225, 407), (245,117,16), -1)

                    # Class data
                    cv2.putText(image, 'CLASS', (15,427), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(body_language_class), 
                                (10,467), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

                    # Prob data
                    cv2.putText(image, 'PROB', (125,427), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(body_language_prob1), 
                                (120,467), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)

                    # Used to end early
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                except:
                    pass   
            sets_counter += 1                
            if (sets_counter!=sets):
                try:
                    cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    cv2.waitKey(1)
                    time.sleep(secs)   

                except:
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    pass 
                           
    cv2.rectangle(image, (50,180), (600,400), (0,255,0), -1)
    cv2.putText(image, 'FINISHED EXERCISE', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(image, 'REST FOR 30s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    #df_results_coords_total.to_csv("./resultados_costos/Squats_resultados_costos.csv",index=False)   
    #cv2.imshow('Mediapipe Feed', image)
    stframe.image(image,channels = 'BGR',use_column_width=True)
    #cv2.waitKey(1) 
    time.sleep(10)          
    cap.release()
    cv2.destroyAllWindows()
    #cv2.destroyAllWindows()

def LoadModel():
    model_weights = './Exercises/model_weights/weights_body_language.pkl'
    with open(model_weights, "rb") as f:
        model = pickle.load(f)
    return model