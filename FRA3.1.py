#########################################################################################################
# School: Singapore Polytechnic
# Students: Foong Hao Juen and Zenden Tan
# Program Name: S10_facerecAppV1.0_withSSD.py
# Project Title: AIOT Facial Recognition Attendance Taking FPGA System
# Create Date: - 25/11/21
# Description: Face recognition using tf2
# Multi Threaded Webcam Display.
# Local Logging and Comparison Logging.
# Display attendance list using Seperate GUI displaying Comparison Log.
# Check Classroom timetable
# Check if the students entered the correct class
# late
# Firebase upload
# Tkinter Display to show different status
# Changed Tkinter Attendance list to mark students that are late in red
# # calculation thread notify user still has unsolved duplicated issues
# # capture frame modification to improve fps
# # queue empty calculation thread condition
# tkinter refresh only when .txt files are updated
# display unknown status
# Audio
# let the bounding boxes, status and names last for 15 frames
# readjusted tkinter display to fit advantech monitor and extra column for studentid
# Only display studentid on recognized students
# SSD phone detection for anti spoofing
########################################################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import calendar
import copy
import datetime
import numpy as np
import os
import pickle
import pyttsx3
import queue
import time
import tensorflow as tf
import threading
from tkinter import *
import tkinter as tk
from PIL import Image

import fireb
import facenet
import notify_user
import logfileyd_new as logfileyd
import check_timetable_new as check_timetable

modeldir = './model/20170511-185253.pb'
# modeldir = './model/20180402-114759.pb'
classifier_filename = './class/HJV6.pkl'
file_name = "logsv4.txt"

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
classFile = "coco.names"

# face_cascade = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

ASmodeldir = './ASFolder/mbnAS.h5'
ASweightdir = './ASFolder/mbnWeight.h5'
ASmodel = tf.keras.models.load_model(ASmodeldir)
ASmodel.load_weights(ASweightdir)

global check_quit, video_capture, embedding_size, images_placeholder, phase_train_placeholder, sess, embeddings, model, \
    net, classNames, refresh, refresh2, TTS_name, attendance_line, display_lines, queue_speech, queue_attendance_upload, \
    queue_image_upload, queueout_frame, uploading, speaking

root = tk.Tk()
screen_width = int(root.winfo_screenwidth())
screen_height = int(root.winfo_screenheight())

frame_height = int(screen_height * 0.425)
frame_width = int(frame_height * 1.33)

attendance_win_width = (screen_width - frame_width)
attendance_win_height = int(screen_height)
cell_size = int(attendance_win_width / 6 / 8.75)

status_win_width = int(frame_width)
status_win_height = int(screen_height - frame_height)


def capture_frame(queueout_frame, queue_recognition):
    global check_quit, video_capture, net, classNames, model, refresh2
    frame_count = 0
    currentDate = datetime.datetime.now().day
    print("Current date:", currentDate)
    while not check_quit:
        if datetime.datetime.now().day == currentDate:
            phone_detected = False
            _, frame = video_capture.read()
            if frame is not None:
                frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

            if frame_count == 30:
                notify_user.check_for_delete(calendar.timegm(time.gmtime()))
                queueout_frame.put(frame)
                classIds, confs, bbox = net.detect(frame, confThreshold=0.63)
                if len(classIds) != 0:
                    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                        if classNames[classId - 1] == "cell phone":
                            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(frame, "NO PHONES", (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                            for i in range(3):
                                queueout_frame.put(frame)
                            phone_detected = True
                            notify_user.write_text("Phone", "Detected")
                            refresh2 = True
                            frame_count = 0

                if not phone_detected:
                    bounding_boxes = []
                    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=4, minSize=(100, 100))

                    for (x, y, w, h) in faces:
                        bounding_boxes.append((x + 8, y + 4, x + w - 15, y + h - 4))

                    nrof_faces = len(bounding_boxes)
                    queue_recognition.put((nrof_faces, bounding_boxes, frame))
                    a = threading.Thread(target=calculation_frame, args=(queue_recognition,))
                    a.start()
                    frame_count = 0
            else:
                queueout_frame.put(frame)
                frame_count += 1
        else:
            currentDate = datetime.datetime.now().day
            print("Current date:", currentDate)

            classroom = check_timetable.getRoomName()
            fireb.getTimetable(classroom)
            fireb.getclassifier(classroom)

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                print(f'Files updated for {classroom}')
    else:
        sys.exit()


def calculation_frame(queue_recognition):
    global check_quit, embedding_size, images_placeholder, phase_train_placeholder, sess, embeddings, model, refresh, \
        refresh2, queue_speech, queue_attendance_upload, queue_image_upload, queueout_frame

    image_size = 182
    input_image_size = 160

    Status = {0: ["Attendance is already taken", (0, 255, 0)],
              1: ["Late", (0, 0, 255)],
              2: ["Class is closed", (255, 0, 0)],
              3: ["You do not have your class now", (255, 0, 255)],
              4: ["Unknown", "Unknown", (204, 204, 255)]}

    while not queue_recognition.empty():
        nrof_faces, bounding_boxes, frame = queue_recognition.get()
        gotonext = True

        if nrof_faces > 0:
            save_frame = copy.copy(frame)
            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

            for i in range(nrof_faces):
                emb_array = np.zeros((1, embedding_size))

                bb[i][0] = bounding_boxes[i][0]
                bb[i][1] = bounding_boxes[i][1]
                bb[i][2] = bounding_boxes[i][2]
                bb[i][3] = bounding_boxes[i][3]

                ###################
                x0 = bb[i][0]
                y0 = bb[i][1]
                x1 = bb[i][2]
                y1 = bb[i][3]
                ''''
                #antispoof part
                frame1 = copy.copy(frame)
                face = frame1[y0:y0+y1,x0:x0+x1]
                resized_face = cv2.resize(face, (160, 160))
                resized_face = resized_face.astype("float") / 255.0
                resized_face = img_to_array(resized_face)
                resized_face = np.expand_dims(resized_face, axis=0)
                preds2 = ASmodel.predict(resized_face)[0]
                try:
                    if preds2 > 0.91:
                        print('spoof detected')
                        notify_user.write_text("Spoof", "Detected")
                        cv2.putText(frame1, "spoof", (x0, y0 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(frame1, (x0, y0), (x0 + x1, y0 + y1),
                                      (0, 0, 255), 2)
                        queueout_frame.put(frame1)

                    else:
                        gotonext = False
                except(IndexError):
                    continue
                '''
                ##################
                if gotonext:
                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        # print('Face is very close!')
                        continue
                    try:
                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                    except IndexError:
                        print("Error 2")
                        continue
                    scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size),
                                                                              Image.BILINEAR)).astype(np.double))
                    # NEED TO BE FIXED
                    scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                           interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    print(best_class_indices, ' with accuracy ', best_class_probabilities)

                    # plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3]

                    HumanNames = check_timetable.get_HumanName()

                    result_names = HumanNames[best_class_indices[0]]
                    studentid = result_names[result_names.find("(") + 1:-1]
                    text = ""

                    if best_class_probabilities >= 0.7:  # original one is 0.53
                        student_lists, class_name, module, status, start_time, end_time = check_timetable. \
                            get_student_list()
                        text, colour = Status[status]

                        if student_lists:
                            if result_names in student_lists:
                                # Text to speech
                                if TTS_name != result_names[:-10]:
                                    queue_speech.put(result_names[:-10])
                                    a = threading.Thread(target=text_to_speech)
                                    a.start()

                                # Uploading
                                temp = logfileyd.logging(result_names, module, class_name, start_time, end_time, status)
                                if temp:
                                    queue_attendance_upload.put(temp)
                                if nrof_faces == 1:
                                    queue_image_upload.put((result_names, save_frame))
                                refresh = True
                            else:
                                text, colour = Status[3]
                    else:
                        studentid, result_names, colour = Status[4]

                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), colour, 2)
                    cv2.putText(frame, studentid, (text_x, text_y + 20), cv2.FONT_HERSHEY_COMPLEX, 1, colour, 2, 2)
                    cv2.putText(frame, text, (text_x, text_y + 40), cv2.FONT_HERSHEY_COMPLEX, 1, colour, 2, 1)
                    if text == "":
                        text = "Unknown"
                    notify_user.write_text(result_names, text)
                    refresh2 = True
            a = threading.Thread(target=upload)
            a.start()
            for i in range(5):
                queueout_frame.put(frame)


def upload():
    global check_quit, uploading, queue_image_upload, queue_attendance_upload
    if not uploading:
        uploading = True
        while not queue_image_upload.empty() or not queue_attendance_upload.empty():
            if check_quit:
                break
            if not queue_image_upload.empty():
                result_names, save_frame = queue_image_upload.get()
                fireb.to_firebase(result_names, save_frame)
            if not queue_attendance_upload.empty():
                ct, student, module_code, class_name, start_time, end_time, status = queue_attendance_upload.get()
                fireb.upload(ct, student, module_code, class_name, start_time, end_time, status)
        uploading = False


def text_to_speech():
    global check_quit, TTS_name, speaking, queue_speech
    engine = pyttsx3.init()
    if not speaking:
        speaking = True
        while not queue_speech.empty():
            if check_quit:
                break
            result_names = queue_speech.get()
            TTS_name = result_names
            engine.say(f"{result_names}, your attendance is already taken")
            engine.runAndWait()
            TTS_name = None
        speaking = False


def display_frame(queueout_frame):
    global check_quit
    while not check_quit:
        if not queueout_frame.empty():
            frame = queueout_frame.get()
            cv2.imshow('Video', frame)
            cv2.moveWindow('Video', attendance_win_width, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            check_quit = True

    video_capture.release()
    cv2.destroyAllWindows()


def get_log_content():
    content = []
    with open(file_name, "r+") as file:
        records = file.readlines()
        for row in records:
            if row != "\n":
                x = row.split(" ")
                Student = x[3][:x[3].find("(")]
                ID = x[3][x[3].find("(") + 1:-1]
                Class = x[5]
                Module = x[7]
                Date = x[9]
                Time = x[11].replace("\n", "")
                content.append((Student, ID, Class, Module, Date, Time))
    return content


def display_attendance():
    global check_quit, attendance_line, display_lines
    ws = Tk()
    ws.title("Attendance List")

    ws.geometry('%dx%d+%d+%d' % (attendance_win_width - 5, attendance_win_height, 0, 0))
    title = [("Student Name", "Student ID", "Class", "Module Code", "Date", "Time")]
    for j in range(6):
        e = Entry(ws, width=cell_size, fg='green', font=('Arial', 12, 'bold'))
        e.grid(row=0, column=j)
        e.insert(END, title[0][j])
    attendance_line = 0

    def update_attendance():
        global check_quit, attendance_line, refresh
        if check_quit:
            ws.destroy()
        else:
            log_content = get_log_content()
            if len(log_content) < attendance_line:
                attendance_line = len(log_content)
                i = 0
                for widget in ws.winfo_children():
                    if isinstance(widget, tk.Entry):
                        i += 1
                        if i > 6:
                            widget.destroy()
                refresh = True

            elif len(log_content) > attendance_line or refresh:
                attendance_line = len(log_content)
                late = int(check_timetable.getLateTime())
                late *= 100
                for i in range(len(log_content)):
                    for x in range(6):
                        if int(log_content[i][5]) > late:
                            entry = Entry(ws, width=cell_size, fg='red', font=('Arial', 12, 'bold'))
                        else:
                            entry = Entry(ws, width=cell_size, fg='green', font=('Arial', 12, 'bold'))

                        entry.grid(row=i + 1, column=x)
                        entry.insert(END, log_content[i][x])
                refresh = False
            ws.after(500, update_attendance)

    update_attendance()

    top = tk.Toplevel(ws)
    top.title("Status")
    top.geometry('%dx%d+%d+%d' % (status_win_width, status_win_height - 40, attendance_win_width, frame_height + 40))

    data2 = [("Student Name", "Status")]
    e = Entry(top, width=20, fg='blue', font=('Arial', 12, 'bold'))
    e.grid(row=0, column=0)
    e.insert(END, data2[0][0])
    e = Entry(top, width=90, fg='blue', font=('Arial', 12, 'bold'))
    e.grid(row=0, column=1)
    e.insert(END, data2[0][1])
    display_lines = 0

    def update_status():
        global check_quit, display_lines, refresh2
        if check_quit:
            top.destroy()
        else:
            if refresh2:
                i = 0
                with open("notify.txt", "r") as notify:
                    line = notify.readlines()
                    for row in line:
                        x = row.split(":")
                        e1 = Entry(top, width=20, fg='blue', font=('Arial', 12, 'bold'))
                        e1.grid(row=i + 1, column=0)
                        e2 = Entry(top, width=90, fg='blue', font=('Arial', 12, 'bold'))
                        e2.grid(row=i + 1, column=1)
                        e1.insert(END, x[2].split(" ")[1])
                        e2.insert(END, x[3][1:])
                        i += 1

                    temp = display_lines - len(line)

                    if temp > 0:
                        i = 0
                        temp2 = 2 + (len(line) * 2)
                        for widget in top.winfo_children():
                            i += 1
                            if i > temp2:
                                widget.destroy()
                    display_lines = len(line)
                    if display_lines == 0:
                        refresh2 = False

            top.after(500, update_status)

    update_status()
    ws.mainloop()


def main():
    global check_quit, video_capture, embedding_size, images_placeholder, phase_train_placeholder, sess, embeddings, \
        model, net, classNames, refresh, refresh2, TTS_name, queue_speech, queue_attendance_upload, queue_image_upload, \
        queueout_frame, uploading, queue_speech, speaking

    if not os.path.isfile("roomTimetable_new.xls"):
        classroom = input("Enter the classroom:")
        classroom = classroom.upper()
    else:
        classroom = check_timetable.getRoomName()

    fireb.getTimetable(classroom)
    fireb.getclassifier(classroom)

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            check_quit = False
            refresh = False
            refresh2 = False
            uploading = False
            speaking = False
            TTS_name = None

            print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            video_capture = cv2.VideoCapture(0)

            classNames = []
            with open(classFile, 'rt') as f:
                classNames = f.read().rstrip('\n').split('\n')
            net = cv2.dnn_DetectionModel(weightsPath, configPath)
            net.setInputSize(320, 320)
            net.setInputScale(1.0 / 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)

            print(f'Starting Recognition for class {classroom}')

            queueout_frame = queue.Queue()
            queue_speech = queue.Queue()
            queue_image_upload = queue.Queue()
            queue_recognition = queue.Queue()
            queue_attendance_upload = queue.Queue()
            threads = []

            t1 = threading.Thread(target=capture_frame, args=(queueout_frame, queue_recognition))
            threads.append(t1)
            t2 = threading.Thread(target=display_frame, args=(queueout_frame,))
            threads.append(t2)
            t3 = threading.Thread(target=display_attendance)
            threads.append(t3)

            for x in threads:
                x.start()

            for x in threads:
                x.join()


if __name__ == "__main__":
    main()
