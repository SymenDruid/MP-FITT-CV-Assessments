import cv2
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk

LABEL_WIDTH = 600
LABEL_HEIGHT = 338

# Load the pre-trained face detector model from OpenCV
prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNet(prototxt, model)

def detect_faces(frame):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    return faces


def update_live_video(label):

    ret, frame = cap.read()
    if not ret:
        return

    faces = detect_faces(frame)
    for (startX, startY, endX, endY) in faces:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2image)
    resized_image = pil_image.resize((LABEL_WIDTH, LABEL_HEIGHT), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=resized_image)

    label.config(image=imgtk)
    label.image = imgtk
    label.after(10, update_live_video, label)  # Update continuously


def detect_faces_from_image(label):

    filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not filename:
        return

    image = cv2.imread(filename)
    faces = detect_faces(image)
    for (startX, startY, endX, endY) in faces:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2image)
    resized_image = pil_image.resize((LABEL_WIDTH, LABEL_HEIGHT), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=resized_image)

    label.config(image=imgtk)
    label.image = imgtk


def show_placeholder(label):

    image = cv2.imread('placeholder.jpg')
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2image)
    resized_image = pil_image.resize((LABEL_WIDTH, LABEL_HEIGHT), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=resized_image)
    label.config(image=imgtk)
    label.image = imgtk


def main():
    global cap

    # window
    window = ttk.Window(themename = 'darkly')
    window.title('Face Detection App')
    window.geometry('1200x550')
    window.minsize(1200,500)

    main_frame = ttk.Frame(window)
    main_frame.pack()


    main_frame.columnconfigure((0,1), weight = 1)
    main_frame.rowconfigure((0,1,2,3), weight = 1)

    Heading = ttk.Label(main_frame, text = "Face Detection", bootstyle='inverse-dark', padding=15)
    Heading.grid(row = 0, column = 0, columnspan = 2, padx = 4, pady = 10)

    LHeading = ttk.Label(main_frame, text = "Live Camera Feed", bootstyle='inverse', padding=8)
    LHeading.grid(row = 1, column = 0, padx = 4, pady = 4)

    RHeading = ttk.Label(main_frame, text = "Image Selection", bootstyle='inverse', padding=8)
    RHeading.grid(row = 1, column = 1, columnspan = 1, padx = 4, pady = 4)


    live_video_label = tk.Label(main_frame)
    live_video_label.grid(row=2, column=0, padx = 4, pady = 4)

    image_label = tk.Label(main_frame)
    image_label.grid(row=2, column=1, padx = 4, pady = 4)

    select_image_button = ttk.Button(main_frame, text="Select Image", command=lambda: detect_faces_from_image(image_label))
    select_image_button.grid(row = 3, column = 1, sticky = 'we', columnspan = 1, padx = 4, pady = 4)


    cap = cv2.VideoCapture(0)

    update_live_video(live_video_label)

    show_placeholder(image_label)

    window.mainloop()


if __name__ == "__main__":
    main()
