import streamlit as st
import cv2
import numpy as np
from PIL import Image

CONF_THRESHOLD = 0.55
CONFIG_FILE = "datafile.pbtxt"
MODEL_FILE = "frozen_inference_graph.pb"
CLASS_FILE = "objects.names"

@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromTensorflow(MODEL_FILE, CONFIG_FILE)
    with open(CLASS_FILE, "rt") as f:
        class_labels = f.read().rstrip("\n").split("\n")
    return net, class_labels

net, class_labels = load_model()
st.title("Object Detection App")

mode = st.sidebar.radio("Choose mode", ["Image Upload", "Webcam Detection"])

if mode == "Image Upload":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        height, width, _ = image_np.shape
        blob = cv2.dnn.blobFromImage(image_np, 1.0/127.5, (320, 320), (127.5,127.5,127.5), swapRB=True, crop=False)
        net.setInput(blob)
        prediction = net.forward()
        for i in range(prediction.shape[2]):
            confidence = prediction[0,0,i,2]
            if confidence > CONF_THRESHOLD:
                class_id = int(prediction[0,0,i,1])
                class_name = class_labels[class_id - 1]
                x = int(prediction[0,0,i,3]*width)
                y = int(prediction[0,0,i,4]*height)
                xbr = int(prediction[0,0,i,5]*width)
                ybr = int(prediction[0,0,i,6]*height)
                cv2.rectangle(image_np, (x, y), (xbr, ybr), (0,255,0), 2)
                cv2.putText(image_np, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                st.subheader(f"Detected: **{class_name}** with **{confidence*100}%** confidence ")
        st.image(image_np, caption="Detected Objects", channels="BGR")

elif mode == "Webcam Detection":
    st.write("Press **Start** to begin webcam object detection.")
    start_btn = st.button("Start Detection")
    if start_btn:
        cap = st.VideoCapture()
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Unable to access webcam.")
                break
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1.0/127.5, (320,320),(127.5,127.5,127.5), swapRB=True, crop=False)
            net.setInput(blob)
            prediction = net.forward()
            for i in range(prediction.shape[2]):
                confidence = prediction[0,0,i,2]
                if confidence > CONF_THRESHOLD:
                    class_id = int(prediction[0,0,i,1])
                    class_name = class_labels[class_id - 1]
                    x = int(prediction[0,0,i,3]*width)
                    y = int(prediction[0,0,i,4]*height)
                    xbr = int(prediction[0,0,i,5]*width)
                    ybr = int(prediction[0,0,i,6]*height)
                    cv2.rectangle(frame, (x,y), (xbr,ybr), (0,255,0), 2)
                    cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            stframe.image(frame, channels="BGR")
        cap.release()

