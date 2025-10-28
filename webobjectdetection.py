import cv2
import streamlit as st
import numpy as np

st.set_page_config(page_title="Object Detection", page_icon="👀")

st.title("Object Detection")


uploaded_file = st.file_uploader("Image:", type=["jpg", "jpeg", "png"])

confThreshold = 0.5
configfile = "datafile.pbtxt"
model = "frozen_inference_graph.pb"
classLabels = "objects.names"

@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromTensorflow(model, configfile)
    return net

@st.cache_data
def load_labels():
    with open(classLabels, "rt") as fpt:
        return fpt.read().rstrip("\n").split("\n")

net = load_model()
classlabelsarray = load_labels()

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1.0/127.5, (320,320), (127.5,127.5,127.5), swapRB=True, crop=False)
    net.setInput(blob)
    prediction = net.forward()

    detected_classes = []
    st.header("Detected Objects")

    for i in range(prediction.shape[2]):
        confidence = prediction[0,0,i,2]
        if confidence > confThreshold:
            class_id = int(prediction[0,0,i,1])
            class_name = classlabelsarray[class_id-1]
            detected_classes.append(class_name)
            x = int(prediction[0,0,i,3]*width)
            y = int(prediction[0,0,i,4]*height)
            xbr = int(prediction[0,0,i,5]*width)
            ybr = int(prediction[0,0,i,6]*height)
            cv2.rectangle(image, (x,y), (xbr,ybr), (0,255,0), 2)
            cv2.putText(image, class_name, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2)
            confidence_percent = round(confidence * 100)
            st.subheader(f"{class_name} with {confidence_percent}% confidence")
        

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Detected Objects", use_container_width=True)


