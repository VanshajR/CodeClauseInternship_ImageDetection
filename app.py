import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, ClientSettings
import numpy as np

# Load COCO labels
labels_path = 'coco.names'
with open(labels_path, 'rt') as f:
    labels = f.read().rstrip("\n").split("\n")

# Load the DNN model
prototxt = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

try:
    net = cv2.dnn_DetectionModel(weights, prototxt)
    net.setInputSize(300, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Define the WebRTC client settings for better webcam handling
client_settings = ClientSettings(
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)

# Define the video processor class for object detection
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.confidence_threshold = 0.65  # Default threshold

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection
        detections = net.detect(img, confThreshold=self.confidence_threshold)
        l_ids, confs, bbox = detections if len(detections) == 3 else (None, None, None)

        if l_ids is not None:
            for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, (0, 255, 0), 2)
                label = f"{labels[l_id - 1]}: {conf * 100:.2f}%"
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# Streamlit app setup
st.title("🔍 Object Detection App")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.65, 0.05)

# Choice for input source
choice = st.sidebar.radio(
    "Choose Input Source",
    ("Live Webcam Feed", "Upload Image", "Capture Image")
)

if choice == "Live Webcam Feed":
    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=client_settings,
        video_processor_factory=VideoProcessor,
    )

elif choice == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Perform object detection
        detections = net.detect(img, confThreshold=confidence_threshold)
        l_ids, confs, bbox = detections if len(detections) == 3 else (None, None, None)

        if l_ids is not None:
            for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, (0, 255, 0), 2)
                label = f"{labels[l_id - 1]}: {conf * 100:.2f}%"
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(img, channels="BGR")

elif choice == "Capture Image":
    img_file_buffer = st.camera_input("Capture an Image")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Perform object detection
        detections = net.detect(img, confThreshold=confidence_threshold)
        l_ids, confs, bbox = detections if len(detections) == 3 else (None, None, None)

        if l_ids is not None:
            for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, (0, 255, 0), 2)
                label = f"{labels[l_id - 1]}: {conf * 100:.2f}%"
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(img, channels="BGR")
