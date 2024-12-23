import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
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

# Define the video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = 0.65  # Default threshold

    def recv(self, frame):
        # Convert frame to a NumPy array
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

# Streamlit app
st.title("🔍 Object Detection App")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.65, 0.05)

# Start the webcam
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    },
)
