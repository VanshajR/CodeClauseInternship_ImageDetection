import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load COCO labels
labels_path = 'coco.names'
with open(labels_path, 'rt') as f:
    labels = f.read().rstrip("\n").split("\n")

# Load the DNN model
prototxt = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

# Ensure OpenCV headless compatibility
try:
    net = cv2.dnn_DetectionModel(weights, prototxt)
    net.setInputSize(300, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
except Exception as e:
    st.error(f"Error loading model: {str(e)}. Ensure {prototxt} and {weights} are available.")
    st.stop()

# Set Streamlit page config with dark theme
st.set_page_config(page_title="Object Detection", layout="wide")

# Custom dark theme CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #040e0e;
        color: white;
    }
    .stImage img {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Object Detection App")

st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.65, 0.05)  # Default set to 0.65

# Define a class for the webrtc video transformer
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")
        
        # Object detection
        detections = net.detect(img, confThreshold=confidence_threshold)
        l_ids, confs, bbox = detections if len(detections) == 3 else (None, None, None)

        if l_ids is not None:
            for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, (0, 255, 0), 2)
                label = f"{labels[l_id - 1]}: {conf * 100:.2f}%"
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

# Webcam mode using streamlit-webrtc
webrtc_streamer(
    key="object-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
