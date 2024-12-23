import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration

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
    st.error(f"Error loading model: {str(e)}. Ensure {prototxt} and {weights} are available.")
    st.stop()

st.set_page_config(page_title="Object Detection", layout="wide")

# Sidebar settings
st.sidebar.title("⚙️ Settings")
source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Image", "Capture Image"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)  # Default: 0.5

# Detection function
def detect_objects(image):
    # Detect objects
    detections = net.detect(image, confThreshold=confidence_threshold)
    if len(detections) == 3:
        l_ids, confs, bbox = detections
    else:
        return image  # Return the original image if no detections

    # Draw detections
    for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
        x, y, w, h = box
        label = f"{labels[l_id - 1]}: {conf * 100:.2f}%"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10 if y > 20 else y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Handle input sources
if source == "Webcam":
    st.info("Using Webcam for real-time object detection. Please allow access.")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_resized = cv2.resize(img, (640, 480))
            detected_image = detect_objects(img_resized)
            return cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]})
    webrtc_streamer(
        key="webcam-detection",
        video_processor_factory=VideoProcessor,
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

elif source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(uploaded_image)
        detected_image = detect_objects(image_array)
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_container_width=True)

elif source == "Capture Image":
    img_file_buffer = st.camera_input("Capture an Image")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        detected_image = detect_objects(img)
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_container_width=True)