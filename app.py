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

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
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
source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Image", "Capture Image"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.65, 0.05)  # Default set to 0.65


# Object detection function
def detect_objects(image):
    detections = net.detect(image, confThreshold=confidence_threshold)

    if len(detections) == 3:  # Ensure detections have three components
        l_ids, confs, bbox = detections
    else:
        l_ids, confs, bbox = None, None, None

    # Draw detections on the image
    if l_ids is not None:
        for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
            cv2.rectangle(image, box, (0, 255, 0), 2)
            label = f"{labels[l_id - 1]}: {conf * 100:.2f}%"
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        st.warning("No objects detected!")

    return image


# Streamlit WebRTC video processor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (640, 480))  # Resize for performance
        detected_image = detect_objects(img_resized)
        return cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)


if source == "Webcam":
    st.info("Webcam mode: Allow access to your camera.")
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
    )

    webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

elif source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(uploaded_image)
        detected_image = detect_objects(image_array)

        st.image(detected_image, caption="Detected Image", use_column_width=True)

elif source == "Capture Image":
    img_file_buffer = st.camera_input("Capture an image")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        detected_image = detect_objects(img)

        st.image(detected_image, caption="Captured Image", use_column_width=True)