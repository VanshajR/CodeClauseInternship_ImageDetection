import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Load COCO labels
labels_path = 'coco.names'
with open(labels_path, 'rt') as f:
    labels = f.read().rstrip("\n").split("\n")

# Load the DNN model (adjust paths if needed)
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
    .st-au {
        background-color: #262730;  
        color: white;  
    }
    .css-ffhzg2 {
        max-width: 100% !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Object Detection App")

st.sidebar.title("⚙️ Settings")
source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Image", "Capture Image"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.65, 0.05)  # Default set to 0.65

# Detection function
def detect_objects(image):
    detections = net.detect(image, confThreshold=confidence_threshold)
    l_ids, confs, bbox = detections if len(detections) == 3 else (None, None, None)

    if l_ids is not None:
        for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
            cv2.rectangle(image, box, (0, 255, 0), 2)
            label = f"{labels[l_id - 1]}: {conf * 100:.2f}%"
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Webcam mode
if source == "Webcam":
    webcam_button = st.button("📹 Toggle Webcam")
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if webcam_button:
        st.session_state.webcam_running = not st.session_state.webcam_running

    frame_placeholder = st.empty()

    if st.session_state.webcam_running:
        st.info("Webcam is running. Click 'Toggle Webcam' to stop.")
        cap = cv2.VideoCapture(0)
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            # Object detection
            frame = detect_objects(frame)

            # Convert the frame from BGR to RGB (for Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to fit the screen without scrolling (Fixed window size)
            frame_resized = cv2.resize(frame_rgb, (640, 480))  # Adjust the size as needed
            frame_placeholder.image(frame_resized, use_column_width=False)

        cap.release()
        frame_placeholder.empty()
    else:
        st.warning("Webcam is stopped. Click 'Toggle Webcam' to start.")

# Image upload mode
elif source == "Upload Image":
    # Stop webcam if switched to upload mode
    if 'webcam_running' in st.session_state and st.session_state.webcam_running:
        st.session_state.webcam_running = False

    uploaded_file = st.file_uploader("📂 Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open the uploaded image using PIL
        uploaded_image = Image.open(uploaded_file)

        # Ensure image is in RGB mode (removes alpha channel if exists)
        uploaded_image = uploaded_image.convert("RGB")

        # Convert the PIL image to a NumPy array (this will preserve RGB format)
        image_array = np.array(uploaded_image)

        # Object detection
        detected_image = detect_objects(image_array)

        # Resize the image to fit the screen width without scrolling
        detected_image_resized = cv2.resize(detected_image, (640, 480))

        # Display results without changing colors
        st.image(detected_image_resized, caption="Detected Image", use_column_width=False)

# Capture Image mode
elif source == "Capture Image":
    # Stop webcam if switched to capture mode
    if 'webcam_running' in st.session_state and st.session_state.webcam_running:
        st.session_state.webcam_running = False

    # Use Streamlit's camera input for capturing an image
    img_file_buffer = st.camera_input("📸 Capture an Image")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Perform object detection
        detected_image = detect_objects(img)

        # Resize the image to fit the screen width without scrolling
        detected_image_resized = cv2.resize(detected_image, (640, 480))

        # Display results without changing colors
        st.image(detected_image_resized, caption="Captured & Detected Image", use_column_width=False)


