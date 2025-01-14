import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration

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
st.sidebar.markdown(
    """
    As of the last update to this code, the live feed detection isn't working on this deployed application due to a defect in the streamlit library that allows for usage of the webcam for live feeds, however, it works when the app is run locally.
    
    Other funcationalities are operational.
    
    """
)
source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Image", "Capture Image"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.65, 0.05)  # Default set to 0.65

# Detection function
def detect_objects(image):
    try:
        # Detect objects
        detections = net.detect(image, confThreshold=confidence_threshold)

        # Check if detections contain three outputs
        if detections and len(detections) == 3:
            l_ids, confs, bbox = detections
        else:
            # If detections are invalid, return the original image without modifications
            st.warning("No valid detections were made.")
            return image

        # Process detections if valid
        if l_ids is not None and confs is not None and bbox is not None:
            for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
                cv2.rectangle(image, box, (0, 255, 0), 2)
                label = f"{labels[l_id - 1]}: {conf * 100:.2f}%"
                cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return image
    except Exception as e:
        # Log and display any detection errors
        st.warning(f"Error during object detection: {str(e)}")
        return image


# Create a WebRTC video processor for object detection
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        # Convert frame to numpy array and detect objects
        img = frame.to_ndarray(format="bgr24")

        # Resize the frame for better performance (resize to 640x480)
        img_resized = cv2.resize(img, (640, 480))

        # Perform object detection on the resized image
        detected_image = detect_objects(img_resized)

        # Convert the image to RGB format for correct display
        detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        return detected_image_rgb

# Webcam mode using WebRTC
if source == "Webcam":
    st.info("Webcam is running using WebRTC. Please allow camera access.")
    
    # WebRTC configuration for video only (disabling audio)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}  # STUN server
    )

    # Make sure audio is completely disabled and only video is used
    webrtc_streamer(
        key="example", 
        video_processor_factory=VideoProcessor,  # Using the new video processor
        mode=WebRtcMode.SENDONLY,  # Send only video, no audio
        rtc_configuration=rtc_configuration,  # Pass RTC configuration with no audio
        media_stream_constraints={"video": True, "audio": False}  # Explicitly disable audio
    )

# Image upload mode
elif source == "Upload Image":
    # Stop webcam if switched to upload mode
    if 'webcam_running' in st.session_state and st.session_state.webcam_running:
        st.session_state.webcam_running = False

    uploaded_file = st.file_uploader("📂 Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
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

            # Display results
            st.image(detected_image_resized, caption="Detected Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error processing the uploaded image: {str(e)}")

# Capture Image mode
elif source == "Capture Image":
    # Stop webcam if switched to capture mode
    if 'webcam_running' in st.session_state and st.session_state.webcam_running:
        st.session_state.webcam_running = False

    # Use Streamlit's camera input for capturing an image
    img_file_buffer = st.camera_input("📸Capture an Image")
    if img_file_buffer is not None:
        try:
            bytes_data = img_file_buffer.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # Perform object detection
            detected_image = detect_objects(img)

            # Convert BGR to RGB for proper display in Streamlit
            detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

            # Resize the image to fit the screen width without scrolling
            detected_image_resized = cv2.resize(detected_image_rgb, (640, 480))

            # Display results
            st.image(detected_image_resized, caption="Captured & Detected Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error processing the captured image: {str(e)}")
