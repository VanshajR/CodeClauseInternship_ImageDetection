# Object Detection Application

This repository contains an object detection application built using OpenCV and TensorFlow's pre-trained SSD MobileNet model. The app allows users to detect objects in real-time using a webcam feed or to upload an image for detection.
It also includes a streamlit app that does the same.

## Features

- **Real-time Object Detection**: Uses a webcam feed for live object detection.
- **Image Upload Functionality**: Detects objects in uploaded images.
- **Customizable Confidence Threshold**: Set the confidence level for object detection.
- **Dark Mode GUI**: Built with a dark-themed GUI using `customtkinter`.
- **Streamlit UI**: Built a deployable web based GUI using `streamlit`.

## File Overview

- `main.py`: Contains the source code for the local GUI-based object detection application.
- `app.py`: Contains the code for the Streamlit UI object detection application.
- `coco.names`: Contains the list of object class labels used by the model.
- `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`: Configuration file for the SSD MobileNet model.
- `frozen_inference_graph.pb`: Pre-trained model file for SSD MobileNet.
- `requirements.txt`: Lists Python dependencies for the application.
- `LICENSE`: License information for the project.

## Prerequisites

- Python 3.8 or above.
- A working webcam for real-time detection (optional if using the image upload feature).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VanshajR/Object-Detection.git
   cd Object-Detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Run Locally**:
   ```bash
   python main.py
   ```
   Or, to run the streamlit app:
   
   ```bash
   streamlit run app.py
   ```

3. **Usage**:
   - Click "Start Detection" to begin real-time object detection.
   - Use "Upload Image" to upload an image and detect objects in it.
   - Click "Stop Detection" to stop the live webcam feed.

## Notes

- Ensure the `coco.names`, `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`, and `frozen_inference_graph.pb` files are in the same directory as `app.py`.
- For real-time detection, a webcam should be connected and accessible.

## License

This project is licensed under the [MIT License](LICENSE).
