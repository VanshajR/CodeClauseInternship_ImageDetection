import cv2
import customtkinter as cust
from PIL import Image, ImageTk
from customtkinter import CTkImage

cust.set_appearance_mode("dark")
# cust.set_default_color_theme("green")

class ObjectDetectionApp(cust.CTk):
    def __init__(self, title, width, height):
        super().__init__()

        # Initialize the main window
        self.title(title)
        self.geometry(f"{width}x{height}")

        # Setup the video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # Load COCO class labels
        self.labels = []
        labels_path = 'coco.names'
        with open(labels_path, 'rt') as f:
            self.labels = f.read().rstrip("\n").split("\n")

        # Load the model
        prototxt = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weights = 'frozen_inference_graph.pb'
        self.net = cv2.dnn_DetectionModel(weights, prototxt)
        self.net.setInputSize(300, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Define confidence threshold
        self.confidence_threshold = 0.5

        # GUI elements
        self.control_frame = cust.CTkFrame(self)
        self.control_frame.pack(pady=20)

        self.start_button = cust.CTkButton(self.control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10, padx=20, side='left')

        self.stop_button = cust.CTkButton(self.control_frame, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=10, padx=20, side='left')

        self.video_frame = cust.CTkFrame(self)
        self.video_frame.pack(pady=20)

        self.video_label = cust.CTkLabel(self.video_frame,text="No Video Feed", font=("Helvetica", 16))
        self.video_label.pack()

        self.running = False
        self.update_frame()

    def start_detection(self):
        self.running = True
        self.video_label.pack_forget()
        self.geometry("700x700")

    def stop_detection(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                l_ids, confs, bbox = self.net.detect(frame, confThreshold=self.confidence_threshold)
                if len(l_ids) != 0:
                    for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
                        cv2.rectangle(frame, box, (0, 255, 0), 2)
                        cv2.putText(frame, self.labels[l_id - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, str(round(conf * 100, 2)) + "%", (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                # Convert the image to RGB
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                ctk_img = CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.video_label.configure(image=ctk_img)
                self.video_label.pack()

        self.after(10, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

# Main Function
if __name__ == "__main__":
    app = ObjectDetectionApp("Object Detection App", 400, 400)
    app.mainloop()
