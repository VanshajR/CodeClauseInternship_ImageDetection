import cv2
import customtkinter as cust
from PIL import Image, ImageTk
from customtkinter import CTkImage
from tkinter import filedialog

cust.set_appearance_mode("dark")

class ObjectDetectionApp(cust.CTk):
    def __init__(self, title, width, height):
        super().__init__()

        self.title(title)
        self.geometry(f"{width}x{height}")
  
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.labels = []
        labels_path = 'coco.names'
        with open(labels_path, 'rt') as f:
            self.labels = f.read().rstrip("\n").split("\n")

        prototxt = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weights = 'frozen_inference_graph.pb'
        self.net = cv2.dnn_DetectionModel(weights, prototxt)
        self.net.setInputSize(300, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.confidence_threshold = 0.5

        self.control_frame = cust.CTkFrame(self)
        self.control_frame.pack(pady=20)

        self.start_button = cust.CTkButton(self.control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10, padx=20, side='left')

        self.stop_button = cust.CTkButton(self.control_frame, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=10, padx=20, side='left')

        self.upload_button = cust.CTkButton(self.control_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10, padx=20, side='left')

        self.video_frame = cust.CTkFrame(self)
        self.video_frame.pack(pady=20)

        self.video_label = cust.CTkLabel(self.video_frame, text="No Video/Image Feed", font=("Helvetica", 16))
        self.video_label.pack()

        self.placeholder_img = Image.new('RGB', (640, 480), color='gray')
        self.ctk_placeholder_img = CTkImage(light_image=self.placeholder_img, dark_image=self.placeholder_img, size=(640, 480))
        self.video_label.configure(image=self.ctk_placeholder_img)

        self.running = False
        self.update_frame()

    def start_detection(self):
        if not self.cap.isOpened():
            self.cap.open(0)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
        self.running = True
        self.geometry("700x700")
        self.video_label.configure(image=self.ctk_placeholder_img, text="")

    def stop_detection(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.video_label.configure(image=self.ctk_placeholder_img, text="No Video Feed, Detecting again takes a few seconds....")
        self.geometry("400x400")

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            self.process_image(img)

    def process_image(self, img):
        l_ids, confs, bbox = self.net.detect(img, confThreshold=self.confidence_threshold)
        if len(l_ids) != 0:
            for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, (0, 255, 0), 2)
                cv2.putText(img, self.labels[l_id - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(conf * 100, 2)) + "%", (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2image)
        ctk_img = CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 480))
        self.video_label.configure(image=ctk_img, text="")

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

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                ctk_img = CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.video_label.configure(image=ctk_img, text="")

        self.after(10, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ObjectDetectionApp("Object Detection App", 700, 500)
    app.mainloop()
