import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

# Define confidence threshold
confidence_threshold = 0.5

labels=[]
# Load COCO class labels
labels_path = 'coco.names'
with open(labels_path, 'rt') as f:
  labels = f.read().rstrip("\n").split("\n")

# Load the model
print(labels)

# prototxt = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weights = "MobileNetSSD_deploy.caffemodel"
prototxt = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weights, prototxt)
net.setInputSize(300, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# Start video capture
while True:
  success, img = cap.read()
  l_ids, confs, bbox = net.detect(img, confThreshold=confidence_threshold)

  if len(l_ids) != 0:
    for l_id, conf, box in zip(l_ids.flatten(), confs.flatten(), bbox):
      cv2.rectangle(img, box, (0, 255, 0), 2)
      cv2.putText(img, labels[l_id - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
      cv2.putText(img, str(round(conf * 100, 2)) + "%", (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

  # Capture frame-by-frame
  cv2.imshow("Output", img)
  # Exit with 'q' key press
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release resources
cap.release()
cv2.destroyAllWindows()