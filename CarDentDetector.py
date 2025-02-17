import cv2
import math
import cvzone
import os
from ultralytics import YOLO

# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Load the image
image_path = "Media/dent1.jpg"
img = cv2.imread(image_path)

# Perform object detection
results = yolo_model(img)

# Loop through the detections and draw bounding boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        
        conf = math.ceil((box.conf[0] * 100)) / 100

        if conf > 0.3:
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'Damage {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

# Ensure the 'outputs' directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Extract the original image name without extension
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Define the output image path
output_image_path = os.path.join(output_dir, f"{image_name}_output.jpg")

# Save the image with detections
cv2.imwrite(output_image_path, img)

# Display the image with detections
cv2.imshow("Image", img)

# Close window when 'q' button is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
