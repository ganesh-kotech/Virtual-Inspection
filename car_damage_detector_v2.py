import cv2
import math
import cvzone
import os
import sys
from datetime import datetime
from ultralytics import YOLO

# Set the current working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Get the image path from the command line arguments
image_path = sys.argv[1]

# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Load the image
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

# Extract the directory, original image name without extension, and extension
image_dir = os.path.dirname(image_path)
image_name, image_ext = os.path.splitext(os.path.basename(image_path))

# Get the current epoch timestamp
timestamp = int(datetime.now().timestamp())

# Define the output image name with the epoch timestamp
output_image_name = f"{image_name}_{timestamp}{image_ext}"

# Define the output image path in the same directory as the input image
output_image_path = os.path.join(image_dir, output_image_name)

# Save the image with detections
cv2.imwrite(output_image_path, img)
print('Success::', output_image_path,"::")

# # Display the image with detections
# cv2.imshow("Image", img)

# # Close window when 'q' button is pressed
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
# cv2.waitKey(1)