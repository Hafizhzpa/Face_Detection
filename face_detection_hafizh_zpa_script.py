from ultralytics import YOLO
import cv2, torch
from PIL import Image

class FaceDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_face = YOLO(model_path).to(self.device)

    def process_image(self, input_path):
        # Read the image
        img = cv2.imread(input_path)
        # Predict faces
        output_face = self.model_face.predict(img)
        # Draw rectangles around detected faces
        for obj in output_face[0].boxes:
            x1, y1, x2, y2 = obj.xyxy[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Save the output image
        output_path = input_path.split('.')[0] + '_detected.jpg'
        cv2.imwrite(output_path, img)
        return output_path

    def process_video(self, input_path):
        # Read the video
        cap = cv2.VideoCapture(input_path)
        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(input_path.split('.')[0] + '_detected.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Predict faces
            output_face = self.model_face.predict(frame)
            # Draw rectangles around detected faces
            for obj in output_face[0].boxes:
                x1, y1, x2, y2 = obj.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Write the frame
            out.write(frame)
        # Release everything
        cap.release()
        out.release()
        return input_path.split('.')[0] + '_detected.mp4'

# Usage
model_path = 'yolov8n-face.pt'
detector = FaceDetector(model_path)
detector.process_image('input.jpg')  # For image files
detector.process_video('input.mp4')  # For video files
