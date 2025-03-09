import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    lane_frame = detect_lanes(image)
    
    try:
        cv2.imshow("Lane Detection", lane_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("OpenCV GUI not supported. Using Matplotlib instead.")
        plt.imshow(cv2.cvtColor(lane_frame, cv2.COLOR_BGR2RGB))
        plt.title("Lane Detection")
        plt.show()

def process_video():
    cap = cv2.VideoCapture(0)  # Use webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Starting live video lane detection... Press 'Q' to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        lane_frame = detect_lanes(frame)
        
        try:
            cv2.imshow("Lane Detection (Press Q to Exit)", lane_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            print("OpenCV GUI not supported. Using Matplotlib instead.")
            plt.imshow(cv2.cvtColor(lane_frame, cv2.COLOR_BGR2RGB))
            plt.title("Lane Detection")
            plt.show()
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Choose mode - (1) Image or (2) Live Video: ")
    if mode == "1":
        image_path = input("Enter image path (e.g., test_images/image.jpg): ")
        process_image(image_path)
    elif mode == "2":
        process_video()
    else:
        print("Invalid option! Choose 1 for image or 2 for live video.")