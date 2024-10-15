import cv2
import numpy as np

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    erode = cv2.erode(blur, np.ones((3, 3)))
    dilated = cv2.dilate(erode, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closing

# Function to detect cars in a frame
def detect_cars(frame):
    processed = preprocess_image(frame)
    
    # Use a pre-trained car cascade classifier
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
    
    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(processed, 1.1, 1)
    
    return cars

# Function to calculate car density
def calculate_car_density(frame, cars):
    frame_area = frame.shape[0] * frame.shape[1]
    car_area = sum([w*h for (x,y,w,h) in cars])
    density = car_area / frame_area
    return density

# Function to determine traffic light timing
def determine_light_timing(density):
    if density < 0.1:
        return 30  # Low density, short green light duration
    elif density < 0.3:
        return 60  # Medium density, medium green light duration
    else:
        return 90  # High density, long green light duration

# Function to check if road is congested
def is_congested(density):
    return density > 0.5  # Consider the road congested if more than 50% is covered by cars

# Main function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_traffic_analysis.mp4', fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cars = detect_cars(frame)
        density = calculate_car_density(frame, cars)
        green_light_duration = determine_light_timing(density)
        congestion_status = is_congested(density)
        
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Create a semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 120), (0, 0, 0), -1)
        
        # Add the overlay to the original image
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add text with improved visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Car Density: {density:.2f}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Green Light Duration: {green_light_duration}s", (10, 70), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Congested: {'Yes' if congestion_status else 'No'}", (10, 110), font, 0.7, (255, 255, 255), 2)
        
        # Write the frame to the output video
        out.write(frame)
        
        # Display the frame (optional, for real-time viewing)
        cv2.imshow('Traffic Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'video.mp4'
process_video(video_path)