import cv2

# Initialize video capture object for the default camera (index 0)
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the captured frame
    cv2.imshow('Camera Feed', frame)

    # Wait for a key press (1 millisecond) and break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()