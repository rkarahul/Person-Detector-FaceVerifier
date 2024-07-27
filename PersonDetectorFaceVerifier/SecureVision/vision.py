from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import face_recognition
import os
# Load the YOLO model
model = YOLO("yolov8s-seg.pt")  # Adjust with your model path
# Print out the model's class names
print("Model class names:", model.names)
# Get the index of the 'person' class
person_class_index = [k for k, v in model.names.items() if v == 'person'][0]
print("Class index:", person_class_index)
# Read the image
rand_img = cv2.imread("person.jpg")
# Predict with the model
results = model.predict(source=rand_img, conf=0.45)  # Adjust other parameters as needed
# Initialize a variable to store the cropped person image
cropped_person = None

# Filter predictions to only include 'person'
for result in results:
    boxes = result.boxes
    for box in boxes:
        if box.cls.item() == person_class_index:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[box.cls.item()]
            confidence = box.conf.item()
            cv2.rectangle(rand_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rand_img, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Crop the detected person
            cropped_person = rand_img[y1:y2, x1:x2]

# Convert the image from BGR to RGB for Matplotlib
rand_img_rgb = cv2.cvtColor(rand_img, cv2.COLOR_BGR2RGB)

# Display the image with bounding box using Matplotlib
plt.imshow(rand_img_rgb)
# Display the cropped person image using Matplotlib
if cropped_person is not None:
    cropped_person_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
    # Save the cropped person image
    output_path = "cropped_person.jpg"
    cv2.imwrite(output_path, cropped_person)
    print(f"Cropped person image saved as {output_path}")

    # Load the input person image
    input_person_img = cv2.imread("farhan.jpg")
    if input_person_img is None:
        print("Error loading input person image.")
    else:
        # Convert images to RGB for face_recognition
        cropped_person_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
        input_person_rgb = cv2.cvtColor(input_person_img, cv2.COLOR_BGR2RGB)

        # Encode faces
        cropped_person_encoding = face_recognition.face_encodings(cropped_person_rgb)
        input_person_encoding = face_recognition.face_encodings(input_person_rgb)

        if len(cropped_person_encoding) > 0 and len(input_person_encoding) > 0:
            # Compare faces
            match = face_recognition.compare_faces([cropped_person_encoding[0]], input_person_encoding[0])

            if match[0]:
                print("Same person")
            else:
                print("Different person")
        else:
            print("Could not find faces in one or both images.")
else:
    print("No person detected.")
