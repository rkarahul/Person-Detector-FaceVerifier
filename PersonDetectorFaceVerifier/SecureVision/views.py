from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import cv2
import matplotlib.pyplot as plt
import face_recognition
from ultralytics import YOLO
import os

def vision(request):
    return render(request, 'vision.html')

def upload_image(request):
    if request.method == 'POST':
        person1 = request.FILES['person1']
        person2 = request.FILES['person2']
        print("person1",person1)
        print("person2",person2)

        fs = FileSystemStorage()
        person1_name = fs.save(person1.name, person1)
        
        person2_name = fs.save(person2.name, person2)
        person1_path = fs.path(person1_name)
        person2_path = fs.path(person2_name)

        # Load the YOLO model
        model = YOLO("yolov8s-seg.pt")  # Adjust with your model path
        # Print out the model's class names
        print("Model class names:", model.names)
        # Get the index of the 'person' class
        person_class_index = [k for k, v in model.names.items() if v == 'person'][0]
        print("Class index:", person_class_index)

        # Initialize variables to store the cropped person images
        cropped_person1 = None
        cropped_person2 = None

        # Function to detect and crop person
        def detect_and_crop_person(image_path):
            img = cv2.imread(image_path)
            results = model.predict(source=img, conf=0.45)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.cls.item() == person_class_index:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_img = img[y1:y2, x1:x2]
                        return cropped_img
            return None

        # Detect and crop persons from both images
        cropped_person1 = detect_and_crop_person(person1_path)
        cropped_person2 = detect_and_crop_person(person2_path)

        if cropped_person1 is not None and cropped_person2 is not None:
            # Save the cropped person images
            cropped_person1_path = os.path.join(settings.MEDIA_ROOT, 'cropped_' + person1.name)
            cropped_person2_path = os.path.join(settings.MEDIA_ROOT, 'cropped_' + person2.name)
            cv2.imwrite(cropped_person1_path, cropped_person1)
            cv2.imwrite(cropped_person2_path, cropped_person2)

            # Load the input person images
            input_person1_img = cv2.imread(cropped_person1_path)
            input_person2_img = cv2.imread(cropped_person2_path)

            if input_person1_img is None or input_person2_img is None:
                print("Error loading input person images.")
            else:
                # Convert images to RGB for face_recognition
                input_person1_rgb = cv2.cvtColor(input_person1_img, cv2.COLOR_BGR2RGB)
                input_person2_rgb = cv2.cvtColor(input_person2_img, cv2.COLOR_BGR2RGB)

                # Encode faces
                person1_encoding = face_recognition.face_encodings(input_person1_rgb)
                person2_encoding = face_recognition.face_encodings(input_person2_rgb)

                if len(person1_encoding) > 0 and len(person2_encoding) > 0:
                    # Compare faces
                    match = face_recognition.compare_faces([person1_encoding[0]], person2_encoding[0])

                    if match[0]:
                        comparison_result = "Same person"
                    else:
                        comparison_result = "Different person"
                else:
                    comparison_result = "Could not find faces in one or both images."
        else:
            comparison_result = "No person detected in one or both images."

        context = {
            'person1_url': fs.url(person1_name),
            'person2_url': fs.url(person2_name),
            'cropped_person1_url': fs.url('cropped_' + person1.name) if cropped_person1 is not None else None,
            'cropped_person2_url': fs.url('cropped_' + person2.name) if cropped_person2 is not None else None,
            'comparison_result': comparison_result,
        }

        return render(request, 'vision.html', context)

    return render(request, 'vision.html')
