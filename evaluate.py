import os
import face_recognition
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def load_test_data(data_path, test_size=0.3):
    encodings, names = [], []
    
    for person in os.listdir(data_path):
        person_dir = os.path.join(data_path, person)
        if not os.path.isdir(person_dir):
            continue
        all_images = [
            os.path.join(person_dir, img) 
            for img in os.listdir(person_dir)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if len(all_images) > 1: 
            train_imgs, test_imgs = train_test_split(
                all_images, 
                test_size=test_size, 
                random_state=42
            )
            
            # Only use test images for evaluation
            for img_path in test_imgs:
                image = face_recognition.load_image_file(img_path)
                face_enc = face_recognition.face_encodings(image)
                if face_enc:
                    encodings.append(face_enc[0])
                    names.append(person)
    
    return np.array(encodings), np.array(names)

def evaluate_model(model_path, data_path):
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test, y_test = load_test_data(data_path)
    y_pred = model.predict(X_test)
    
    print("\n Evaluation Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    accuracy = np.mean(y_test == y_pred)
    print(f" Test Accuracy: {accuracy:.1%}")
    print(f" Tested on {len(X_test)} samples ({len(set(y_test))} classmates)")

if __name__ == "__main__":
    evaluate_model(
        model_path="classmates_model.pkl",
        data_path=r"C:\Users\LENOVO\Desktop\project AI\classi"
    )