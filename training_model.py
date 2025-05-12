import os
import face_recognition
import pickle
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def prepare_dataset(data_path):
    encodings = []
    names = []
    print("Preparing dataset...")
    for person in os.listdir(data_path):
        person_dir = os.path.join(data_path, person)
        if not os.path.isdir(person_dir):
            continue
            
        print(f" Processing {person}: ", end="")
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                encodings.append(face_encodings[0])
                names.append(person)
    return np.array(encodings), np.array(names)

def train_model(encodings, names):
    print(" Training model...")
    model = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric='cosine'
    )
    model.fit(encodings, names)
    return model

if __name__ == "__main__":

    DATA_PATH = r"C:\Users\LENOVO\Desktop\project AI\classi"
    MODEL_PATH = "classmates_model.pkl"

    X, y = prepare_dataset(DATA_PATH)
    model = train_model(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n💾 Model saved to {MODEL_PATH}")
    print(f"✨ Trained on {len(X)} samples ({len(set(y))} classmates)")