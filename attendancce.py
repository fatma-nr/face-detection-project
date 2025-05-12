import os
import face_recognition
import pickle
import cv2
from datetime import datetime

def test_model(model_path="classmates_model.pkl"):
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f) 
    attendance_file = "attendance.csv"
    marked_students = set()

    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write("Date,Name,Time,Confidence\n")

    cap = cv2.VideoCapture(0)
    print(" Live Attendance System - Press Q to quit")
    print(f"• Saving to: {os.path.abspath(attendance_file)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locs = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locs)
        
        for (top, right, bottom, left), face_encoding in zip(face_locs, face_encodings):

            distances, indices = model.kneighbors([face_encoding])  # Fixed this line
            name = model.predict([face_encoding])[0]
            confidence = 1 - distances[0][0]

            current_time = datetime.now()
            if confidence > 0.80:  # Confidence threshold
                color = (0, 255, 0)  # Green
                if name not in marked_students:
                    with open(attendance_file, 'a') as f:
                        f.write(
                            f"{current_time.strftime('%Y-%m-%d')},"
                            f"{name},"
                            f"{current_time.strftime('%H:%M:%S')},"
                            f"{confidence:.2f}\n"
                        )
                    print(f"{current_time.strftime('%H:%M:%S')} - Marked {name} ({confidence:.2f})")
                    marked_students.add(name)
            else:
                color = (0, 0, 255)  # Red
                name = f"Unknown ({confidence:.2f})"
            
            # Draw UI
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left+6, bottom-6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n📅 Attendance Summary for", datetime.now().strftime('%Y-%m-%d'))
    print("Marked present:", ", ".join(marked_students) if marked_students else "No students marked")
    print(f"Full record saved to {attendance_file}")

if __name__ == "__main__":
    test_model()