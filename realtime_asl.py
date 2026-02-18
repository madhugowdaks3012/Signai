import cv2
import numpy as np
from tensorflow.keras.models import load_model
import string

# ===============================
# Load Model
# ===============================
model = load_model("asl_model.h5")

# ===============================
# Labels (A-Z + SPACE)
# ===============================
labels = list(string.ascii_uppercase)
labels.append("SPACE")

# ===============================
# Word storage
# ===============================
current_word = ""
last_prediction = ""
cooldown = 0

# ===============================
# Webcam
# ===============================
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")
print("Press 'c' to clear word")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI (Region of Interest)
    x1, y1 = 100, 100
    x2, y2 = 400, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, 3))

    # Prediction
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_letter = labels[class_index]

    # Add to word (only if confidence high)
    if confidence > 0.85:
        if predicted_letter != last_prediction:
            cooldown += 1

            if cooldown > 15:
                if predicted_letter == "SPACE":
                    current_word += " "
                else:
                    current_word += predicted_letter

                last_prediction = predicted_letter
                cooldown = 0
    else:
        cooldown = 0

    # Display
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"Letter: {predicted_letter}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Word: {current_word}", (50,450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("ASL Detector", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    if key == ord('c'):
        current_word = ""

cap.release()
cv2.destroyAllWindows()
