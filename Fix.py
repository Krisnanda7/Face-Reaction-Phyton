import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# Load the model
try:
    model = keras.models.load_model('reaksiwajah_2.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    
    exit()

# definisi emosi
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  

# inisialisasi kamera
cap = cv2.VideoCapture(0)  # 0 usually represents the default camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    
    # inisialisasi kamera
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
     # merubah frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale image to RGB by repeating the single channel
    rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # menyiapkan gambar untuk di prediksi
    resized_frame = cv2.resize(rgb_frame, (48, 48))  # Sesuaikan ukuran dengan input model
    normalized_frame = resized_frame / 255.0
    reshaped_frame = normalized_frame.reshape(1, 48, 48, 3)  # Sesuaikan dengan input model (3 channel RGB)
    
    
    # prediksi emosi wajah
    prediction = model.predict(reshaped_frame, verbose=0)  # untuk mencegah log dari model.predict
    predicted_class = np.argmax(prediction)
    predicted_emotion = emotions[predicted_class]
    
    

    # Display the predicted emotion on the frame
    cv2.putText(frame, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # menampilkan persentase dari hasil prediksi 
    y_offset = 60
    for i, emotion in enumerate(emotions):
        probability = prediction[0][i] * 100
        text = f"{emotion}: {probability:.2f}%"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset += 30

    # Display the resulting frame
    cv2.imshow('Realtime Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()