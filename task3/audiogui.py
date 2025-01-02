import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import keras
from keras.models import model_from_json
import sounddevice as sd
import scipy.io.wavfile as wav

# Load model
json_file = open('model_config.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model_weights.weights.h5')

# Emotion labels
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'bored']

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Padding or truncating to ensure consistent size
    target_size = 162
    if mfccs.shape[1] < target_size:
        pad_width = target_size - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :target_size]
    
    # Transpose and reshape to match model input (162, 1)
    mfccs = np.mean(mfccs.T, axis=1, keepdims=True)  # Shape: (162, 1)
    mfccs = np.expand_dims(mfccs, axis=0)             # Batch dimension (1, 162, 1)
    
    return mfccs

def predict_emotion(file_path):
    audio_features = preprocess_audio(file_path)
    predictions = model.predict(audio_features)
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    return predicted_emotion

def record_voice():
    fs = 16000
    seconds = 4
    messagebox.showinfo('Recording', 'Recording will start now. Speak for 4 seconds.')
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    wav.write('recorded_audio.wav', fs, audio)
    try:
        emotion = predict_emotion('recorded_audio.wav')
        messagebox.showinfo('Emotion Detected', f'Detected Emotion: {emotion}')
    except Exception as e:
        messagebox.showerror('Error', f'Failed to process audio: {str(e)}')

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        try:
            emotion = predict_emotion(file_path)
            messagebox.showinfo('Emotion Detected', f'Detected Emotion: {emotion}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to process audio: {str(e)}')

root = tk.Tk()
root.title('Emotion Detection through Voice')
root.geometry('400x300')

label = tk.Label(root, text='Emotion Detection (Female Voice Only)', font=('Helvetica', 14))
label.pack(pady=20)

record_button = tk.Button(root, text='Record Voice', command=record_voice)
record_button.pack(pady=10)

upload_button = tk.Button(root, text='Upload Voice Note', command=upload_file)
upload_button.pack(pady=10)

exit_button = tk.Button(root, text='Exit', command=root.destroy)
exit_button.pack(pady=20)

root.mainloop()
