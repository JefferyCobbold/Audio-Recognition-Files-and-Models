import os
import librosa
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from micromlgen import port 

# Function to extract features from audio files with padding/truncation
def extract_features(audio_path, max_frames=300):
    audio, _ = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)
    
    # Pad or truncate to the specified maximum number of frames
    if mfccs.shape[1] < max_frames:
        pad_width = max_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_frames]
    
    return mfccs.flatten()

# Load and preprocess data
data_path = "/media/jeffery/New Volume/audio/new"
labels = []
features = []

for label in os.listdir(data_path):
    label_path = os.path.join(data_path, label)
    for file in os.listdir(label_path):
        file_path = os.path.join(label_path, file)
        feature = extract_features(file_path)
        features.append(feature)
        labels.append(label)

X = np.array(features)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize label encoder
label_encoder = LabelEncoder()

# Fit and transform labels for training data
y_train_encoded = label_encoder.fit_transform(y_train)

# Transform labels for test data
y_test_encoded = label_encoder.transform(y_test)

# Define the deep learning model
model = Sequential([
    Dense(128, activation='relu', input_shape=(300 * 13,)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=150, batch_size=32, validation_data=(X_test, y_test_encoded))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {test_acc}")

# Save the model
model.save('Sound_Classification1.h5')

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_converted_model = converter.convert()
with open('Sound_Classifier.tflite1', 'wb') as f:
    f.write(tflite_converted_model)

if __name__=='__main__':
    model
    c_code=port(model,variable_name='Sound_Classifier1',
                pretty_print=True)
    print(c_code)