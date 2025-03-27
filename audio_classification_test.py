import tensorflow as tf
import numpy as np
import librosa
from tinymlgen import port
from keras.models import load_model

# Load Keras model
model1 = load_model('/media/jeffery/New Volume/Sound_Classification1.h5')

# Load TensorFlow Lite model
model = tf.lite.Interpreter(model_path="/media/jeffery/New Volume/Sound_Classifier.tflite")
model.allocate_tensors()

# Get input and output details
input_details = model.get_input_details()
output_details = model.get_output_details()

# Function to preprocess input data for inference
def preprocess_input(audio_path, max_frames=300):
    audio, _ = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)
    
    # Pad or truncate to the specified maximum number of frames
    if mfccs.shape[1] < max_frames:
        pad_width = max_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_frames]
    
    return mfccs.flatten()

# Example: Test the TensorFlow Lite model with a sample audio file
sample_audio_path = "/media/jeffery/New Volume/audio/clean/F_20200222133626_1_T22.7.wav"
input_data = preprocess_input(sample_audio_path)

# Resize the input tensor to match the model's expected input shape
model.set_tensor(input_details[0]['index'], input_data.reshape(1, -1))

# Run inference
model.invoke()

# Get the output tensor
output_data = model.get_tensor(output_details[0]['index'])

# Post-process the output, e.g., get predicted class
predicted_class = np.argmax(output_data)

# Print the predicted class
print(f"Predicted Class: {predicted_class}")
if predicted_class==1:
    print('Infected')
else: 
 print("Clean")
if __name__=='__main__':
    model1
    c_code=port(model1,variable_name='Sound_classification',
                pretty_print=True)
    print(c_code)

