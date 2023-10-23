import streamlit as st
import joblib
import torch
import torchaudio
import torchaudio.transforms as transforms

# Load the pre-trained ResNet-18 model saved as a .pkl file
model = joblib.load('model_CNN.pkl')

# Define audio preprocessing function to create a spectrogram
def preprocess_audio(audio):
    # Load the audio file and convert it to a spectrogram
    waveform, sample_rate = torchaudio.load(audio)
    
    # Apply audio transformations to create a spectrogram
    transform = transforms.MelSpectrogram(sample_rate=sample_rate)
    spectrogram = transform(waveform)
    
    # Expand the spectrogram to add a batch dimension
    spectrogram = spectrogram.unsqueeze(0)
    
    return spectrogram

# Define a function to post-process the model's output
def post_process_transcription(outputs):
    # Replace this logic with your specific post-processing
    # For this example, let's assume the model outputs a label index.
    labels = ['hapana',
                  'kumi',
                  'mbili',
                  'moja',
                  'nane',
                  'ndio',
                  'nne',
                  'saba',
                  'sita',
                  'tano',
                  'tatu',
                  'tisa'] #label names
    predicted_index = torch.argmax(outputs, dim=1).item()
    transcription = labels[predicted_index]
    
    return transcription

# Create a Streamlit web app
st.title("Swahili Audio Transcription")

# Create file uploader
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])  # Adjust the accepted audio formats as needed

if audio_file is not None:
    spectrogram = preprocess_audio(audio_file)

    if st.button("Transcribe"):
        # Use the ResNet-18 model for transcription
        model.eval()
        with torch.no_grad():
            outputs = model(spectrogram)
            transcription = post_process_transcription(outputs)  # Define 'post_process_transcription' function earlier

        st.write(f"Transcription: {transcription}")
