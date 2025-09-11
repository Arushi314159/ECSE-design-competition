import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import whisper

# Settings
fs = 44100       # Sample rate
silence_thresh = 500  # Adjust based on mic sensitivity
silence_duration = 2  # seconds of silence before stopping
filename = "detected_recording.wav"

def rms(data):
    """Root mean square (volume level) of audio chunk."""
    return np.sqrt(np.mean(np.square(data)))

print("Start speaking... Recording will stop after silence.")
frames = []
silence_start = None

with sd.InputStream(samplerate=fs, channels=1, dtype="int16") as stream:
    while True:
        data, _ = stream.read(1024)
        frames.append(data)
        
        # Check volume
        volume = rms(data)
        
        if volume < silence_thresh:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= silence_duration:
                print("Silence detected. Stopping recording.")
                break
        else:
            silence_start = None  # reset if voice resumes

# Save to file
audio = np.concatenate(frames, axis=0)
write(filename, fs, audio)
print(f"Saved recording to {filename}")

# load a model
model = whisper.load_model("small")

# transcribe an audio file
result = model.transcribe("detected_recording.wav")

print(result["text"])

