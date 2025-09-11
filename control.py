import serial, serial.tools.list_ports
import asyncio, dotenv
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import whisper

from openai import OpenAI

import pasimple

dotenv.load_dotenv()

client = OpenAI()

# Get all available serial ports
ports = serial.tools.list_ports.comports()
print("Available serial ports:")
for port in ports:
    print(" -", port.device, port.description)

TARGET = "/dev/ttyUSB0"

# Connect to the target serial port
SER = None

def record():
    fs = 44100       # Sample rate
    silence_thresh = 2000  # Adjust based on mic sensitivity
    silence_duration = 1  # seconds of silence before stopping
    filename = "detected_recording.wav"

    def rms(data):
        """Root mean square (volume level) of audio chunk."""
        # Convert to float to avoid overflow
        data = data.astype(np.float32)
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

    return result["text"]

MESSAGE_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def chat():
    prompt = record()
    print(f"Transcribed: {prompt}")

    MESSAGE_HISTORY.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=MESSAGE_HISTORY
    )

    reply = response.choices[0].message.content
    print("AI:", reply)
    MESSAGE_HISTORY.append({"role": "assistant", "content": reply})

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=reply,
        instructions="Speak in a cheerful and positive tone.",
        response_format="wav"
    ) as response:
        response.stream_to_file("speech.wav")
    pasimple.play_wav("speech.wav")

def main():
    global SER
    with serial.Serial(TARGET, 115200, timeout=1) as SER:
        print(f"Connected to {TARGET}")

        while True:
            line = SER.readline().decode('utf-8').strip()
            if line:
                if line == "4":
                    print("Button pressed!")
                    chat()

main()
