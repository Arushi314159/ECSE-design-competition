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

TARGET = "/dev/cu.usbserial-0001"

# Connect to the target serial port
SER = None

def record():
    fs = 44100       # Sample rate
    silence_thresh = 400  # Adjust based on mic sensitivity
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
    {"role": "system", "content": "You are a comforting AI that lives inside a soft toy doctor. "
            "Your role is to gently listen, provide emotional support, and help "
            "people feel safe and cared for. Speak in a warm, soothing, and "
            "encouraging tone, like a supportive friend who always wants the "
            "best for them. "
            "You can record heart rate data (including spO220) and the user's temperature using sensors in the toy."
            "Avoid giving medical or professional advice — instead, focus on "
            "empathy, reassurance, and simple coping suggestions. "
            "Be concise and clear, but emotionally intelligent. "
            "Always prioritize kindness and emotional validation."
            "Keep responses brief and to the point, no more than two sentences."
            "To trigger the sensors to record heart rate, make the last line of your response be 'HR' by itself without the quotes. This will not be shown to the user. When recording heart rate, also tell the user to place their wrist on the sensor. Once it is recorded, you will get the heart rate data."
            }
]

def chat():
    prompt = record()
    print(f"Transcribed: {prompt}")

    MESSAGE_HISTORY.append({"role": "user", "content": prompt})
    prompt_and_speak()

def prompt_and_speak():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=MESSAGE_HISTORY
    )

    reply = response.choices[0].message.content
    print("AI:", reply)

    hr = False

    if reply.endswith("HR"):
        # Trigger heart rate recording
        print("Triggering heart rate recording...")
        hr = True
        reply = reply[:-2].strip()

    MESSAGE_HISTORY.append({"role": "assistant", "content": reply})

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=reply,
        instructions="Speak in a cheerful and positive tone.",
        response_format="wav"
    ) as response:
        response.stream_to_file("speech.wav")
    # pasimple.play_wav("speech.wav")

    if hr:
        SER.write(b'1\n')  # Start heart rate measurement
        SER.flush()

def parse_sensor_data(s):
    if "," not in s or "=" not in s:
        return None, None
    
    parts = s.split(",")
    title = parts[0]
    val = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=")
            try:
                value = float(value)
            except ValueError:
                pass
            val[key] = value
    return title, val

def main():
    global SER
    with serial.Serial(TARGET, 115200, timeout=1) as SER:
        print(f"Connected to {TARGET}")

        while True:
            line = SER.readline().decode('latin-1').strip()
            if line:
                if line == "4":
                    print("Button pressed!")
                    chat()
                else:
                    title, val = parse_sensor_data(line)
                    if title:
                        print(f"{title}: {val}")
                        if title == "HR_RESULT":
                            hr = val.get("hr")
                            spo2 = val.get("spo2")
                            msg = f"The recorded heart rate is {hr} bpm and blood oxygen level is {spo2}%. Report this to the user."
                            MESSAGE_HISTORY.append({"role": "system", "content": msg})

                            prompt_and_speak()
                    else:
                        print("Received:", line)
            else:
                print("No data received from serial port.")

main()
