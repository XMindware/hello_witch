import sounddevice as sd
import numpy as np
import vosk
import json
import queue
import subprocess
import openai
import requests
from gtts import gTTS
import os
import time
from dotenv import load_dotenv, dotenv_values 

# Initialize Vosk model
model = vosk.Model("./vosk-es-model")  # Update with your model path
load_dotenv() 
# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual OpenAI API key
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')  # Replace with your actual ElevenLabs API key


# Threshold for sound detection
threshold = 10
frase = "hola bruja"
idle_timeout = 10  # Seconds of silence before returning to idle

def ask_gpt(question):
    """Send the user's question to GPT-4 and return the response."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres una bruja, un poco traviesa. Das respuestas cortas."},
            {"role": "user", "content": "Responde a esta pregunta: " + question}
        ]
    )
    return response.choices[0].message['content']  # Extract the message content

def tell_something(something):
    """Send the user's question to GPT-4 and return the response."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres una bruja, un poco traviesa. Das respuestas cortas"},
            {"role": "user", "content": something}
        ]
    )
    return response.choices[0].message['content']  # Extract the message content

def despidete_por_un_rato():
    """Send the user's question to GPT-4 and return the response."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres una bruja, un poco traviesa. Das respuestas cortas, en pocas ocasiones respondes con informacion de brujas."},
            {"role": "user", "content": "Despidete por un rato"}
        ]
    )
    return response.choices[0].message['content']  # Extract the message content

def presentacion():
    """Send the user's question to GPT-4 and return the response."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres una bruja, un poco traviesa. Das respuestas cortas, en pocas ocasiones respondes con informacion de brujas."},
            {"role": "user", "content": "presentate en pocas palabras"}
        ]
    )
    return response.choices[0].message['content']  # Extract the message content


def detect_sound(device_id=0):
    """Detect sound above the threshold using the microphone."""
    def callback(indata, frames, time, status):
        if status:
            print(status)
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > threshold:
            print(f"Sound detected! Volume: {volume_norm}")
            # Stop the sound detection stream
            raise sd.CallbackStop()

    print("Listening for sound...")
    # Start the stream using the specific device
    with sd.InputStream(callback=callback, device=device_id):
        sd.sleep(10000)  # Adjust the sleep time as needed

def recognize_speech(device_id=0):
    """Recognize the trigger phrase and follow-up questions."""
    print("Listening for trigger phrase...")
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    # Initialize the recognizer
    rec = vosk.KaldiRecognizer(model, 16000)

    # Open the audio stream
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback, device=device_id):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                print("Recognized:", result['text'])
                # Check if the recognized text contains the trigger phrase
                if frase in result['text'].lower():
                    # tell something
                    response = tell_something("¿Qué quieres saber?")  # Get response from GPT
                    print(f"GPT Response: {response}")
                    speak_response_with_elevenlabs(response)
                    # Enter question-answer loop
                    question_answer_loop(device_id)
                    break
            else:
                partial_result = json.loads(rec.PartialResult())
                # Optionally, print partial results
                # print("Partial result:", partial_result['partial'])

def question_answer_loop(device_id=0):
    """Keep asking and answering questions until silence is detected."""
    last_sound_time = time.time()
    
    while True:
        print("Listening for a question...")

        question = listen_for_question(device_id)
        if question:
            response = ask_gpt(question)  # Get response from GPT
            print(f"GPT Response: {response}")
            speak_response_with_elevenlabs(response)
            #speak_response(response)  # Use TTS to speak the response
            last_sound_time = time.time()  # Reset the silence timer
        else:
            # If no sound has been detected for `idle_timeout` seconds, return to idle
            if time.time() - last_sound_time > idle_timeout:
                print("No more questions. Returning to idle state.")
                response = despidete_por_un_rato()  # Get response from GPT
                print(f"GPT Response: {response}")
                speak_response_with_elevenlabs(response)
                #speak_response(response)  # Use TTS to speak the response
                break

def listen_for_question(device_id=0):
    """Listen for the user's question or detect silence."""
    print("Listening for a question or silence...")
    q = queue.Queue()
    rec = vosk.KaldiRecognizer(model, 16000)

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback, device=device_id):
        start_time = time.time()
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                question = result['text']
                print(f"Recognized question: {question}")
                return question  # Return the recognized question
            elif time.time() - start_time > idle_timeout:
                print("Silence detected, no question asked.")
                return None  # Return None if silence detected

def speak_response_with_elevenlabs(response):
    """Convert the GPT response to speech using ElevenLabs API and play it."""
    url = "https://api.elevenlabs.io/v1/text-to-speech/r3SDVYUIvcC4EweQtSj0"
    headers = {
        "xi-api-key": elevenlabs_api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": response,
        "model_id" : "eleven_turbo_v2_5",
        "language_code": "es",
        "voice_settings": {
            "stability": 0.50,
            "similarity_boost": 0.50
        }
    }
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        audio_file_path = "response.mp3"
        with open(audio_file_path, "wb") as f:
            f.write(response.content)
        # Play the MP3 using afplay
        subprocess.call(["afplay", audio_file_path])
    else:
        print("Error with ElevenLabs TTS:", response.status_code, response.text)


def play_audio(audio_file):
    """Play a .wav audio file using afplay (macOS)."""
    subprocess.call(["afplay", audio_file])

def speak_response(response):
    """Convert the GPT response to speech using gTTS and play it."""
    # Convert the GPT response to speech
    tts = gTTS(text=response, lang='es')  # Change 'es' for Spanish or 'en' for English
    tts.save("response.mp3")  # Save the response as an MP3 file

    # Play the MP3 using afplay
    subprocess.call(["afplay", "response.mp3"])

def main():
    device_id = 0  # Update if your device ID is different
    response = presentacion() # Get response from GPT
    print(f"GPT Response: {response}")
    speak_response_with_elevenlabs(response)
    #speak_response(response)  # Use TTS to speak the response

    while True:
        detect_sound(device_id=device_id)
        recognize_speech(device_id=device_id)
        # After silence, the script returns to listening for the trigger phrase

if __name__ == "__main__":
    main()
