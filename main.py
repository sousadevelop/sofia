# Main File

#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer
import psutil, os
import subprocess
import pyaudio
import pyttsx3
import json
import core
from nlu.classifier import classify
import webbrowser

# Síntese de fala
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[-2].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def close_program(name):
    for process in (process for process in psutil.process_iter() if process.name() == name):
        process.kill()

def evaluate(text):

        # Reconhecimento da entidade de texto
    
    entity = classify(text)

    if entity == 'time|getTime':
        speak(core.SystemInfo.get_time())
    elif entity == 'time|getDate':
        speak(core.SystemInfo.get_date())

        # ABRIR PROGRAMAS ( WINDOWS )
    
    elif entity == 'open|notepad':
        speak('Abrindo o bloco de notas')
        subprocess.Popen('notepad.exe', shell=True).pid
    elif entity == 'open|chrome':
        speak('Abrindo o Google Chrome')
        os.system('"C:/Program Files/Google/Chrome/Application/chrome.exe"')
    elif entity == 'open|google':
        speak('Abrindo o Google')
        webbrowser.open("www.google.com")
        
        # Acessos recorrentes no Youtube
    elif entity == 'open|youtube':
        speak('Abrindo o Youtube')
        webbrowser.open("www.youtube.com")
    elif entity == 'open|lofi':
        speak('Abrindo o Canal de Música Lofi')
        webbrowser.open("www.youtube.com/watch?v=92VQEDu72go")

        # Redes sociais
    elif entity == 'open|linkedin':
        speak('Abrindo o Linkedin')
        webbrowser.open("www.linkedin.com/feed/")

        # FECHAR PROGRAMAS

    elif entity == 'close|notepad':
        speak('Fechando o bloco de notas')
        close_program('notepad.exe')
    
    print(text)

# Reconhecimento de fala

model = Model('model')
rec = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)
stream.start_stream()

# Loop do reconhecimento de fala
while True:
    data = stream.read(2048)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = rec.Result()
        result = json.loads(result)

        if result is not None:
            text = result['text']
            evaluate(text)