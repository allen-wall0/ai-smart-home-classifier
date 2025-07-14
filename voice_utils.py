import tkinter as tk
from tkinter import ttk, messagebox
import requests
import speech_recognition as sr
import edge_tts
import pygame
import io
import os
import sys
import tempfile
import uuid
import threading
import asyncio
import time
import subprocess
import shutil

# Initialize pygame mixer for audio playback
pygame.mixer.init()
tts_lock = threading.Lock()

# Global flag to signal AI voice chat to stop listening
stop_ai_chat_flag = threading.Event()

# Global variable to store the user-selected AI voice
selected_global_ai_voice = "en-US-AriaNeural" # Default fallback, will be overwritten by selection

# Define a more comprehensive list of fallback voices if edge-tts command is not found.
FALLBACK_VOICES = [
    ("en-US-AriaNeural (Female, Default)", "en-US-AriaNeural"),
    ("en-US-JennyNeural (Female)", "en-US-JennyNeural"),
    ("en-US-GuyNeural (Male)", "en-US-GuyNeural"),
    ("en-GB-SoniaNeural (Female, UK)", "en-GB-SoniaNeural"),
    ("en-GB-RyanNeural (Male, UK)", "en-GB-RyanNeural"),
    ("en-AU-NatashaNeural (Female, Australia)", "en-AU-NatashaNeural"),
    ("en-AU-WilliamNeural (Male, Australia)", "en-AU-WilliamNeural"),
    ("en-IN-NeerjaNeural (Female, India)", "en-IN-NeerjaNeural"),
    ("en-IN-PrabhatNeural (Male, India)", "en-IN-PrabhatNeural"),
    ("en-US-AnaNeural (Female)", "en-US-AnaNeural"),
    ("en-US-AndrewNeural (Male)", "en-US-AndrewNeural"),
    ("en-US-AvaNeural (Female)", "en-US-AvaNeural"),
    ("en-US-BrianNeural (Male)", "en-US-BrianNeural"),
    ("en-US-ChristopherNeural (Male)", "en-US-ChristopherNeural"),
    ("en-US-EmmaNeural (Female)", "en-US-EmmaNeural"),
    ("en-US-MichelleNeural (Female)", "en-US-MichelleNeural"),
    ("en-US-RogerNeural (Male)", "en-US-RogerNeural"),
    ("en-US-SteffanNeural (Male)", "en-US-SteffanNeural"),
    ("en-US-ElizabethNeural (Female)", "en-US-ElizabethNeural")
]

def get_all_edge_tts_voices():
    """
    Retrieves all available voices from edge-tts by running a subprocess command.
    Returns a list of tuples: (Voice Name (e.g., 'AriaNeural'), Voice ID (e.g., 'en-US-AriaNeural'))
    Provides fallback voices if edge-tts command is not found.
    """
    voices = []
    if shutil.which('edge-tts') is None:
        messagebox.showwarning(
            "edge-tts Not Found",
            "The 'edge-tts' command was not found on your system.\n\n"
            "AI voice features will use a limited set of default voices.\n\n"
            "To get all voices, please install 'edge-tts' by running:\n"
            "pip install edge-tts\n"
            "And ensure it's in your system's PATH."
        )
        print("Warning: 'edge-tts' command not found. Using fallback voices.")
        return FALLBACK_VOICES

    try:
        result = subprocess.run(
            ['edge-tts', '--list-voices'],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore',
            shell=True if sys.platform == "win32" else False,
        )
        lines = result.stdout.splitlines()

        for line in lines:
            line = line.strip()
            if line.startswith("Name:"):
                try:
                    voice_id_part = line.split('(')[-1].replace(')', '').strip()
                    parts = voice_id_part.split(',')
                    if len(parts) > 1:
                        voice_id = parts[0].strip() + "-" + parts[1].strip().replace('Neural', '') + "Neural"
                        display_name = f"{voice_id} ({parts[1].strip()})"
                    else:
                        voice_id = voice_id_part
                        display_name = voice_id_part

                    voices.append((display_name, voice_id))
                except IndexError:
                    print(f"Warning: Could not parse voice line: {line}")
                    voices.append((line.replace("Name:", "").strip(), line.replace("Name:", "").strip()))

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to list edge-tts voices: {e.stderr}\n\nUsing fallback voices.")
        print(f"Error calling edge-tts: {e.stderr}")
        voices = FALLBACK_VOICES
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred while getting voices: {e}\n\nUsing fallback voices.")
        print(f"Unexpected error getting voices: {e}")
        voices = FALLBACK_VOICES

    unique_voices = {}
    for display_name, voice_id in voices:
        if "Neural" in voice_id or len(voice_id.split('-')) >= 3:
            unique_voices[voice_id] = display_name
        elif voice_id not in unique_voices:
            unique_voices[voice_id] = display_name

    final_voices = sorted([(display_name, voice_id) for voice_id, display_name in unique_voices.items()], key=lambda x: x[0])

    if not final_voices:
        print("No voices obtained from edge-tts, falling back to predefined list.")
        final_voices = FALLBACK_VOICES
    return final_voices

def speak(text, voice_id):
    """Thread-safe TTS using edge-tts with user-selected voice."""
    async def _speak_async_task():
        tmp_file_path = None
        try:
            with tts_lock:
                communicate = edge_tts.Communicate(text, voice_id)
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]

                if audio_data:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_file.flush()
                        tmp_file_path = tmp_file.name

                    if not pygame.mixer.get_init():
                        pygame.mixer.init()

                    try:
                        pygame.mixer.music.load(tmp_file_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.1)
                    finally:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        await asyncio.sleep(0.05)
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"TTS Fallback - Would say: {text}")
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    print(f"Error cleaning up temp file '{tmp_file_path}': {e}")

    def run_async_in_thread_wrapper():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_speak_async_task())
        except Exception as e:
            print(f"Async TTS Thread Error: {e}")
        finally:
            loop.close()

    threading.Thread(target=run_async_in_thread_wrapper, daemon=True).start()

class VoiceSelectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Select AI Assistant Voice")
        master.geometry("400x200")
        master.resizable(False, False)
        master.grab_set()

        self.selected_voice_id = None
        self.all_voices = get_all_edge_tts_voices()

        tk.Label(master, text="Choose your AI Assistant Voice:", font=('Arial', 12, 'bold')).pack(pady=10)

        voice_display_names = [v[0] for v in self.all_voices]
        self.voice_combo = ttk.Combobox(
            master,
            values=voice_display_names,
            state="readonly",
            font=('Arial', 10),
            width=40
        )
        default_set = False
        for display_name, voice_id in self.all_voices:
            if "en-US-AriaNeural" in voice_id:
                self.voice_combo.set(display_name)
                default_set = True
                break
        if not default_set and voice_display_names:
            self.voice_combo.set(voice_display_names[0])
        elif not voice_display_names:
            self.voice_combo.set("No voices available")

        self.voice_combo.pack(pady=5)

        tk.Button(master, text="Confirm Voice & Launch App", command=self.confirm_selection,
                  bg='#28a745', fg='white', font=('Arial', 10, 'bold')).pack(pady=15)

        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def confirm_selection(self):
        selected_display_name = self.voice_combo.get()
        self.selected_voice_id = None
        for display_name, voice_id in self.all_voices:
            if display_name == selected_display_name:
                self.selected_voice_id = voice_id
                break
        if self.selected_voice_id:
            self.master.destroy()
        else:
            messagebox.showwarning("No Voice Selected", "Please select an AI voice before launching.")

    def on_closing(self: object) -> None:
        if messagebox.askokcancel("Exit", "Are you sure you want to exit without selecting a voice?"):
            self.master.destroy()
            sys.exit(0)