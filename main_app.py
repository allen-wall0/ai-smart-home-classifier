import tkinter as tk
from tkinter import ttk, messagebox
import sys
import threading
import numpy as np
import subprocess # For web report opening
import webbrowser # For web report opening
import os # For path manipulation
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import uuid
import speech_recognition as sr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



# Import functions/variables from newly created modules
from voice_utils import speak, VoiceSelectionApp, selected_global_ai_voice, stop_ai_chat_flag
from data_handler import load_and_prepare_data, ask_retrain_prompt
from ollama_interface import ask_ollama

class SmartHomeClassifier:
    def __init__(self, retrain_models=False, initial_ai_voice=selected_global_ai_voice):
        self.root = tk.Tk()
        self.root.title("Smart Home Activity Classifier")
        self.root.geometry("900x900")
        self.root.configure(bg="#f0f0f0")

        self.retrain_models = retrain_models
        self.current_ai_voice = initial_ai_voice
        print(f"User chose to {'RETRAIN' if self.retrain_models else 'LOAD'} the models.")
        print(f"AI Assistant Voice set to: {self.current_ai_voice}")

        self.data = None # Initialize data to None

        self.setup_gui()
        self._start_model_loading() # Start loading models in a separate thread

    def setup_gui(self):
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', pady=5)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🏠 Smart Home Activity Classifier",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(expand=True)

        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)

        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', fill='y', padx=(0, 10))

        input_frame = tk.LabelFrame(left_frame, text="Input Parameters",
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0')
        input_frame.pack(fill='x', pady=10)

        tk.Label(input_frame, text="Time:", font=('Arial', 10, 'bold'),
                bg='#f0f0f0').pack(anchor='w', padx=10, pady=(10, 0))
        self.time_combo = ttk.Combobox(
            input_frame,
            values=[], # Empty initially, filled after data load
            state="readonly",
            font=('Arial', 10)
        )
        self.time_combo.pack(fill='x', padx=10, pady=(0, 10))

        tk.Label(input_frame, text="Location:", font=('Arial', 10, 'bold'),
                bg='#f0f0f0').pack(anchor='w', padx=10)
        self.location_combo = ttk.Combobox(
            input_frame,
            values=[], # Empty initially, filled after data load
            state="readonly",
            font=('Arial', 10)
        )
        self.location_combo.pack(fill='x', padx=10, pady=(0, 10))

        tk.Label(input_frame, text="Sensor Object:", font=('Arial', 10, 'bold'),
                bg='#f0f0f0').pack(anchor='w', padx=10)
        self.sensor_combo = ttk.Combobox(
            input_frame,
            values=[], # Empty initially, filled after data load
            state="readonly",
            font=('Arial', 10)
        )
        self.sensor_combo.pack(fill='x', padx=10, pady=(0, 10))

        tk.Label(input_frame, text="Select Model:", font=('Arial', 10, 'bold'),
                bg='#f0f0f0').pack(anchor='w', padx=10)
        self.model_combo = ttk.Combobox(
            input_frame,
            values=['MLP', 'CNN', 'Transformer'],
            state="readonly",
            font=('Arial', 10)
        )
        self.model_combo.set('MLP')
        self.model_combo.bind("<<ComboboxSelected>>", self._update_model_description) # Bind to update description
        self.model_combo.pack(fill='x', padx=10, pady=(0, 5))

        # NEW: Model description label
        self.model_description_label = tk.Label(input_frame, text="", font=('Arial', 9, 'italic'),
                                               bg='#f0f0f0', wraplength=200, justify='left', fg='#555')
        self.model_description_label.pack(fill='x', padx=10, pady=(0, 10))
        self._update_model_description() # Initial call

        button_frame = tk.Frame(left_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=10)

        self.classify_btn = tk.Button(
            button_frame,
            text="🎯 Classify Activity",
            command=self.classify_activity,
            bg='#3498db', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8
        )
        self.classify_btn.pack(fill='x', pady=5)

        self.voice_btn = tk.Button(
            button_frame,
            text="🎤 Voice Input",
            command=self.voice_input,
            bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8
        )
        self.voice_btn.pack(fill='x', pady=5)

        self.stop_ai_chat_btn = tk.Button(
            button_frame,
            text="🛑 Stop AI Chat",
            command=self.stop_ai_chat,
            bg='#f39c12', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8
        )
        self.stop_ai_chat_btn.pack(fill='x', pady=5)

        self.accuracy_btn = tk.Button(
            button_frame,
            text="📊 Show Accuracy Metrics",
            command=self.show_accuracy,
            bg='#27ae60', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8
        )
        self.accuracy_btn.pack(fill='x', pady=5)

        self.confusion_btn = tk.Button(
            button_frame,
            text="📈 Confusion Matrix",
            command=self.show_confusion_matrix,
            bg='#9b59b6', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8
        )
        self.confusion_btn.pack(fill='x', pady=5)

        exit_btn = tk.Button(
            button_frame,
            text="🚪 Exit Application",
            command=self.on_closing,
            bg='#7f8c8d', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8
        )
        exit_btn.pack(fill='x', pady=5)


        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', fill='both', expand=True)

        self.result_frame = tk.LabelFrame(right_frame, text="Status & Results",
                                         font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.result_frame.pack(fill='both', expand=True, pady=10)

        self.result_text = tk.Text(
            self.result_frame,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='white',
            fg='black',
            padx=10,
            pady=10
        )
        self.result_text.pack(fill='both', expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(self.result_frame, command=self.result_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.result_text.config(yscrollcommand=scrollbar.set)

        # NEW: Progress bar and label for loading
        self.progress_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.progress_frame.pack(fill='x', padx=20, pady=(0,10))
        self.progress_label = tk.Label(self.progress_frame, text="Initializing...", bg='#f0f0f0', font=('Arial', 10, 'italic'), fg='#444')
        self.progress_label.pack(side='left', padx=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(side='right', fill='x', expand=True, padx=5)

        self._set_gui_state(False) # Disable all input/action buttons until models are loaded
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _set_gui_state(self, enabled):
        state = "normal" if enabled else "disabled"
        self.time_combo.config(state=state)
        self.location_combo.config(state=state)
        self.sensor_combo.config(state=state)
        self.model_combo.config(state=state)
        self.classify_btn.config(state=state)
        self.voice_btn.config(state=state)
        self.stop_ai_chat_btn.config(state="normal") # Ensure stop button remains active
        self.accuracy_btn.config(state=state)
        self.confusion_btn.config(state=state)

    def _update_progress(self, message, value=None):
        self.root.after(0, self.progress_label.config, {'text': message})
        if value is not None:
            self.root.after(0, self.progress_bar.config, {'value': value})

    def _start_model_loading(self):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Starting model loading and training process...\nThis may take a moment, please wait.\n")
        speak(text="Starting model loading. Please wait.", voice_id=self.current_ai_voice)
        self._set_gui_state(False) # Disable buttons
        def loading_task():
            self.data = load_and_prepare_data(retrain=self.retrain_models, progress_callback=self._update_progress)
            self.root.after(0, self._on_models_loaded)
        threading.Thread(target=loading_task, daemon=True).start()

    def _on_models_loaded(self):
        if self.data is None:
            self.result_text.insert(tk.END, "\nERROR: Failed to load models. Please check console for details.")
            self.progress_label.config(text="Loading failed!")
            self.progress_bar.config(value=0)
            messagebox.showerror("Initialization Error", "Failed to load models. Application may not function correctly.")
            return

        # Populate comboboxes
        self.time_combo['values'] = list(self.data['encoders']['time'].classes_)
        self.location_combo['values'] = list(self.data['encoders']['location'].classes_)
        self.sensor_combo['values'] = list(self.data['encoders']['sensor'].classes_)

        # Set default selections if possible
        if self.time_combo['values']:
            self.time_combo.set(self.time_combo['values'][0])
        if self.location_combo['values']:
            self.location_combo.set(self.location_combo['values'][0])
        if self.sensor_combo['values']:
            self.sensor_combo.set(self.sensor_combo['values'][0])

        welcome_msg = f"""Welcome to Smart Home Activity Classifier!
Model Performance:
• MLP Accuracy: {self.data['accuracy']['mlp'] * 100:.2f}%
• CNN Accuracy: {self.data['accuracy']['cnn'] * 100:.2f}%
• Transformer Accuracy: {self.data['accuracy']['transformer'] * 100:.2f}%

Instructions:
1. Select Time, Location, and Sensor Object.
2. Select a Model (MLP, CNN, Transformer) and see its description below.
3. Click 'Classify Activity' to get prediction and AI explanation.
4. Use 'Voice Input' for speech recognition, 'Stop AI Chat' to end it.
5. View model performance with accuracy metrics and confusion matrix buttons (opens a web report).
6. Click 'Exit Application' to close.

Ready to classify activities!"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, welcome_msg)
        speak(text="Smart Home Activity Classifier is ready!", voice_id=self.current_ai_voice)
        self._set_gui_state(True) # Enable buttons
        self.progress_label.config(text="Ready!")
        self.progress_bar.config(value=100)
        self.progress_frame.pack_forget() # Hide progress bar once done

    def _update_model_description(self, event=None):
        selected_model = self.model_combo.get()
        descriptions = {
            'MLP': "Multi-Layer Perceptron: A foundational neural network, good for learning complex patterns in structured data. Often fast to train.",
            'CNN': "Convolutional Neural Network: Excels at feature extraction. Useful for sequence data where local patterns are important.",
            'Transformer': "Transformer Model: Powerful for capturing long-range dependencies and global context within sequences. Can be more complex. (Note: SHAP explanation might be slower for this model.)"
        }
        self.model_description_label.config(text=descriptions.get(selected_model, "No description available."))

    def classify_activity(self):
        if self.data is None or not self.data['models']['mlp']:
            messagebox.showerror("Model Error", "Models are not loaded. Please wait for initialization or check for errors.")
            return

        time_input = self.time_combo.get()
        location_input = self.location_combo.get()
        sensor_input = self.sensor_combo.get()

        if not all([time_input, location_input, sensor_input]):
            messagebox.showwarning("Input Error", "Please select all parameters!")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Classifying activity...\n")
        self._set_gui_state(False) # Disable buttons during classification
        self.stop_ai_chat_btn.config(state="disabled") # Disable stop chat button during classification too

        def classification_task():
            try:
                encoded_input = [
                    self.data['encoders']['time'].transform([time_input])[0],
                    self.data['encoders']['location'].transform([location_input])[0],
                    self.data['encoders']['sensor'].transform([sensor_input])[0]
                ]

                selected_model_key = self.model_combo.get().lower()
                model_to_use = self.data['models'].get(selected_model_key)

                if model_to_use is None:
                    self.root.after(0, messagebox.showerror, "Model Error", f"Selected model '{selected_model_key}' is not available.")
                    self.root.after(0, self._set_gui_state, True)
                    self.root.after(0, self.stop_ai_chat_btn.config, {"state": "normal"})
                    return

                input_array = np.array([encoded_input])
                prediction = model_to_use.predict(input_array, verbose=0)[0]
                predicted_class = np.argmax(prediction)

                if predicted_class >= len(self.data['encoders']['activity'].classes_):
                    raise ValueError(f"Predicted class {predicted_class} out of range for activities.")

                predicted_label = self.data['encoders']['activity'].inverse_transform([predicted_class])[0]
                confidence = prediction[predicted_class] * 100

                feature_names = ['Time_Encoded', 'Location_Encoded', 'Sensor_Encoded']


                self.root.after(0, self.result_text.insert, tk.END, "Generating AI analysis...\n")
                ollama_prompt = f"""
    Analyze this smart home activity prediction:

    Input Parameters:
    - Time: {time_input}
    - Location: {location_input}
    - Sensor: {sensor_input}

    Prediction Results:
    - Predicted Activity: {predicted_label}
    - Confidence: {confidence:.2f}%


    Provide a concise analysis of the smart home activity prediction below, incorporating the feature importance.
    Explain which input parameters most influenced the prediction and why.
    Avoid making up numbers or using unrelated percentages. Provide a brief technical analysis of this prediction, considering the logical relationship between the input parameters and the predicted activity.
    Keep the response conversational and under 150 words.
                """
                ai_analysis = ask_ollama(ollama_prompt)

                result_text = f"""
    🎯 CLASSIFICATION RESULTS
    {'='*50}
    Input Parameters:
    • Time: {time_input}
    • Location: {location_input}
    • Sensor: {sensor_input}

    Prediction:
    • Activity: {predicted_label}
    • Confidence: {confidence:.2f}%

    AI Analysis:
    {ai_analysis}
    {'='*50}
                """
                self.root.after(0, self.result_text.delete, 1.0, tk.END)
                self.root.after(0, self.result_text.insert, tk.END, result_text)
                
                # FIXED: Added self.root.after to ensure thread-safe execution
                # This will speak both the prediction and the AI analysis
                speech_text = f"Predicted activity is {predicted_label} with {confidence:.2f} percent confidence. {ai_analysis}"
                self.root.after(0, lambda: speak(speech_text, voice_id=self.current_ai_voice))

            except Exception as e:
                error_message = f"An error occurred during classification: {e}"
                print(error_message)
                self.root.after(0, self.result_text.delete, 1.0, tk.END)
                self.root.after(0, self.result_text.insert, tk.END, f"Error: {error_message}\n")
                self.root.after(0, messagebox.showerror, "Classification Error", error_message)
            finally:
                self.root.after(0, self._set_gui_state, True)
                self.root.after(0, self.stop_ai_chat_btn.config, {"state": "normal"})

        threading.Thread(target=classification_task, daemon=True).start()

    def voice_input(self):
        """
        AI Voice Chat functionality - handles speech recognition and AI responses
        """
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Listening for voice input...\n")
        speak(text="I am listening.", voice_id=self.current_ai_voice)
        self._set_gui_state(False)
        self.stop_ai_chat_btn.config(state="normal") # Keep stop button active

        def recognize_speech_task():
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                try:
                    # Use a global flag to stop listening if the user clicks the stop button
                    audio_data = r.listen(source, timeout=5, phrase_time_limit=10) # Listens for up to 5 seconds, max phrase 10 seconds
                    self.root.after(0, self.result_text.insert, tk.END, "Recognizing...\n")
                    text = r.recognize_google(audio_data)
                    self.root.after(0, self.result_text.insert, tk.END, f"You said: {text}\n")
                    self.root.after(0, self.result_text.insert, tk.END, "Asking AI for response...\n")
                    speak(f"You said: {text}. Getting AI response.", voice_id=self.current_ai_voice)

                    # AI RESPONSE GENERATION
                    ollama_prompt = f"The user said: '{text}'. Respond to the user's query about smart home activities or related topics. Keep the response concise, under 100 words, and helpful."
                    ai_response = ask_ollama(ollama_prompt)  # This calls the AI (Ollama)

                    self.root.after(0, self.result_text.insert, tk.END, f"AI Response: {ai_response}\n")
                    speak(ai_response, voice_id=self.current_ai_voice)  # AI speaks the response

                except sr.WaitTimeoutError:
                    if not stop_ai_chat_flag.is_set():
                        self.root.after(0, self.result_text.insert, tk.END, "No speech detected. Please try again.\n")
                        speak(text="No speech detected.", voice_id=self.current_ai_voice)
                except sr.UnknownValueError:
                    if not stop_ai_chat_flag.is_set():
                        self.root.after(0, self.result_text.insert, tk.END, "Could not understand audio.\n")
                        speak(text="Could not understand audio.", voice_id=self.current_ai_voice)
                except sr.RequestError as e:
                    self.root.after(0, self.result_text.insert, tk.END, f"Could not request results from Google Speech Recognition service; {e}\n")
                    speak(text=f"Speech recognition service error: {e}.", voice_id=self.current_ai_voice)
                except Exception as e:
                    self.root.after(0, self.result_text.insert, tk.END, f"An unexpected error occurred: {e}\n")
                    speak(text=f"An unexpected error occurred: {e}.", voice_id=self.current_ai_voice)
                finally:
                    stop_ai_chat_flag.clear() # Reset the flag
                    self.root.after(0, self._set_gui_state, True)

        threading.Thread(target=recognize_speech_task, daemon=True).start()

    def stop_ai_chat(self):
        """
        Stops the AI voice chat by setting a global flag
        """
        stop_ai_chat_flag.set() # Set the flag to signal stopping
        self.result_text.insert(tk.END, "\nAI voice chat stopped.\n")
        print("AI voice chat stopped by user.")
        # FIXED: Removed self. prefix since speak is imported from voice_utils
        speak(text="AI chat stopped.", voice_id=self.current_ai_voice)

    def show_accuracy(self):
        if self.data is None:
            messagebox.showerror("Data Error", "Model data is not loaded. Please wait for initialization.")
            return

        mlp_acc = self.data['accuracy'].get('mlp', 0.0)
        cnn_acc = self.data['accuracy'].get('cnn', 0.0)
        transformer_acc = self.data['accuracy'].get('transformer', 0.0)

        mlp_val_loss = self.data['validation_loss'].get('mlp', 'N/A')
        cnn_val_loss = self.data['validation_loss'].get('cnn', 'N/A')
        transformer_val_loss = self.data['validation_loss'].get('transformer', 'N/A')

        # Get best epoch values
        mlp_best_epoch = self.data['best_epoch'].get('mlp', 'N/A')
        cnn_best_epoch = self.data['best_epoch'].get('cnn', 'N/A')
        transformer_best_epoch = self.data['best_epoch'].get('transformer', 'N/A')
     
        avg_acc = (mlp_acc + cnn_acc + transformer_acc) / 3 if self.data['accuracy'] else 0.0
        performance = 'good' if avg_acc > 0.8 else ('moderate' if avg_acc > 0.5 else 'low')

        mlp_class_report_dict = self.data['classification_reports'].get('mlp', {})
        cnn_class_report_dict = self.data['classification_reports'].get('cnn', {})
        transformer_class_report_dict = self.data['classification_reports'].get('transformer', {})

        mlp_summary = self.data['model_summaries'].get('mlp', 'N/A')
        cnn_summary = self.data['model_summaries'].get('cnn', 'N/A')
        transformer_summary = self.data['model_summaries'].get('transformer', 'N/A')

        num_classes = len(self.data['encoders']['activity'].classes_)
        activity_names = ', '.join(self.data['encoders']['activity'].classes_)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Smart Home Classifier Accuracy Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4a69bd; color: white; }}
                tr:nth-child(even) {{ background-color: #f8f8f8; }}
                .highlight {{ background-color: #e6f7ff; font-weight: bold; }}
                .performance {{ text-align: center; margin-top: 30px; padding: 15px; border: 1px solid #cce5ff; background-color: #e0f2f7; border-radius: 8px; }}
                .section {{ background-color: #ffffff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                pre {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
            </style>
        </head>
        <body>
            <h1>📊 Smart Home Activity Classifier Accuracy Report</h1>
            <div class="section">
                <h2>Overall Model Performance</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Validation Loss (Lower is Better)</th>
                            <th>Best Epoch</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>MLP</strong></td>
                            <td>{mlp_acc * 100:.2f}%</td>
                            <td>{mlp_val_loss:.4f}</td>
                            <td>{mlp_best_epoch}</td>
                        </tr>
                        <tr>
                            <td><strong>CNN</strong></td>
                            <td>{cnn_acc * 100:.2f}%</td>
                            <td>{cnn_val_loss:.4f}</td>
                            <td>{cnn_best_epoch}</td>
                        </tr>
                        <tr>
                            <td><strong>Transformer</strong></td>
                            <td>{transformer_acc * 100:.2f}%</td>
                            <td>{transformer_val_loss:.4f}</td>
                            <td>{transformer_best_epoch}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        """
        def generate_classification_report_html(report_dict, model_name):
            if report_dict and isinstance(report_dict, dict):
                html = f"""
                <div class="section">
                    <h2>Detailed Classification Report ({model_name} Model)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Activity</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>Support</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                for label, metrics in report_dict.items():
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        html += f"""
                            <tr>
                                <td>{label}</td>
                                <td>{metrics['precision']:.2f}</td>
                                <td>{metrics['recall']:.2f}</td>
                                <td>{metrics['f1-score']:.2f}</td>
                                <td>{metrics['support']}</td>
                            </tr>
                        """
                if 'macro avg' in report_dict:
                    macro_avg = report_dict['macro avg']
                    html += f"""
                            <tr class="highlight">
                                <td><strong>macro avg</strong></td>
                                <td>{macro_avg['precision']:.2f}</td>
                                <td>{macro_avg['recall']:.2f}</td>
                                <td>{macro_avg['f1-score']:.2f}</td>
                                <td>{macro_avg['support']}</td>
                            </tr>
                    """
                if 'weighted avg' in report_dict:
                    weighted_avg = report_dict['weighted avg']
                    html += f"""
                            <tr class="highlight">
                                <td><strong>weighted avg</strong></td>
                                <td>{weighted_avg['precision']:.2f}</td>
                                <td>{weighted_avg['recall']:.2f}</td>
                                <td>{weighted_avg['f1-score']:.2f}</td>
                                <td>{weighted_avg['support']}</td>
                            </tr>
                    """
                html += f"""
                        </tbody>
                    </table>
                    <p style="margin-top: 15px;"><strong>Overall Accuracy ({model_name}):</strong> {report_dict.get('accuracy', 0.0) * 100:.2f}%</p>
                </div>
                """
                return html
            else:
                return f"""
                <div class="section">
                    <h2>Detailed Classification Report ({model_name} Model)</h2>
                    <p>Classification report for {model_name} not available.</p>
                </div>
                """

        html_content += generate_classification_report_html(mlp_class_report_dict, "MLP")
        html_content += generate_classification_report_html(cnn_class_report_dict, "CNN")
        html_content += generate_classification_report_html(transformer_class_report_dict, "Transformer")

        # NEW: Model Architectures
        html_content += f"""
            <div class="section">
                <h2>Model Architecture - MLP</h2>
                <pre>{mlp_summary}</pre>
            </div>
            <div class="section">
                <h2>Model Architecture - CNN</h2>
                <pre>{cnn_summary}</pre>
            </div>
            <div class="section">
                <h2>Model Architecture - Transformer</h2>
                <pre>{transformer_summary}</pre>
            </div>
        """

        html_content += f"""
            <div class="section">
                <h2>Training Configuration</h2>
                <pre>
    • Max Epochs: 200 (with Early Stopping)
    • Early Stopping Patience: 10 epochs (monitors validation loss)
    • Optimizer: Adam
    • Loss Function: Sparse Categorical Crossentropy</pre>
            </div>

            <div class="section">
                <h2>Dataset Information</h2>
                <pre>
    • Total Classes: {num_classes}
    • Activities: {activity_names}</pre>
            </div>

            <div class="performance">
                <p>The models collectively show <strong>{performance}</strong> performance based on average accuracy across MLP, CNN, and Transformer.</p>
            </div>

        </body>
        </html>
        """
        
        # Save and open the HTML report, and capture the file path
        file_path = self._save_and_open_html_report("accuracy_report.html", html_content)
        
        # Update the result text with the correct file path
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"📊 Accuracy report generated and opened in your web browser:\n{file_path}\n\n"
                                        "Please check your browser for detailed metrics and model information.")
        speak(text="Accuracy report generated and opened in your web browser", voice_id=self.current_ai_voice)


    def format_classification_report(self, report_dict):
        if report_dict == "N/A":
            return "Report not available."
        
        # Manually format the dictionary into a human-readable string, similar to sklearn's print
        lines = []
        headers = ["precision", "recall", "f1-score", "support"]
        
        # Max width for class names for alignment
        max_label_width = max(len(label) for label in report_dict.keys() if label not in ['accuracy', 'macro avg', 'weighted avg']) if report_dict else 0
        max_label_width = max(max_label_width, len('accuracy'), len('macro avg'), len('weighted avg'))
        
        # Header row - Fixed the f-string syntax
        header_line = f"{'':>{max_label_width}} {'':>9}{'precision':>9}{'recall':>9}{'f1-score':>9}{'support':>9}"
        lines.append(header_line)
        
        # Per-class metrics
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict): # Individual class reports
                line = f"{label:<{max_label_width}} {metrics['precision']:>9.2f}{metrics['recall']:>9.2f}{metrics['f1-score']:>9.2f}{metrics['support']:>9.0f}"
                lines.append(line)
            elif label == 'accuracy': # Overall accuracy
                lines.append("") # Blank line before accuracy
                support_val = report_dict.get('macro avg', {}).get('support', 0)
                line = f"{label:<{max_label_width}} {'':>9}{metrics:>9.2f}{'':>9}{support_val:>9.0f}"
                lines.append(line)
            
        # Macro and Weighted Averages
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_dict:
                metrics = report_dict[avg_type]
                line = f"{avg_type:<{max_label_width}} {metrics['precision']:>9.2f}{metrics['recall']:>9.2f}{metrics['f1-score']:>9.2f}{metrics['support']:>9.0f}"
                lines.append(line)
        
        return "\n".join(lines)


    def show_confusion_matrix(self):
        if self.data is None:
            messagebox.showerror("Data Error", "Model data is not loaded. Please wait for initialization.")
            return

        cm = self.data.get('confusion_matrix')
        if cm is None or cm.size == 0:
            messagebox.showinfo("Confusion Matrix", "Confusion matrix data not available. Please ensure models were trained successfully.")
            return

        cm_window = tk.Toplevel(self.root)
        cm_window.title("Confusion Matrix")
        cm_window.geometry("1000x900")
        cm_window.configure(bg='#f0f0f0')

        # Generate the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.data['encoders']['activity'].classes_,
            yticklabels=self.data['encoders']['activity'].classes_,
            ax=ax
        )

        ax.set_title('Confusion Matrix (MLP Model)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Activity', fontsize=12)
        ax.set_ylabel('True Activity', fontsize=12)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

        plt.tight_layout()

        # Center the canvas in the window
        canvas = FigureCanvasTkAgg(fig, master=cm_window)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(expand=True)  # Don't use fill='both' if you want it centered

        # Update result display and voice
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "✅ Confusion matrix displayed in a separate window.\n")
        speak(text="Confusion matrix is now displayed.", voice_id=self.current_ai_voice)


    def _save_and_open_html_report(self, filename, html_content):
        """
        Save HTML content to a file and open it in the default web browser.
        Returns the full file path.
        """
        
        # Create the full file path
        file_path = os.path.join(os.getcwd(), filename)
        
        try:
            # Write the HTML content to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Open the file in the default web browser
            webbrowser.open(f'file://{file_path}')
            
            # Return the file path so it can be used in the calling method
            return file_path
            
        except Exception as e:
            print(f"Error saving or opening HTML report: {e}")
            return None

    def on_closing(self):
        if messagebox.askokcancel("Exit Application", "Do you want to exit the application?"):
            self.root.destroy()
            sys.exit(0)

    def run(self: object) -> None:
        self.root.mainloop()

def main():
    global selected_global_ai_voice # Use the global variable defined in voice_utils
    print("🚀 Starting Smart Home Activity Classifier...")
    try:
        # Step 1: Voice Selection before main GUI
        voice_root = tk.Tk()
        voice_app = VoiceSelectionApp(voice_root)
        voice_root.mainloop()

        if voice_app.selected_voice_id is None and not messagebox.askyesno(
            "Continue Anyway?",
            "No AI voice was selected. Do you want to continue with the default voice and potentially limited voice features?"
        ):
            print("User opted to exit after no voice selection.")
            sys.exit(0)
        elif voice_app.selected_voice_id is not None:
             selected_global_ai_voice = voice_app.selected_voice_id

        print(f"User selected AI Voice: {selected_global_ai_voice}")

        # Step 2: Ask about retraining
        root = tk.Tk()
        root.withdraw() # Hide the root window
        
        retrain_choice = ask_retrain_prompt()
        root.destroy() # Destroy the hidden root window

        # Step 3: Launch main application with selected voice
        app = SmartHomeClassifier(retrain_models=retrain_choice, initial_ai_voice=selected_global_ai_voice)
        app.run()

    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}\nApplication will now exit.")
        sys.exit(1)

if __name__ == "__main__":
    main()