import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os
import tkinter as tk
from tkinter import messagebox
import pickle

# Assuming model_builders.py is in the same directory or accessible in PYTHONPATH
from model_builders import build_mlp_model, build_cnn_model, build_transformer_model

def get_model_summary_str(model):
    """Helper to capture model summary as a string."""
    if model:
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)
    return "Model not available."

def save_training_info(model_name, best_epoch, total_epochs, model_dir):
    """Save training information including best epoch."""
    training_info = {
        'best_epoch': best_epoch,
        'total_epochs': total_epochs
    }
    info_path = os.path.join(model_dir, f"{model_name}_training_info.pkl")
    with open(info_path, 'wb') as f:
        pickle.dump(training_info, f)

def load_training_info(model_name, model_dir):
    """Load training information including best epoch."""
    info_path = os.path.join(model_dir, f"{model_name}_training_info.pkl")
    if os.path.exists(info_path):
        try:
            with open(info_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {'best_epoch': 'N/A', 'total_epochs': 'N/A'}

def load_and_prepare_data(retrain=False, progress_callback=None):
    """
    Loads data, trains/loads models, and prepares all necessary data for the app.
    Includes a progress_callback for GUI updates.
    """
    def update_progress(message, value):
        if progress_callback:
            progress_callback(message, value)

    try:
        csv_path = "c:\\UniMAP Mechatronic\\Year 3 Sem 2\\Artificial Intelligence for Mechatronic Engineering\\3. Assignments\\Mini Project_Smart Home\\Dataset\\Smart_Home_Data_Converted.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")

        update_progress("Loading dataset...", 10)
        df = pd.read_csv(csv_path)

        required_cols = ['Time', 'Location', 'Sensor-object', 'Activity label (Ground Truth)']
        df = df[required_cols].copy()
        df.columns = ['Time', 'Location', 'Sensor_Object', 'Activity']
        df = df.dropna()

        update_progress("Encoding features...", 20)
        le_time = LabelEncoder()
        le_location = LabelEncoder()
        le_sensor = LabelEncoder()
        le_activity = LabelEncoder()

        df['Time_Encoded'] = le_time.fit_transform(df['Time'])
        df['Location_Encoded'] = le_location.fit_transform(df['Location'])
        df['Sensor_Encoded'] = le_sensor.fit_transform(df['Sensor_Object'])
        df['Activity_Encoded'] = le_activity.fit_transform(df['Activity'])

        X = df[['Time_Encoded', 'Location_Encoded', 'Sensor_Encoded']]
        y = df['Activity_Encoded']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y))

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        mlp_path = os.path.join(model_dir, "mlp_model.keras")
        cnn_path = os.path.join(model_dir, "cnn_model.keras")
        transformer_path = os.path.join(model_dir, "transformer_model.keras")

        mlp_model, cnn_model, transformer_model = None, None, None
        mlp_val_loss, cnn_val_loss, transformer_val_loss = float('inf'), float('inf'), float('inf')
        mlp_summary, cnn_summary, transformer_summary = "N/A", "N/A", "N/A"
        
        # Initialize best epoch tracking
        mlp_best_epoch, cnn_best_epoch, transformer_best_epoch = 'N/A', 'N/A', 'N/A'

        # MLP Model
        if not retrain and os.path.exists(mlp_path):
            try:
                update_progress("Loading saved MLP model...", 30)
                mlp_model = tf.keras.models.load_model(mlp_path)
                _ = mlp_model.predict(np.zeros((1, input_shape[0])), verbose=0) # Warm-up prediction
                mlp_val_loss = mlp_model.evaluate(X_test, y_test, verbose=0)[0]
                mlp_summary = get_model_summary_str(mlp_model)
                
                # Load training info for best epoch
                training_info = load_training_info("mlp", model_dir)
                mlp_best_epoch = training_info['best_epoch']
                
                update_progress("MLP model loaded.", 40)
            except Exception as e:
                print(f"❌ Failed to load MLP model: {e}")
                update_progress("Training MLP model...", 30)
        if mlp_model is None or retrain:
            update_progress("Training MLP model...", 30)
            mlp_model = build_mlp_model(input_shape, num_classes)
            mlp_history = mlp_model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=0,
                                        callbacks=[early_stopping_callback])
            mlp_val_loss = mlp_history.history['val_loss'][-1]
            
            # Calculate best epoch (1-indexed)
            best_epoch_idx = np.argmin(mlp_history.history['val_loss'])
            mlp_best_epoch = best_epoch_idx + 1
            total_epochs = len(mlp_history.history['val_loss'])
            
            mlp_model.save(mlp_path)
            save_training_info("mlp", mlp_best_epoch, total_epochs, model_dir)
            mlp_summary = get_model_summary_str(mlp_model)
            update_progress("MLP model trained and saved.", 40)

        # CNN Model
        if not retrain and os.path.exists(cnn_path):
            try:
                update_progress("Loading saved CNN model...", 50)
                cnn_model = tf.keras.models.load_model(cnn_path)
                _ = cnn_model.predict(np.zeros((1, input_shape[0])), verbose=0) # Warm-up prediction
                cnn_val_loss = cnn_model.evaluate(X_test, y_test, verbose=0)[0]
                cnn_summary = get_model_summary_str(cnn_model)
                
                # Load training info for best epoch
                training_info = load_training_info("cnn", model_dir)
                cnn_best_epoch = training_info['best_epoch']
                
                update_progress("CNN model loaded.", 60)
            except Exception as e:
                print(f"❌ Failed to load CNN model: {e}")
                update_progress("Training CNN model...", 50)
        if cnn_model is None or retrain:
            update_progress("Training CNN model...", 50)
            cnn_model = build_cnn_model(input_shape, num_classes)
            cnn_history = cnn_model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=0,
                                        callbacks=[early_stopping_callback])
            cnn_val_loss = cnn_history.history['val_loss'][-1]
            
            # Calculate best epoch (1-indexed)
            best_epoch_idx = np.argmin(cnn_history.history['val_loss'])
            cnn_best_epoch = best_epoch_idx + 1
            total_epochs = len(cnn_history.history['val_loss'])
            
            cnn_model.save(cnn_path)
            save_training_info("cnn", cnn_best_epoch, total_epochs, model_dir)
            cnn_summary = get_model_summary_str(cnn_model)
            update_progress("CNN model trained and saved.", 60)

        # Transformer Model
        if not retrain and os.path.exists(transformer_path):
            try:
                update_progress("Loading saved Transformer model...", 70)
                transformer_model = tf.keras.models.load_model(transformer_path)
                _ = transformer_model.predict(np.zeros((1, input_shape[0])), verbose=0) # Warm-up prediction
                transformer_val_loss = transformer_model.evaluate(X_test, y_test, verbose=0)[0]
                transformer_summary = get_model_summary_str(transformer_model)
                
                # Load training info for best epoch
                training_info = load_training_info("transformer", model_dir)
                transformer_best_epoch = training_info['best_epoch']
                
                update_progress("Transformer model loaded.", 80)
            except Exception as e:
                print(f"❌ Failed to load Transformer model: {e}")
                update_progress("Training Transformer model...", 70)
        if transformer_model is None or retrain:
            update_progress("Training Transformer model...", 70)
            transformer_model = build_transformer_model(input_shape, num_classes)
            transformer_history = transformer_model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=0,
                                                        callbacks=[early_stopping_callback])
            transformer_val_loss = transformer_history.history['val_loss'][-1]
            
            # Calculate best epoch (1-indexed)
            best_epoch_idx = np.argmin(transformer_history.history['val_loss'])
            transformer_best_epoch = best_epoch_idx + 1
            total_epochs = len(transformer_history.history['val_loss'])
            
            transformer_model.save(transformer_path)
            save_training_info("transformer", transformer_best_epoch, total_epochs, model_dir)
            transformer_summary = get_model_summary_str(transformer_model)
            update_progress("Transformer model trained and saved.", 80)

        update_progress("Evaluating models...", 90)
        mlp_acc = mlp_model.evaluate(X_test, y_test, verbose=0)[1] if mlp_model else 0.0
        cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)[1] if cnn_model else 0.0
        transformer_acc = transformer_model.evaluate(X_test, y_test, verbose=0)[1] if transformer_model else 0.0

        mlp_classification_report = "N/A"
        cnn_classification_report = "N/A"
        transformer_classification_report = "N/A"

        if mlp_model is not None:
            y_pred_mlp = np.argmax(mlp_model.predict(X_test, verbose=0), axis=1)
            mlp_classification_report = classification_report(
                y_test, y_pred_mlp,
                target_names=le_activity.classes_,
                output_dict=True,
                zero_division=0
            )

        if cnn_model is not None:
            y_pred_cnn = np.argmax(cnn_model.predict(X_test, verbose=0), axis=1)
            cnn_classification_report = classification_report(
                y_test, y_pred_cnn,
                target_names=le_activity.classes_,
                output_dict=True,
                zero_division=0
            )

        if transformer_model is not None:
            y_pred_transformer = np.argmax(transformer_model.predict(X_test, verbose=0), axis=1)
            transformer_classification_report = classification_report(
                y_test, y_pred_transformer,
                target_names=le_activity.classes_,
                output_dict=True,
                zero_division=0
            )

        cm = np.array([[]])
        if mlp_model is not None:
            y_pred_mlp_on_test = np.argmax(mlp_model.predict(X_test, verbose=0), axis=1)
            cm = confusion_matrix(y_test, y_pred_mlp_on_test)

        update_progress("Models ready!", 100)
        return {
            'models': {
                'mlp': mlp_model,
                'cnn': cnn_model,
                'transformer': transformer_model
            },
            'encoders': {
                'time': le_time,
                'location': le_location,
                'sensor': le_sensor,
                'activity': le_activity
            },
            'test_data': (X_test, y_test),
            'X_train': X_train, # Added for SHAP background data
            'accuracy': {
                'mlp': mlp_acc,
                'cnn': cnn_acc,
                'transformer': transformer_acc
            },
            'validation_loss': {
                'mlp': mlp_val_loss,
                'cnn': cnn_val_loss,
                'transformer': transformer_val_loss
            },
            'best_epoch': {  # NEW: Best epoch information
                'mlp': mlp_best_epoch,
                'cnn': cnn_best_epoch,
                'transformer': transformer_best_epoch
            },
            'classification_reports': {
                'mlp': mlp_classification_report,
                'cnn': cnn_classification_report,
                'transformer': transformer_classification_report
            },
            'model_summaries': { # Store model summaries
                'mlp': mlp_summary,
                'cnn': cnn_summary,
                'transformer': transformer_summary
            },
            'confusion_matrix': cm,
        }

    except Exception as e:
        update_progress(f"Error: {str(e)}", 0)
        print(f"Error loading data or training models: {e}")
        messagebox.showerror("Data Load Error", f"Failed to load data or train models: {str(e)}\nPlease ensure the CSV path is correct and data is valid.")
        return None

def ask_retrain_prompt():
    answer = messagebox.askyesno(
        "Retrain Models?",
        "Do you want to retrain the models from scratch?\n\n"
        "Yes = Retrain models from scratch (deletes old .keras models)\n"
        "No = Load saved models if available (from .keras files)"
    )
    return answer