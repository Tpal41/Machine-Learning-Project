"""
Flask Web Application for Student Performance Prediction
=========================================================
This application provides a user-friendly web interface to predict
student final exam marks using the trained Machine Learning model.

Routes:
- / (GET): Home page with input form
- /predict (POST): Handles prediction request and shows results
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize Flask application
app = Flask(__name__)

# Load the trained Machine Learning model
MODEL_PATH = 'model/student_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print("✓ Machine Learning model loaded successfully!")
except FileNotFoundError:
    print("✗ Error: Model not found. Please run 'python train_model.py' first.")
    model = None

# Home route - Display input form
@app.route('/')
def home():
    """
    Renders the home page with input form for student data
    """
    return render_template('index.html')

# Prediction route - Process form and predict
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction request:
    1. Receives form data (study hours, attendance, previous marks)
    2. Validates inputs
    3. Makes prediction using ML model
    4. Returns result page with prediction
    """
    
    if model is None:
        return render_template('result.html', 
                             error="Model not found. Please train the model first.")
    
    try:
        # Get input values from form
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        previous_marks = float(request.form['previous_marks'])
        
        # Validate input ranges
        errors = []
        
        if not (0 <= study_hours <= 24):
            errors.append("Study hours must be between 0 and 24")
        
        if not (0 <= attendance <= 100):
            errors.append("Attendance must be between 0 and 100")
        
        if not (0 <= previous_marks <= 100):
            errors.append("Previous marks must be between 0 and 100")
        
        if errors:
            return render_template('result.html', 
                                 error="<br>".join(errors),
                                 study_hours=study_hours,
                                 attendance=attendance,
                                 previous_marks=previous_marks)
        
        # Prepare input for prediction
        # Model expects: [StudyHours, Attendance, PreviousMarks]
        input_features = np.array([[study_hours, attendance, previous_marks]])
        
        # Make prediction using trained ML model
        predicted_marks = model.predict(input_features)[0]
        
        # Ensure prediction is within valid range (0-100)
        predicted_marks = max(0, min(100, predicted_marks))
        
        # Round to 2 decimal places
        predicted_marks = round(predicted_marks, 2)
        
        # Determine performance category
        if predicted_marks >= 90:
            category = "Outstanding"
            grade = "A+"
            message = "Excellent work! Keep it up!"
        elif predicted_marks >= 80:
            category = "Excellent"
            grade = "A"
            message = "Great performance! You're doing well!"
        elif predicted_marks >= 70:
            category = "Good"
            grade = "B"
            message = "Good job! There's room for improvement."
        elif predicted_marks >= 60:
            category = "Average"
            grade = "C"
            message = "Fair performance. Focus on improving study habits."
        elif predicted_marks >= 50:
            category = "Below Average"
            grade = "D"
            message = "Need improvement. Increase study hours and attendance."
        else:
            category = "Poor"
            grade = "F"
            message = "Needs significant improvement. Please consult with teachers."
        
        # Return result page with prediction
        return render_template('result.html',
                             prediction=predicted_marks,
                             study_hours=study_hours,
                             attendance=attendance,
                             previous_marks=previous_marks,
                             category=category,
                             grade=grade,
                             message=message)
    
    except ValueError:
        return render_template('result.html', 
                             error="Invalid input. Please enter valid numbers.")
    except Exception as e:
        return render_template('result.html', 
                             error=f"An error occurred: {str(e)}")

# Run the Flask application
if __name__ == '__main__':
    print("=" * 80)
    print("STUDENT PERFORMANCE PREDICTION - WEB APPLICATION")
    print("=" * 80)
    print("\n✓ Starting Flask server...")
    print("✓ Machine Learning model loaded")
    print("\n➜ Open your browser and go to: http://127.0.0.1:5000/")
    print("➜ Press CTRL+C to stop the server\n")
    print("=" * 80)
    
    # Run Flask app in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000)
