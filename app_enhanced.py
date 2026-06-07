"""
Enhanced College Placement Predictor with Authentication & Quiz
==============================================================
Features:
- User authentication (Login/Signup)
- Interactive quiz before prediction
- Session management
- User dashboard
"""

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'college_placement_predictor_2024_secret_key'

# Load ML models
MODEL_PATH = 'model/college_placement_model.pkl'

try:
    model_data = joblib.load(MODEL_PATH)
    placement_model = model_data['placement_model']
    package_model = model_data['package_model']
    print("✓ ML Models loaded!")
except:
    placement_model = None
    package_model = None
    print("⚠ Models not found!")

# Simple user database (in production, use proper database)
USERS_FILE = 'users_data.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

@app.route('/')
def index():
    """Home page - show main form directly"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        
        if email in users and users[email]['password'] == password:
            session['user_email'] = email
            session['user_name'] = users[email]['first_name']
            return redirect(url_for('quiz_page'))
        else:
            return render_template('login.html', error='Invalid email or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        college = request.form.get('college')
        branch = request.form.get('branch')
        year = request.form.get('year')
        
        users = load_users()
        
        if email in users:
            return render_template('signup.html', error='Email already registered')
        
        users[email] = {
            'password': password,
            'first_name': first_name,
            'last_name': last_name,
            'college': college,
            'branch': branch,
            'year': year,
            'created_at': datetime.now().isoformat()
        }
        
        save_users(users)
        
        return render_template('signup.html', 
                             success='Account created! Please login.')
    
    return render_template('signup.html')

@app.route('/quiz')
def quiz_page():
    """Quiz page"""
    return render_template('quiz.html')

@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction (same as app_college.py)"""
    if placement_model is None or package_model is None:
        return render_template('result.html', 
                             error="Models not trained yet.")
    
    try:
        # Get all form data (same as before)
        cgpa = float(request.form.get('cgpa', 7.0))
        attendance = float(request.form.get('attendance', 80))
        study_hours = float(request.form.get('study_hours', 6))
        math_marks = float(request.form.get('math_marks', 70))
        programming_marks = float(request.form.get('programming_marks', 75))
        theory_marks = float(request.form.get('theory_marks', 70))
        
        dsa_score = float(request.form.get('dsa_score', 60))
        coding_score = float(request.form.get('coding_score', 65))
        web_dev_score = float(request.form.get('web_dev_score', 50))
        
        java_prof = float(request.form.get('java_proficiency', 60))
        python_prof = float(request.form.get('python_proficiency', 65))
        cpp_prof = float(request.form.get('cpp_proficiency', 55))
        js_prof = float(request.form.get('javascript_proficiency', 50))
        
        frontend_skills = float(request.form.get('frontend_skills', 50))
        backend_skills = float(request.form.get('backend_skills', 55))
        database_skills = float(request.form.get('database_skills', 50))
        
        projects_count = int(request.form.get('projects_count', 3))
        github_repos = int(request.form.get('github_repos', 5))
        internships_count = int(request.form.get('internships_count', 1))
        
        leetcode_solved = int(request.form.get('leetcode_solved', 100))
        codechef_rating = int(request.form.get('codechef_rating', 1400))
        
        communication = float(request.form.get('communication', 6.5))
        aptitude = float(request.form.get('aptitude', 65))
        reasoning = float(request.form.get('reasoning', 65))
        
        mock_test = float(request.form.get('mock_test', 65))
        interview_prep_hours = float(request.form.get('interview_prep_hours', 50))
        
        # Calculate resume score
        resume_score = min(100, 
            20 + 0.15 * projects_count * 8 + 0.20 * internships_count * 20 +
            0.10 * (leetcode_solved / 6) + 0.05 * github_repos * 2
        )
        
        # Prepare input
        input_features = np.array([[
            study_hours, attendance, cgpa, math_marks, programming_marks, theory_marks,
            dsa_score, coding_score, web_dev_score, leetcode_solved, codechef_rating,
            java_prof, python_prof, cpp_prof, js_prof,
            frontend_skills, backend_skills, database_skills,
            projects_count, github_repos, internships_count,
            communication, aptitude, reasoning, mock_test, interview_prep_hours, resume_score
        ]])
        
        # Make predictions
        placement_probability = float(placement_model.predict(input_features)[0])
        expected_package = float(package_model.predict(input_features)[0])
        
        placement_probability = max(5, min(100, placement_probability))
        expected_package = max(3, min(30, expected_package))
        
        # Generate category
        if placement_probability >= 75:
            category = "Excellent"
            category_class = "excellent"
        elif placement_probability >= 60:
            category = "Good"
            category_class = "good"
        elif placement_probability >= 45:
            category = "Average"
            category_class = "average"
        else:
            category = "Needs Improvement"
            category_class = "poor"
        
        # Generate analysis (simplified for space)
        skills_analysis = [
            {'skill': 'DSA', 'level': 'Strong' if dsa_score >= 75 else 'Moderate' if dsa_score >= 50 else 'Weak',
             'icon': 'check-circle' if dsa_score >= 75 else 'info-circle' if dsa_score >= 50 else 'exclamation-triangle',
             'class': 'good' if dsa_score >= 75 else 'moderate' if dsa_score >= 50 else 'poor'},
            {'skill': 'Coding', 'level': 'Strong' if coding_score >= 75 else 'Moderate' if coding_score >= 50 else 'Weak',
             'icon': 'check-circle' if coding_score >= 75 else 'info-circle' if coding_score >= 50 else 'exclamation-triangle',
             'class': 'good' if coding_score >= 75 else 'moderate' if coding_score >= 50 else 'poor'},
            {'skill': 'Projects', 'level': 'Excellent' if projects_count >= 5 else 'Good' if projects_count >= 3 else 'Need More',
             'icon': 'check-circle' if projects_count >= 5 else 'info-circle' if projects_count >= 3 else 'exclamation-triangle',
             'class': 'good' if projects_count >= 5 else 'moderate' if projects_count >= 3 else 'poor'},
        ]
        
        roadmap = []
        if dsa_score < 70:
            roadmap.append({
                'priority': 'HIGH', 'area': 'DSA Skills', 'current': dsa_score, 'target': 85,
                'action': 'Solve 300+ LeetCode problems', 'timeline': '3-4 months',
                'resources': 'LeetCode, GeeksforGeeks'
            })
        
        companies = [
            {'name': 'Google', 'package': '25-45 LPA', 'level': 'Dream'},
            {'name': 'Microsoft', 'package': '20-42 LPA', 'level': 'Dream'},
            {'name': 'Amazon', 'package': '18-44 LPA', 'level': 'Dream'},
        ] if placement_probability >= 70 else [
            {'name': 'Infosys', 'package': '3.5-6 LPA', 'level': 'Mass Recruiter'},
            {'name': 'TCS', 'package': '3.6-7 LPA', 'level': 'Mass Recruiter'},
        ]
        
        time_allocation = {
            'DSA Practice': '3 hrs',
            'Development': '2 hrs',
            'Projects': '2 hrs',
            'Mock Tests': '1 hr'
        }
        
        strengths = ['Strong CGPA'] if cgpa >= 8 else []
        weaknesses = ['Improve DSA'] if dsa_score < 70 else []
        
        return render_template('result.html',
            year='3rd Year', branch='CSE', cgpa=cgpa,
            placement_probability=round(placement_probability, 2),
            expected_package=round(expected_package, 2),
            category=category, category_class=category_class,
            skills_analysis=skills_analysis, roadmap=roadmap,
            companies=companies, time_allocation=time_allocation,
            strengths=strengths, weaknesses=weaknesses,
            input_data={'DSA': dsa_score, 'Projects': projects_count}
        )
    
    except Exception as e:
        return render_template('result.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    print("=" * 80)
    print("ENHANCED COLLEGE PLACEMENT PREDICTOR")
    print("=" * 80)
    print("\n✓ Features: Login, Signup, Quiz, Prediction, Dashboard")
    print("✓ Open: http://127.0.0.1:5000/")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
