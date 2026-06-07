"""
College Placement Prediction - Flask Web Application
===================================================
Advanced placement prediction system for engineering students with:
- Placement probability prediction
- Package prediction
- Personalized roadmap generation
- Skill gap analysis
- Company recommendations
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

# Initialize Flask application
app = Flask(__name__)

# Try to load trained model (will create if not exists)
MODEL_PATH = 'model/college_placement_model.pkl'

try:
    model_data = joblib.load(MODEL_PATH)
    placement_model = model_data['placement_model']
    package_model = model_data['package_model']
    print("✓ Machine Learning models loaded successfully!")
    print(f"  - Placement Model R² Score: {model_data.get('placement_r2', 'N/A')}")
    print(f"  - Package Model R² Score: {model_data.get('package_r2', 'N/A')}")
except:
    placement_model = None
    package_model = None
    print("⚠ Models not found. Please run 'python train_college_model.py' first.")

@app.route('/')
def home():
    """Render home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request and generate results"""
    
    if placement_model is None or package_model is None:
        return render_template('result.html', 
                             error="Models not trained yet. Please run training script first.")
    
    try:
        # Extract form data
        year = request.form.get('year', '3rd_Year')
        branch = request.form.get('branch', 'CSE')
        
        # Academic data
        cgpa = float(request.form.get('cgpa', 7.0))
        attendance = float(request.form.get('attendance', 80))
        study_hours = float(request.form.get('study_hours', 6))
        math_marks = float(request.form.get('math_marks', 70))
        programming_marks = float(request.form.get('programming_marks', 75))
        theory_marks = float(request.form.get('theory_marks', 70))
        
        # Technical skills
        dsa_score = float(request.form.get('dsa_score', 60))
        coding_score = float(request.form.get('coding_score', 65))
        web_dev_score = float(request.form.get('web_dev_score', 50))
        
        # Programming languages
        java_prof = float(request.form.get('java_proficiency', 60))
        python_prof = float(request.form.get('python_proficiency', 65))
        cpp_prof = float(request.form.get('cpp_proficiency', 55))
        js_prof = float(request.form.get('javascript_proficiency', 50))
        
        # Development skills
        frontend_skills = float(request.form.get('frontend_skills', 50))
        backend_skills = float(request.form.get('backend_skills', 55))
        database_skills = float(request.form.get('database_skills', 50))
        
        # Experience
        projects_count = int(request.form.get('projects_count', 3))
        github_repos = int(request.form.get('github_repos', 5))
        internships_count = int(request.form.get('internships_count', 1))
        
        # Competitive programming
        leetcode_solved = int(request.form.get('leetcode_solved', 100))
        codechef_rating = int(request.form.get('codechef_rating', 1400))
        
        # Soft skills
        communication = float(request.form.get('communication', 6.5))
        aptitude = float(request.form.get('aptitude', 65))
        reasoning = float(request.form.get('reasoning', 65))
        
        # Preparation
        mock_test = float(request.form.get('mock_test', 65))
        interview_prep_hours = float(request.form.get('interview_prep_hours', 50))
        
        # Calculate derived features
        overall_avg = (math_marks + programming_marks + theory_marks) / 3
        lang_avg = (java_prof + python_prof + cpp_prof + js_prof) / 4
        dev_avg = (frontend_skills + backend_skills + database_skills) / 3
        
        # Calculate resume score
        resume_score = min(100, 
            20 + 0.15 * projects_count * 8 + 0.20 * internships_count * 20 +
            0.10 * (leetcode_solved / 6) + 0.05 * github_repos * 2
        )
        
        # Prepare input for ML model (matching training features)
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
        
        # Clamp values to realistic ranges
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
        
        # Skill analysis
        skills_analysis = analyze_skills(dsa_score, coding_score, web_dev_score, projects_count, internships_count, cgpa, communication)
        
        # Generate roadmap
        roadmap = generate_roadmap(placement_probability, dsa_score, coding_score, projects_count, internships_count, communication, expected_package)
        
        # Company recommendations
        companies = recommend_companies(placement_probability, expected_package, cgpa, dsa_score)
        
        # Time allocation
        time_allocation = calculate_time_allocation(dsa_score, coding_score, web_dev_score, projects_count)
        
        # Strengths and weaknesses
        strengths, weaknesses = identify_strengths_weaknesses({
            'cgpa': cgpa, 'dsa': dsa_score, 'coding': coding_score,
            'projects': projects_count, 'internships': internships_count,
            'communication': communication, 'leetcode': leetcode_solved
        })
        
        # Return comprehensive result
        return render_template('result.html',
            # Basic info
            year=year,
            branch=branch,
            cgpa=cgpa,
            
            # Predictions
            placement_probability=round(placement_probability, 2),
            expected_package=round(expected_package, 2),
            category=category,
            category_class=category_class,
            
            # Analysis
            skills_analysis=skills_analysis,
            roadmap=roadmap,
            companies=companies,
            time_allocation=time_allocation,
            strengths=strengths,
            weaknesses=weaknesses,
            
            # Input summary
            input_data={
                'DSA Score': dsa_score,
                'Coding Score': coding_score,
                'Projects': projects_count,
                'Internships': internships_count,
                'LeetCode Solved': leetcode_solved,
                'Communication': communication
            }
        )
    
    except Exception as e:
        return render_template('result.html', error=f"Error processing request: {str(e)}")

def analyze_skills(dsa, coding, web_dev, projects, internships, cgpa, communication):
    """Analyze technical and soft skills"""
    analysis = []
    
    if dsa >= 75:
        analysis.append({'skill': 'DSA', 'level': 'Strong', 'icon': 'check-circle', 'class': 'good'})
    elif dsa >= 50:
        analysis.append({'skill': 'DSA', 'level': 'Moderate', 'icon': 'info-circle', 'class': 'moderate'})
    else:
        analysis.append({'skill': 'DSA', 'level': 'Weak', 'icon': 'exclamation-triangle', 'class': 'poor'})
    
    if coding >= 75:
        analysis.append({'skill': 'Coding', 'level': 'Strong', 'icon': 'check-circle', 'class': 'good'})
    elif coding >= 50:
        analysis.append({'skill': 'Coding', 'level': 'Moderate', 'icon': 'info-circle', 'class': 'moderate'})
    else:
        analysis.append({'skill': 'Coding', 'level': 'Weak', 'icon': 'exclamation-triangle', 'class': 'poor'})
    
    if projects >= 5:
        analysis.append({'skill': 'Projects', 'level': 'Excellent', 'icon': 'check-circle', 'class': 'good'})
    elif projects >= 3:
        analysis.append({'skill': 'Projects', 'level': 'Good', 'icon': 'info-circle', 'class': 'moderate'})
    else:
        analysis.append({'skill': 'Projects', 'level': 'Need More', 'icon': 'exclamation-triangle', 'class': 'poor'})
    
    if cgpa >= 8.0:
        analysis.append({'skill': 'Academics', 'level': 'Strong', 'icon': 'check-circle', 'class': 'good'})
    elif cgpa >= 7.0:
        analysis.append({'skill': 'Academics', 'level': 'Good', 'icon': 'info-circle', 'class': 'moderate'})
    else:
        analysis.append({'skill': 'Academics', 'level': 'Needs Work', 'icon': 'exclamation-triangle', 'class': 'poor'})
    
    return analysis

def generate_roadmap(placement_prob, dsa, coding, projects, internships, communication, package):
    """Generate personalized improvement roadmap"""
    roadmap = []
    
    if dsa < 70:
        roadmap.append({
            'priority': 'HIGH',
            'area': 'Data Structures & Algorithms',
            'current': dsa,
            'target': 85,
            'action': 'Solve 300+ LeetCode problems (Easy: 100, Medium: 150, Hard: 50)',
            'timeline': '3-4 months',
            'resources': 'LeetCode, GeeksforGeeks, InterviewBit'
        })
    
    if coding < 75:
        roadmap.append({
            'priority': 'HIGH',
            'area': 'Problem Solving & Coding',
            'current': coding,
            'target': 85,
            'action': 'Practice daily coding, participate in contests',
            'timeline': '2-3 months',
            'resources': 'CodeChef, Codeforces, HackerRank'
        })
    
    if projects < 4:
        roadmap.append({
            'priority': 'MEDIUM',
            'area': 'Project Development',
            'current': projects,
            'target': 5,
            'action': 'Build 2-3 full-stack projects with deployment',
            'timeline': '2 months',
            'resources': 'GitHub, YouTube tutorials, FreeCodeCamp'
        })
    
    if internships < 2:
        roadmap.append({
            'priority': 'MEDIUM',
            'area': 'Internship Experience',
            'current': internships,
            'target': 2,
            'action': 'Apply to 50+ internships on Internshala, LinkedIn',
            'timeline': '1-2 months application period',
            'resources': 'Internshala, LinkedIn, AngelList'
        })
    
    if communication < 7:
        roadmap.append({
            'priority': 'LOW',
            'area': 'Communication Skills',
            'current': communication,
            'target': 8,
            'action': 'Practice mock interviews, public speaking',
            'timeline': '1-2 months',
            'resources': 'Pramp, InterviewBit mock interviews'
        })
    
    return roadmap

def recommend_companies(placement_prob, package, cgpa, dsa):
    """Recommend companies based on profile"""
    companies = []
    
    if placement_prob >= 70 and cgpa >= 8.0 and dsa >= 75:
        companies.extend([
            {'name': 'Google', 'package': '25-45 LPA', 'level': 'Dream'},
            {'name': 'Microsoft', 'package': '20-42 LPA', 'level': 'Dream'},
            {'name': 'Amazon', 'package': '18-44 LPA', 'level': 'Dream'},
        ])
    
    if placement_prob >= 60 and cgpa >= 7.5:
        companies.extend([
            {'name': 'Flipkart', 'package': '15-24 LPA', 'level': 'Super Dream'},
            {'name': 'Adobe', 'package': '18-30 LPA', 'level': 'Super Dream'},
            {'name': 'Oracle', 'package': '12-20 LPA', 'level': 'Super Dream'},
        ])
    
    if placement_prob >= 45:
        companies.extend([
            {'name': 'Infosys', 'package': '3.5-6 LPA', 'level': 'Mass Recruiter'},
            {'name': 'TCS', 'package': '3.6-7 LPA', 'level': 'Mass Recruiter'},
            {'name': 'Wipro', 'package': '3.5-6.5 LPA', 'level': 'Mass Recruiter'},
            {'name': 'Cognizant', 'package': '4-6 LPA', 'level': 'Mass Recruiter'},
        ])
    
    return companies[:6]  # Return top 6

def calculate_time_allocation(dsa, coding, web_dev, projects):
    """Calculate how much time to spend on each area"""
    total_hours = 8  # Daily preparation hours
    
    # Calculate weightage based on weaknesses
    dsa_weight = 100 - dsa
    coding_weight = 100 - coding
    dev_weight = 100 - web_dev
    project_weight = (5 - projects) * 15 if projects < 5 else 0
    
    total_weight = dsa_weight + coding_weight + dev_weight + project_weight
    
    if total_weight == 0:
        return {
            'DSA': '3 hrs', 'Development': '2 hrs',
            'Projects': '2 hrs', 'Mock Tests': '1 hr'
        }
    
    allocation = {
        'DSA Practice': f"{(dsa_weight / total_weight * total_hours):.1f} hrs",
        'Coding Problems': f"{(coding_weight / total_weight * total_hours):.1f} hrs",
        'Development': f"{(dev_weight / total_weight * total_hours):.1f} hrs",
        'Projects': f"{(project_weight / total_weight * total_hours):.1f} hrs",
    }
    
    return allocation

def identify_strengths_weaknesses(data):
    """Identify top strengths and weaknesses"""
    strengths = []
    weaknesses = []
    
    if data['cgpa'] >= 8.0:
        strengths.append('Strong academic performance')
    elif data['cgpa'] < 7.0:
        weaknesses.append('CGPA needs improvement')
    
    if data['dsa'] >= 75:
        strengths.append('Excellent DSA skills')
    elif data['dsa'] < 50:
        weaknesses.append('Weak in Data Structures & Algorithms')
    
    if data['projects'] >= 5:
        strengths.append('Good project portfolio')
    elif data['projects'] < 3:
        weaknesses.append('Insufficient projects')
    
    if data['internships'] >= 2:
        strengths.append('Valuable internship experience')
    elif data['internships'] == 0:
        weaknesses.append('No internship experience')
    
    if data['leetcode'] >= 200:
        strengths.append('Active in competitive programming')
    elif data['leetcode'] < 50:
        weaknesses.append('Low problem-solving practice')
    
    if data['communication'] >= 8:
        strengths.append('Excellent communication skills')
    elif data['communication'] < 6:
        weaknesses.append('Communication skills need work')
    
    return strengths[:4], weaknesses[:4]

if __name__ == '__main__':
    print("=" * 80)
    print("COLLEGE PLACEMENT PREDICTOR - WEB APPLICATION")
    print("=" * 80)
    print("\n✓ Starting Flask server...")
    if placement_model and package_model:
        print("✓ ML Models loaded")
    else:
        print("⚠ Models not loaded - please train first")
    print("\n➜ Open your browser and go to: http://127.0.0.1:5000/")
    print("➜ Press CTRL+C to stop the server\n")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
