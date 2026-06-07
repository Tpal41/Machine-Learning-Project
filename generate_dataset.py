"""
College Student Performance & Placement Prediction Dataset Generator
====================================================================
This script generates a comprehensive dataset for college students focusing on:
- Academic performance (CGPA, marks)
- Technical skills (DSA, Coding, Development)
- Projects and internships
- Placement preparation
- Communication and soft skills

Target predictions:
- CGPA improvement
- Placement probability
- Expected package (LPA)
- Skill gap analysis
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of student records
num_students = 500

print("=" * 80)
print("COLLEGE STUDENT PLACEMENT PREDICTION - DATASET GENERATION")
print("=" * 80)
print(f"\nGenerating dataset with {num_students} college student records...\n")

# Initialize lists
all_data = []

# Generate college student data
for i in range(num_students):
    # Student year
    year = np.random.choice(['2nd_Year', '3rd_Year', '4th_Year'], p=[0.25, 0.35, 0.40])
    branch = np.random.choice(['CSE', 'IT', 'ECE', 'EEE', 'Mechanical'], p=[0.35, 0.25, 0.20, 0.10, 0.10])
    
    # Academic features
    study_hours = np.random.uniform(3, 15)  # Daily study hours
    attendance = np.random.uniform(65, 100)  # Attendance percentage
    cgpa = np.random.uniform(5.5, 10.0)  # Current CGPA
    
    # Previous semester marks
    math_marks = np.random.uniform(50, 100)
    programming_marks = np.random.uniform(45, 100)
    theory_marks = np.random.uniform(55, 100)
    
    # Technical skills (0-100 scale)
    dsa_score = np.random.uniform(20, 100)  # Data Structures & Algorithms
    coding_score = np.random.uniform(25, 100)  # Overall coding
    web_dev_score = np.random.uniform(15, 100)  # Web development
    
    # Competitive programming
    leetcode_solved = np.random.randint(0, 600)
    codechef_rating = np.random.randint(1000, 2500)
    
    # Programming languages (0-100 proficiency)
    java_proficiency = np.random.uniform(20, 100)
    python_proficiency = np.random.uniform(30, 100)
    cpp_proficiency = np.random.uniform(15, 100)
    javascript_proficiency = np.random.uniform(10, 100)
    
    # Development skills
    frontend_skills = np.random.uniform(10, 100)
    backend_skills = np.random.uniform(10, 100)
    database_skills = np.random.uniform(15, 100)
    
    # Projects and experience
    projects_count = np.random.randint(0, 12)
    github_repos = np.random.randint(0, 30)
    internships_count = np.random.randint(0, 4)
    
    # Soft skills
    communication_skill = np.random.uniform(3, 10)  # 1-10 scale
    aptitude_score = np.random.uniform(30, 100)  # Quantitative/Logical
    reasoning_score = np.random.uniform(35, 100)
    
    # Preparation metrics
    mock_test_score = np.random.uniform(40, 100)
    interview_prep_hours = np.random.uniform(0, 200)  # Total hours
    
    # Resume quality (calculated)
    resume_score = min(100, 
        20 + 
        0.15 * projects_count * 8 +
        0.20 * internships_count * 20 +
        0.10 * (leetcode_solved / 6) +
        0.05 * github_repos * 2 +
        np.random.uniform(-10, 10)
    )
    
    # CGPA improvement prediction
    cgpa_improvement_potential = min(10, cgpa + np.random.uniform(-0.3, 0.8))
    
    # Placement probability calculation (sophisticated formula)
    placement_prob = min(100, max(0,
        # Academic weightage (30%)
        0.08 * cgpa * 10 +
        0.05 * attendance * 0.3 +
        0.05 * ((math_marks + programming_marks + theory_marks) / 3) * 0.17 +
        
        # Technical skills weightage (35%)
        0.15 * dsa_score +
        0.10 * coding_score +
        0.05 * web_dev_score +
        0.05 * ((java_proficiency + python_proficiency + cpp_proficiency) / 3) * 0.1 +
        
        # Experience weightage (20%)
        0.08 * min(100, projects_count * 12) +
        0.12 * min(100, internships_count * 25) +
        
        # Soft skills weightage (15%)
        0.08 * communication_skill * 10 +
        0.07 * aptitude_score +
        
        # Random variance
        np.random.normal(0, 8)
    ))
    
    # Expected package calculation (in LPA)
    base_package = 3.5
    
    expected_package = max(3, 
        base_package +
        0.55 * cgpa +  # CGPA impact
        0.025 * dsa_score +  # DSA skills
        0.020 * coding_score +  # Coding skills
        0.35 * projects_count +  # Projects
        1.2 * internships_count +  # Internships (high impact)
        0.15 * communication_skill +  # Communication
        0.01 * leetcode_solved +  # Competitive programming
        0.002 * codechef_rating +  # Rating
        np.random.uniform(-0.5, 2.5)  # Market variance
    )
    
    # Job role prediction (based on skills)
    if dsa_score > 75 and coding_score > 75:
        preferred_role = 'SDE'
    elif web_dev_score > 70 and (frontend_skills > 65 or backend_skills > 65):
        preferred_role = 'Full_Stack_Developer'
    elif data_science_interest := (python_proficiency > 70 and np.random.random() > 0.7):
        preferred_role = 'Data_Analyst'
    else:
        preferred_role = 'Software_Engineer'
    
    # Add to dataset
    all_data.append({
        'StudentID': f'STU{i+1:04d}',
        'Year': year,
        'Branch': branch,
        'StudyHours': study_hours,
        'Attendance': attendance,
        'CGPA': cgpa,
        'MathMarks': math_marks,
        'ProgrammingMarks': programming_marks,
        'TheoryMarks': theory_marks,
        'DSAScore': dsa_score,
        'CodingScore': coding_score,
        'WebDevScore': web_dev_score,
        'LeetCodeSolved': leetcode_solved,
        'CodeChefRating': codechef_rating,
        'JavaProficiency': java_proficiency,
        'PythonProficiency': python_proficiency,
        'CppProficiency': cpp_proficiency,
        'JavaScriptProficiency': javascript_proficiency,
        'FrontendSkills': frontend_skills,
        'BackendSkills': backend_skills,
        'DatabaseSkills': database_skills,
        'ProjectsCount': projects_count,
        'GitHubRepos': github_repos,
        'InternshipsCount': internships_count,
        'CommunicationSkill': communication_skill,
        'AptitudeScore': aptitude_score,
        'ReasoningScore': reasoning_score,
        'MockTestScore': mock_test_score,
        'InterviewPrepHours': interview_prep_hours,
        'ResumeScore': resume_score,
        'PreferredRole': preferred_role,
        # Target variables
        'CGPAImprovement': cgpa_improvement_potential,
        'PlacementProbability': placement_prob,
        'ExpectedPackage': expected_package
    })

# Create DataFrame
df = pd.DataFrame(all_data)

# Round numeric values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col in ['StudentID', 'LeetCodeSolved', 'CodeChefRating', 'ProjectsCount', 'GitHubRepos', 'InternshipsCount']:
        df[col] = df[col].astype(int) if col != 'StudentID' else df[col]
    else:
        df[col] = np.round(df[col], 2)

# Add some missing values (realistic scenario) - 4%
missing_count = int(0.04 * len(df))
missing_indices = np.random.choice(df.index, size=missing_count, replace=False)
missing_columns = ['Attendance', 'MockTestScore', 'CommunicationSkill', 'WebDevScore']
for idx in missing_indices:
    col = np.random.choice(missing_columns)
    df.loc[idx, col] = np.nan

print("\n" + "=" * 80)
print("Dataset Statistics:")
print("-" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("Category Distribution:")
print("-" * 80)
print("\nYear Distribution:")
print(df['Year'].value_counts())
print("\nBranch Distribution:")
print(df['Branch'].value_counts())

# Create dataset directory
os.makedirs('dataset', exist_ok=True)

# Save to CSV
output_path = 'dataset/college_students_data.csv'
df.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("✓ Dataset successfully saved to: " + output_path)
print(f"✓ Total records: {len(df)}")
print(f"✓ Features: {len(df.columns) - 3} (excluding target variables)")
print(f"✓ Target variables: CGPAImprovement, PlacementProbability, ExpectedPackage")

print("\nKey Features:")
print("  • Academic: CGPA, Marks, Attendance")
print("  • Technical: DSA, Coding, Web Dev, Languages")
print("  • Experience: Projects, Internships, GitHub")
print("  • Soft Skills: Communication, Aptitude, Reasoning")
print("  • Preparation: LeetCode, Mock tests, Interview prep")

print("\nFirst 5 records preview:")
print("-" * 80)
print(df[['StudentID', 'Year', 'CGPA', 'DSAScore', 'ProjectsCount', 'PlacementProbability', 'ExpectedPackage']].head())

print("\n" + "=" * 80)
print("COLLEGE STUDENT DATASET GENERATION COMPLETED!")
print("=" * 80)
