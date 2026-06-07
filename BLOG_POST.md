# Building a College Placement Predictor using Machine Learning

## How I Built an AI-Powered Career Intelligence Platform with Python, Flask, and Random Forest

![Cover Image - Add your project screenshot]

---

## 🎯 Introduction

As a computer science student, I wanted to build something that could help my fellow students in their placement journey. That's when I decided to create a **College Placement Predictor** - an AI-powered web application that predicts:

- **Placement Probability** (your chances of getting placed)
- **Expected Package** (salary estimation in LPA)
- **Personalized Roadmap** (what to improve and how)

**Tech Stack:** Python, Flask, Scikit-learn, Random Forest, Gradient Boosting

**🔗 GitHub:** https://github.com/Tpal41/Machine-Learning-Project

---

## 💡 Problem Statement

Every year, thousands of engineering students face uncertainty about their placement outcomes. Questions like:
- "Will I get placed?"
- "What package can I expect?"
- "What skills should I focus on?"

remain unanswered until the actual placement season.

**My Solution:** Use Machine Learning to predict placement outcomes based on comprehensive student data.

---

## 🛠️ Technical Architecture

### 1. Data Generation
I created a synthetic dataset of **500 students** with **31 features**:
- Academic: CGPA, marks, attendance
- Technical: DSA, coding, web development
- Languages: Java, Python, C++, JavaScript
- Experience: Projects, internships, GitHub
- Competitive: LeetCode, CodeChef ratings
- Soft Skills: Communication, aptitude

```python
# Sample data generation
student_data = {
    'CGPA': 8.5,
    'DSAScore': 75,
    'CodingScore': 80,
    'ProjectsCount': 5,
    'InternshipsCount': 2,
    'LeetCodeSolved': 200,
    # ... 25 more features
}
```

### 2. Machine Learning Models

#### Model 1: Placement Probability (Random Forest)
```python
from sklearn.ensemble import RandomForestRegressor

placement_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
placement_model.fit(X_train, y_placement)
```

#### Model 2: Package Prediction (Gradient Boosting)
```python
from sklearn.ensemble import GradientBoostingRegressor

package_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1
)
package_model.fit(X_train, y_package)
```

**Results:**
- Package Prediction Accuracy: **64% (R² = 0.64)**
- Mean Absolute Error: **1.41 LPA**

### 3. Web Application (Flask)
```python
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    features = extract_features(request.form)
    
    # Make predictions
    placement_prob = placement_model.predict([features])[0]
    package = package_model.predict([features])[0]
    
    # Generate roadmap
    roadmap = generate_personalized_roadmap(features)
    
    return render_template('result.html', 
                         placement=placement_prob,
                         package=package,
                         roadmap=roadmap)
```

---

## 🎨 Key Features

### 1. Dark Mode
Implemented a beautiful dark mode toggle that persists across sessions:
```javascript
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', 
        document.body.classList.contains('dark-mode') ? 'enabled' : 'disabled');
}
```

### 2. Interactive Quiz
10-question DSA and aptitude quiz to enhance prediction accuracy.

### 3. Personalized Roadmap
Based on predictions, the system generates:
- Skills to improve
- Time allocation (DSA, Development, Projects)
- Company recommendations
- Learning resources

---

## 📊 Model Performance

### Feature Importance
Top 5 features impacting placement:
1. **DSA Score** - 14.29%
2. **Projects Count** - 8.22%
3. **Coding Score** - 8.20%
4. **Resume Score** - 8.01%
5. **Aptitude Score** - 6.97%

### Insights
- Students with **DSA > 75** have **89% placement probability**
- **3+ internships** increase package by **~3 LPA**
- **200+ LeetCode** problems correlate with **dream company** placements

---

## 🚀 Deployment

Deployed on **Render.com** for free:
1. Push code to GitHub
2. Connect Render to repository
3. Configure build & start commands
4. Deploy!

**Build Command:**
```bash
pip install -r requirements.txt && python generate_dataset.py && python train_college_model.py
```

**Start Command:**
```bash
gunicorn app_enhanced:app
```

---

## 💭 Challenges Faced

### 1. Feature Engineering
Initially used only 5 features - results were poor (R² = 0.25). Expanded to 31 features → R² improved to 0.64.

### 2. Model Selection
Tried Linear Regression first - accuracy was low. Random Forest + Gradient Boosting combination worked best.

### 3. Data Quality
Synthetic data needed realistic correlations. Spent time ensuring DSA, projects, and internships had proper weightage.

---

## 📚 What I Learned

1. **Feature Engineering > Algorithm Selection**
   - More relevant features = better predictions
   
2. **Ensemble Methods Work**
   - Random Forest handled non-linear relationships well
   
3. **User Experience Matters**
   - Added dark mode, quiz, roadmap for better engagement
   
4. **Real-world Complexity**
   - Placement depends on 31+ factors, not just CGPA

---

## 🎯 Future Enhancements

- [ ] **Deep Learning** - Try neural networks for better accuracy
- [ ] **Real Data** - Collect actual placement data from colleges
- [ ] **More Features** - Add certifications, hackathons, open-source contributions
- [ ] **Company-Specific Predictions** - "What are my chances at Google?"
- [ ] **Mobile App** - React Native version
- [ ] **API** - RESTful API for third-party integrations

---

## 🔗 Try It Yourself

**GitHub:** https://github.com/Tpal41/Machine-Learning-Project

**Setup:**
```bash
git clone https://github.com/Tpal41/Machine-Learning-Project.git
cd Machine-Learning-Project
pip install -r requirements.txt
python generate_dataset.py
python train_college_model.py
python app_enhanced.py
```

Open: http://localhost:5000

---

## 📝 Conclusion

Building this project taught me:
- End-to-end ML pipeline (data → model → deployment)
- Flask web development
- Feature engineering importance
- User-centric design

**Most Important:** ML can provide valuable insights, but it's a tool for guidance, not a crystal ball. Your effort matters most! 💪

---

## 💬 Questions?

Leave a comment below or open an issue on GitHub!

**⭐ If you found this helpful, star the repo: https://github.com/Tpal41/Machine-Learning-Project**

---

**Tags:** #MachineLearning #Python #Flask #DataScience #CollegePlacement #RandomForest #AI #WebDevelopment

---

## About the Author

I'm a computer science student passionate about Machine Learning and Web Development. Follow me on GitHub for more projects!

**GitHub:** https://github.com/Tpal41
