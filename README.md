# 🎓 College Placement Predictor - AI-Powered Career Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7)](https://render.com)

> **Advanced Machine Learning system** that predicts college placement probability and expected package using Random Forest and Gradient Boosting algorithms.

[🚀 Live Demo](#) | [📖 Documentation](#features) | [🎥 Video Demo](#)

---

## 📊 Project Overview

This project leverages machine learning to predict:
- **Placement Probability** (0-100%)
- **Expected Package** (in LPA)
- **Personalized Roadmap** for improvement
- **Skill Gap Analysis**
- **Company Recommendations**

### 🎯 Key Metrics
- **Dataset:** 500 engineering students
- **Features:** 31 comprehensive parameters
- **Models:** Random Forest + Gradient Boosting
- **Accuracy:** 64% for package prediction
- **Web Framework:** Flask with responsive UI

---

## ✨ Features

### 🤖 Machine Learning
- **Dual ML Models:**
  - Random Forest Regressor for placement probability
  - Gradient Boosting Regressor for package prediction
- **Comprehensive Feature Engineering:**
  - Academic: CGPA, Marks, Attendance
  - Technical: DSA, Coding, Web Development
  - Languages: Java, Python, C++, JavaScript
  - Experience: Projects, Internships, GitHub
  - Competitive: LeetCode, CodeChef ratings
  - Soft Skills: Communication, Aptitude

### 🎨 User Interface
- **Dark Mode Toggle** - Switch between light and dark themes
- **Responsive Design** - Works on all devices
- **Interactive Quiz** - 10-question assessment
- **Real-time Validation** - Input feedback
- **Beautiful Visualizations** - Charts and graphs

### 🔐 User Management
- **Authentication System** - Login/Signup
- **Session Management** - Secure user tracking
- **Profile Dashboard** - Track your progress

### 📈 Predictions & Analytics
- **Placement Probability** - AI-calculated success rate
- **Package Prediction** - Expected salary estimation
- **Skills Analysis** - Strengths and weaknesses
- **Personalized Roadmap** - Step-by-step improvement plan
- **Time Allocation** - Daily study schedule
- **Company Recommendations** - Based on your profile

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **Pandas & NumPy** - Data processing
- **Joblib** - Model persistence

### Frontend
- **HTML5 & CSS3**
- **JavaScript** - Interactive features
- **Font Awesome** - Icons
- **Responsive Design** - Mobile-first approach

### Machine Learning
- **Random Forest Regressor** - Placement prediction
- **Gradient Boosting Regressor** - Package prediction
- **Feature Engineering** - 31 engineered features
- **Data Preprocessing** - Handling missing values, outliers

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
Git
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/college-placement-predictor.git
cd college-placement-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate dataset**
```bash
python generate_dataset.py
```

4. **Train ML models**
```bash
python train_college_model.py
```

5. **Run the application**
```bash
python app_enhanced.py
```

6. **Open browser**
```
http://127.0.0.1:5000/
```

---

## 📁 Project Structure

```
college-placement-predictor/
│
├── 📱 Frontend
│   ├── templates/
│   │   ├── index.html          # Main prediction form
│   │   ├── result.html         # Results & roadmap
│   │   ├── login.html          # User login
│   │   ├── signup.html         # User registration
│   │   └── quiz.html           # Interactive quiz
│   └── static/
│       └── style.css           # Styling with dark mode
│
├── 🤖 Backend
│   ├── app_enhanced.py         # Flask app with authentication
│   ├── app_college.py          # Alternative simpler app
│   ├── generate_dataset.py    # Dataset generation
│   └── train_college_model.py # ML model training
│
├── 📊 Data & Models
│   ├── dataset/
│   │   └── college_students_data.csv
│   └── model/
│       └── college_placement_model.pkl
│
├── 📄 Configuration
│   ├── requirements.txt        # Python dependencies
│   ├── render.yaml            # Render deployment config
│   ├── .gitignore             # Git ignore rules
│   └── README.md              # This file
│
└── 📚 Documentation
    └── [Additional docs]
```

---

## 🎮 Usage Guide

### 1️⃣ **Sign Up / Login**
- Create account or login
- Or skip as guest

### 2️⃣ **Take Quiz (Optional)**
- Answer 10 quick questions
- Tests DSA, Programming, Aptitude
- Enhances prediction accuracy

### 3️⃣ **Fill Your Profile**
- **Academic Details:** CGPA, marks, attendance
- **Technical Skills:** DSA, coding, web development
- **Programming Languages:** Java, Python, C++, JS
- **Experience:** Projects, internships, GitHub
- **Competitive Programming:** LeetCode, CodeChef
- **Soft Skills:** Communication, aptitude

### 4️⃣ **Get Predictions**
- **Placement Probability:** Your success rate
- **Expected Package:** Salary estimation
- **Skills Analysis:** Strengths & weaknesses
- **Personalized Roadmap:** Step-by-step plan
- **Company Recommendations:** Target companies
- **Time Allocation:** Daily schedule

---

## 🌐 Deployment on Render

### Step-by-Step Guide

1. **Create Render Account**
   - Go to [Render.com](https://render.com)
   - Sign up with GitHub

2. **Connect Repository**
   - Click "New +" → "Web Service"
   - Connect your GitHub repo

3. **Configure Service**
   - **Name:** `college-placement-predictor`
   - **Environment:** `Python 3`
   - **Build Command:** 
     ```bash
     pip install -r requirements.txt && python generate_dataset.py && python train_college_model.py
     ```
   - **Start Command:** 
     ```bash
     gunicorn app_enhanced:app
     ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes
   - Your app will be live!

### Environment Variables (Optional)
```
PYTHON_VERSION=3.11.0
SECRET_KEY=your_secret_key_here
```

---

## 📊 Model Performance

### Placement Probability Model
- **Algorithm:** Random Forest Regressor
- **Features:** 27 engineered features
- **Training Samples:** 400
- **Testing Samples:** 100

### Package Prediction Model
- **Algorithm:** Gradient Boosting Regressor
- **R² Score:** 0.64 (64% accuracy)
- **MAE:** 1.41 LPA
- **RMSE:** 1.76 LPA

### Top Features by Importance
1. **DSA Score** - 14.29%
2. **Projects Count** - 8.22%
3. **Coding Score** - 8.20%
4. **Resume Score** - 8.01%
5. **Aptitude Score** - 6.97%

---

## 🎨 Screenshots

### Home Page
![Home Page](screenshots/home.png)

### Dark Mode
![Dark Mode](screenshots/dark-mode.png)

### Results Dashboard
![Results](screenshots/results.png)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create your feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

---

## 📝 Future Enhancements

- [ ] Add more ML algorithms (Neural Networks, XGBoost)
- [ ] Implement user dashboard with history
- [ ] Add email notifications for predictions
- [ ] Create mobile app (React Native)
- [ ] Add more features (certifications, hackathons)
- [ ] Implement A/B testing for models
- [ ] Add data visualization dashboard
- [ ] Deploy on AWS/Azure
- [ ] Add API endpoints
- [ ] Implement caching for faster predictions

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)
- Portfolio: [your-website.com](https://your-website.com)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Scikit-learn for ML algorithms
- Flask for web framework
- Render for free hosting
- Font Awesome for icons
- All open-source contributors

---

## 📞 Support

If you found this project helpful, please ⭐ star the repository!

For questions or issues:
- 📧 Email: your.email@example.com
- 💬 Open an [Issue](https://github.com/YOUR_USERNAME/college-placement-predictor/issues)
- 📱 LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

---

## 🔗 Links

- **Live Demo:** [Deployed on Render](#)
- **Blog Post:** [Read detailed article](#)
- **Video Tutorial:** [Watch on YouTube](#)
- **Documentation:** [Full Docs](#)

---

<div align="center">

**Made with ❤️ and Python**

⭐ Star this repo if you found it useful! ⭐

</div>
