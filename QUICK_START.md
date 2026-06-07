# Quick Start Guide - Student Performance Prediction

## 🚀 Fast Setup (5 Minutes)

### Step 1: Install Python
Make sure Python 3.8+ is installed:
```bash
python --version
```

### Step 2: Install Dependencies
```bash
cd StudentScorePrediction
pip install -r requirements.txt
```

### Step 3: Generate Dataset
```bash
python generate_dataset.py
```
✅ Creates: `dataset/student_data.csv` (200 records)

### Step 4: Train ML Model
```bash
python train_model.py
```
✅ Creates: `model/student_model.pkl`  
✅ Creates: Visualizations in `visualizations/` folder  
⏱️ Time: ~10-15 seconds

### Step 5: Run Web Application
```bash
python app.py
```
✅ Server starts at: http://127.0.0.1:5000/  
🌐 Open in browser and start predicting!

---

## 📊 Alternative: Use Jupyter Notebook

```bash
jupyter notebook
```
Open: `notebooks/analysis.ipynb`

---

## 🎯 Test the Application

### Sample Test Cases:

**Excellent Student:**
- Study Hours: 9
- Attendance: 95
- Previous Marks: 90
- Expected: ~94 marks (A+ grade)

**Average Student:**
- Study Hours: 5
- Attendance: 75
- Previous Marks: 65
- Expected: ~70 marks (B grade)

**Below Average:**
- Study Hours: 2.5
- Attendance: 60
- Previous Marks: 50
- Expected: ~55 marks (D grade)

---

## 📁 Project Structure Overview

```
StudentScorePrediction/
├── dataset/              # Generated data
├── model/                # Trained ML model
├── templates/            # HTML files
├── static/               # CSS files
├── visualizations/       # Generated plots
├── documentation/        # Reports, PPT, Viva
├── generate_dataset.py   # Creates data
├── train_model.py        # Trains model
├── app.py                # Web application
└── requirements.txt      # Dependencies
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Model not found"
**Solution:**
```bash
python train_model.py
```

### Issue: "Port already in use"
**Solution:** Change port in app.py:
```python
app.run(debug=True, port=5001)  # Change to 5001
```

### Issue: Flask not starting
**Solution:**
```bash
# Windows
set FLASK_APP=app.py
flask run

# Mac/Linux
export FLASK_APP=app.py
flask run
```

---

## 📚 Documentation Files

- **README.md** - Project overview
- **documentation/Report.md** - Complete detailed report
- **documentation/PPT_Content.md** - Presentation slides
- **documentation/Viva_Questions.md** - 33 Q&A for viva

---

## ✅ Checklist Before Viva

- [ ] Dataset generated (200 records)
- [ ] Model trained (R² > 0.90)
- [ ] Web app runs successfully
- [ ] All visualizations created
- [ ] Read viva questions
- [ ] Tested sample predictions
- [ ] Understood ML concepts
- [ ] Can explain code

---

## 🎓 For Presentation

1. Open web app: http://127.0.0.1:5000/
2. Open visualizations folder
3. Keep code files ready in editor
4. Have viva questions handy
5. Test live prediction during demo

---

## 💡 Pro Tips

✨ **Before Demo:**
- Clear browser cache
- Test on different inputs
- Keep backup of model file

✨ **During Viva:**
- Show live prediction
- Explain visualizations
- Walk through code
- Highlight ML usage

✨ **Important Files to Show:**
- train_model.py (ML pipeline)
- Correlation heatmap (EDA)
- Actual vs Predicted graph (results)
- app.py (deployment)

---

**Ready to Impress! 🚀**
