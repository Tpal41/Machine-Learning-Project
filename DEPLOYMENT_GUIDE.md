# 🚀 Deployment Guide - Render.com

## Deploy Your ML Project for FREE on Render

---

## 📋 Prerequisites

- ✅ GitHub account
- ✅ Project pushed to GitHub
- ✅ Render.com account (free tier)

---

## 🎯 Step-by-Step Deployment

### Step 1: Sign up on Render

1. Go to: **https://render.com**
2. Click **"Get Started"**
3. Sign up with **GitHub** (recommended)
4. Authorize Render to access your repositories

### Step 2: Create New Web Service

1. Click **"New +"** button (top-right)
2. Select **"Web Service"**
3. Connect your **GitHub account** if not already connected

### Step 3: Select Repository

1. Find: **`Machine-Learning-Project`**
2. Click **"Connect"**

### Step 4: Configure Service

Fill in the following details:

**Basic Settings:**
- **Name:** `college-placement-predictor` (or any name you want)
- **Region:** Choose closest to you
- **Branch:** `main`
- **Runtime:** `Python 3`

**Build Settings:**
- **Build Command:**
```bash
pip install -r requirements.txt && python generate_dataset.py && python train_college_model.py
```

- **Start Command:**
```bash
gunicorn app_enhanced:app
```

**Instance Type:**
- Select: **Free** (sufficient for this project)

### Step 5: Environment Variables (Optional)

Click **"Add Environment Variable"** if needed:
- `PYTHON_VERSION` = `3.11.0`
- `SECRET_KEY` = `your_secret_key_here`

### Step 6: Deploy!

1. Click **"Create Web Service"**
2. Wait 5-10 minutes for build to complete
3. Watch the logs for any errors

---

## ✅ Deployment Complete!

Your app will be live at:
```
https://college-placement-predictor.onrender.com
```
(Replace with your actual deployment URL)

---

## 📝 Post-Deployment Steps

### 1. Test Your App

Visit your deployment URL and test:
- ✅ Homepage loads
- ✅ Form submission works
- ✅ Predictions display correctly
- ✅ Dark mode toggle works

### 2. Update README with Live Link

Edit `README.md`:
```markdown
**🚀 Live Demo:** https://your-app.onrender.com
```

### 3. Update Footer in index.html

Add your deployment link:
```html
<a href="https://your-app.onrender.com" target="_blank">
    Visit Live Demo
</a>
```

### 4. Commit and Push Changes

```bash
git add .
git commit -m "Add deployment link"
git push origin main
```

Render will **auto-deploy** the updates!

---

## 🐛 Troubleshooting

### Issue 1: Build Fails
**Error:** `ModuleNotFoundError`

**Solution:**
Check `requirements.txt` includes all dependencies:
```bash
flask==2.3.3
gunicorn==21.2.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
```

### Issue 2: App Crashes on Start
**Error:** `Application failed to start`

**Solution:**
Check start command:
```bash
gunicorn app_enhanced:app
```
Make sure `app_enhanced.py` exists and has `app = Flask(__name__)`

### Issue 3: 404 Not Found
**Error:** Page not found

**Solution:**
Check routes in `app_enhanced.py`:
```python
@app.route('/')
def index():
    return render_template('index.html')
```

### Issue 4: Model Not Loading
**Error:** `Model file not found`

**Solution:**
Ensure build command trains model:
```bash
python train_college_model.py
```

---

## 💡 Pro Tips

### 1. Free Tier Limitations
- Render free tier **sleeps after 15 min of inactivity**
- First request may take 30-60 seconds to wake up
- Sufficient for portfolio/demo projects

### 2. Keep App Alive
Use a free uptime monitor:
- **UptimeRobot** - https://uptimerobot.com
- Pings your app every 5 minutes
- Keeps it from sleeping

### 3. Monitor Logs
- Click **"Logs"** tab in Render dashboard
- Check for errors
- Debug issues

### 4. Auto-Deploy on Git Push
- Render automatically deploys when you push to `main` branch
- No manual re-deployment needed!

---

## 🔗 Useful Links

- **Render Docs:** https://render.com/docs
- **Python on Render:** https://render.com/docs/deploy-flask
- **Troubleshooting:** https://render.com/docs/troubleshooting-deploys

---

## 📊 Deployment Checklist

Before deploying, ensure:

- [ ] `requirements.txt` is up to date
- [ ] `gunicorn` is in requirements
- [ ] Build command trains model
- [ ] Start command uses gunicorn
- [ ] All templates exist in `templates/` folder
- [ ] All static files in `static/` folder
- [ ] `.gitignore` excludes unnecessary files
- [ ] Code is tested locally
- [ ] README is updated
- [ ] GitHub repo is public (or grant Render access)

---

## 🎉 Success!

Your ML project is now live and accessible to the world!

**Share your deployment link:**
- Add to resume
- Share on LinkedIn
- Include in portfolio
- Show to recruiters

---

## 📞 Need Help?

- **Render Community:** https://community.render.com
- **GitHub Issues:** https://github.com/Tpal41/Machine-Learning-Project/issues
- **Documentation:** This guide!

---

**🌟 Don't forget to star the repo if this helped you!**

GitHub: https://github.com/Tpal41/Machine-Learning-Project
