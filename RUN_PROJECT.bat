@echo off
echo ============================================
echo Student Performance Prediction ML Project
echo ============================================
echo.

echo Step 1: Generating Dataset...
python generate_dataset.py
if errorlevel 1 goto error
echo.

echo Step 2: Training Model (this may take 10-15 seconds)...
python train_model.py
if errorlevel 1 goto error
echo.

echo Step 3: Starting Web Application...
echo.
echo ============================================
echo   Web app will start at:
echo   http://127.0.0.1:5000/
echo.
echo   Press CTRL+C to stop
echo ============================================
echo.
python app.py
goto end

:error
echo.
echo ERROR: Something went wrong!
echo Please check if Python and dependencies are installed.
echo Run: pip install -r requirements.txt
pause
goto end

:end
