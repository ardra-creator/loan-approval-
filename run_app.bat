@echo off
title Loan Approval Prediction App
echo Starting Loan Approval Prediction Web Application...
cd /d "C:\Users\ardra\.gemini\antigravity\scratch\loan-approval-prediction"
echo Opening browser at http://127.0.0.1:5000 in 2 seconds...
start "" timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000
"C:\Users\ardra\AppData\Local\Programs\Python\Python311\python.exe" app.py
pause
