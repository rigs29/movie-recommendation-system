@echo off
echo Starting Movie Recommendation System...
call "%~dp0\.venv\Scripts\activate.bat"
streamlit run "%~dp0\app.py"
pause

