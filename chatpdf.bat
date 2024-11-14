@echo off
cd /d D:\learnPython\chatpdf
call venv\Scripts\activate  :: Activates the virtual environment
streamlit run chatpdf.py  :: Runs the Streamlit application
pause  :: Keeps the window open after execution