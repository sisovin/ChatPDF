@echo off
cd /d D:\learnPython\chatpdf
call venv\Scripts\activate  :: Activates the virtual environment
python -m unittest test_rag.py  :: Runs the test of RAG quality response
pause  :: Keeps the window open after execution
