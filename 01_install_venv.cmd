@ECHO OFF
rem Create virtual environment
python -m venv .venv
rem Active the created virtual environment
call .venv\Scripts\activate
python.exe -m pip install --upgrade pip

REM pip install -r requirements.txt

pip install ultralytics