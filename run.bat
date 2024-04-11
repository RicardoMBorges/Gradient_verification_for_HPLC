@echo off
setlocal

rem Define Python version and virtual environment directory
set "python_version=3.11"
set "venv_dir=gradient_verification_hplc"

rem Get the directory path where the batch file is located
set "script_dir=%~dp0"

rem Check if the virtual environment already exists in the script's directory
if exist "%script_dir%\%venv_dir%\Scripts\activate" (
    echo %venv_dir% already exists. Activating...
    call "%script_dir%\%venv_dir%\Scripts\activate"
) else (
    echo Creating virtual environment in "%script_dir%\%venv_dir%"...
    py -%python_version% -m venv "%script_dir%\%venv_dir%"
    call "%script_dir%\%venv_dir%\Scripts\activate"

    echo Installing the requirements...
    python -m pip install --upgrade pip
    python -m pip install -r "%script_dir%\requirements.txt"
    
    echo Installing Jupyter Notebook...
    python -m pip install jupyter notebook
    python -m pip install traitlets
)

rem Start Jupyter Notebook
call jupyter notebook

pause
