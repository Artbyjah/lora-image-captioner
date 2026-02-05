@echo off
echo ============================================
echo LoRA Image Captioner
echo ============================================
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

:: Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create a .env file with your API key:
    echo   ANTHROPIC_API_KEY=your_api_key_here
    echo.
    copy .env.example .env
    echo Created .env from template. Please edit it with your API key.
    pause
    exit /b
)

:: Run the app
echo.
echo Starting server...
echo Open http://localhost:5000 in your browser
echo Press Ctrl+C to stop the server
echo.
python app.py
