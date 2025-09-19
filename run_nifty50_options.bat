@echo off
echo ========================================
echo    NIFTY 50 OPTIONS TRADING SYSTEM
echo ========================================
echo.
echo Choose an option:
echo.
echo 1. Run Basic Demo Script
echo 2. Run Enhanced Demo (Entry/SL/Exit)
echo 3. Run Intraday Trading Demo
echo 4. Run Enhanced CE & PE Demo
echo 5. Launch Streamlit Dashboard
echo 6. Install Dependencies
echo 7. Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo Running Basic Nifty 50 Options Trading Demo...
    echo.
    python nifty50_options_demo.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo Running Enhanced Demo with Entry Price, Stop Loss & Exit...
    echo.
    python nifty50_enhanced_demo.py
    pause
) else if "%choice%"=="3" (
    echo.
    echo Running Intraday Nifty 50 Options Trading Demo...
    echo.
    python intraday_nifty50_demo.py
    pause
) else if "%choice%"=="4" (
    echo.
    echo Running Enhanced CE & PE Intraday Trading Demo...
    echo.
    python ce_pe_intraday_demo.py
    pause
) else if "%choice%"=="5" (
    echo.
    echo Launching Streamlit Dashboard...
    echo.
    streamlit run nifty50_options_dashboard.py
    pause
) else if "%choice%"=="6" (
    echo.
    echo Installing Dependencies...
    echo.
    pip install -r requirements_options.txt
    pause
) else if "%choice%"=="7" (
    echo.
    echo Exiting...
    exit /b 0
) else (
    echo.
    echo Invalid choice. Please enter a number between 1 and 7.
    pause
    goto :eof
)
