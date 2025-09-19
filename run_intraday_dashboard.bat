@echo off
echo ========================================
echo   Nifty 50 Intraday Trading Dashboard
echo ========================================
echo.
echo Starting Intraday Trading Dashboard...
echo.
echo This will open the dashboard in your browser
echo Dashboard URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.
python -m streamlit run intraday_dashboard.py --server.headless true
pause
