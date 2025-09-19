@echo off
echo Starting Enhanced Nifty 50 Intraday Trading Dashboard...
echo.
echo This dashboard provides:
echo - Clear BUY/SELL signals for Nifty 50 stocks
echo - Entry and exit points with stop-loss levels
echo - Real-time technical analysis
echo - Market timing recommendations
echo.
echo Opening in browser...
echo.
py -m streamlit run intraday_dashboard.py
pause
