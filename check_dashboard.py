#!/usr/bin/env python3
"""
Dashboard Checker Script
Checks if Nifty 50 dashboard is running and opens it in browser
"""

import webbrowser
import time
import requests
import subprocess
import os

def check_url(url="http://localhost:8501"):
    """Check if URL is accessible"""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("🚀 Starting Nifty 50 Dashboard...")
    try:
        # Start dashboard in background
        subprocess.Popen([
            "python", "-m", "streamlit", "run", 
            "nifty50_dashboard.py", "--server.headless", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for dashboard to start
        print("⏳ Waiting for dashboard to start...")
        for i in range(20):
            time.sleep(1)
            if check_url():
                print("✅ Dashboard started successfully!")
                return True
            print(f"   Checking... ({i+1}/20)")
        
        print("❌ Dashboard failed to start within 20 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False

def main():
    """Main function"""
    url = "http://localhost:8501"
    
    print("🔍 Checking Nifty 50 Dashboard...")
    
    if check_url(url):
        print("✅ Dashboard is already running!")
        print(f"🌐 Opening {url} in browser...")
        webbrowser.open(url)
    else:
        print("❌ Dashboard is not running")
        print("🚀 Attempting to start dashboard...")
        
        if start_dashboard():
            print(f"🌐 Opening {url} in browser...")
            webbrowser.open(url)
        else:
            print("❌ Failed to start dashboard")
            print("\n💡 Manual steps:")
            print("1. Open a new terminal")
            print("2. Run: python -m streamlit run nifty50_dashboard.py")
            print("3. Wait for 'Local URL: http://localhost:8501' message")
            print("4. Open browser to http://localhost:8501")

if __name__ == "__main__":
    main()

