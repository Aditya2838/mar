
"""
This script demonstrates how to pull live Nifty 50 options data into a Google Sheet, and set up prebuilt indicator logic for technical indicators (RSI, EMA, VWAP) and conditional formatting for trading signals.

1. Live data is fetched from the NSE website using web scraping.
2. The script writes the data to a Google Sheet, and inserts formulas for RSI, EMA, and VWAP using prebuilt indicator logic.
3. Conditional formatting and alerts are set up in the sheet for trading signals (e.g., RSI overbought/oversold, EMA crossovers).

Note: 
- You must set up a Google Cloud project and download your service account credentials as 'credentials.json'.
- After running this script, open your Google Sheet and review the formulas and conditional formatting.
"""

import requests
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import webbrowser
from pathlib import Path

import time

# NIFTY lot size helper (checks file then env)
def get_lot_size() -> int:
    try:
        from pathlib import Path as _Path
        lot_file = _Path('lot_size.txt')
        if lot_file.exists():
            txt = lot_file.read_text(encoding='utf-8').strip()
            val = int(txt)
            if val > 0:
                return val
    except Exception:
        pass
    try:
        val = int(os.getenv('NIFTY_LOT_SIZE', '75'))
        if val > 0:
            return val
    except Exception:
        pass
    return 75

# Zerodha Kite settings via environment
KITE_API_KEY = os.getenv('KITE_API_KEY')
KITE_ACCESS_TOKEN = os.getenv('KITE_ACCESS_TOKEN')

def compute_intraday_vwap(candles):
    """
    candles: list of dicts/objects with keys: 'close' or 'c', 'volume' or 'v', and optionally 'high','low'.
    VWAP = sum(typical_price * volume) / sum(volume)
    typical_price ~ close as approximation when H/L not available.
    Returns (vwap_value, last_price)
    """
    total_pv = 0.0
    total_v = 0.0
    last_price = None
    for c in candles:
        close = c.get('close', c.get('c'))
        volume = c.get('volume', c.get('v'))
        if close is None or volume in (None, 0):
            continue
        try:
            close = float(close)
            volume = float(volume)
        except Exception:
            continue
        total_pv += close * volume
        total_v += volume
        last_price = close
    if total_v <= 0:
        return None, last_price
    return total_pv / total_v, last_price

def get_nifty_vwap_via_kite():
    """
    Fetch today's intraday minute candles for NIFTY spot via Kite and compute VWAP.
    Requires env KITE_API_KEY and KITE_ACCESS_TOKEN. Returns (vwap, last_price) or (None, None) on failure.
    """
    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        return None, None
    try:
        from kiteconnect import KiteConnect
    except Exception:
        return None, None
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(KITE_ACCESS_TOKEN)
        # NIFTY 50 index instrument_token is commonly 256265
        instrument_token = 256265
        from datetime import datetime, time as dtime, timedelta
        today = datetime.now().date()
        start_dt = datetime.combine(today, dtime(9, 15))
        end_dt = datetime.now()
        data = kite.historical_data(instrument_token, start_dt, end_dt, interval="minute", continuous=False, oi=False)
        # Normalize candle dicts to have 'close' and 'volume'
        candles = []
        for item in data:
            candles.append({
                'close': item.get('close'),
                'volume': item.get('volume'),
            })
        vwap, last_price = compute_intraday_vwap(candles)
        return vwap, last_price
    except Exception:
        return None, None

def _compute_ema(values, period):
    """
    Lightweight EMA for a list of floats. Returns last EMA value or None.
    """
    if not values or period <= 0:
        return None
    k = 2.0 / (period + 1.0)
    ema_val = None
    for v in values:
        try:
            v = float(v)
        except Exception:
            continue
        if ema_val is None:
            ema_val = v
        else:
            ema_val = (v - ema_val) * k + ema_val
    return ema_val

def _compute_rsi(values, period=14):
    """
    Classic RSI. Returns last RSI value or None.
    """
    if not values or len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(values)):
        try:
            change = float(values[i]) - float(values[i-1])
        except Exception:
            continue
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def get_intraday_bias_via_kite():
    """
    Best-effort fetch of today's minute candles via Kite to compute:
    - VWAP, last price
    - EMA 9, EMA 21
    - RSI 14
    And return an aggregate bias: "Bullish" / "Bearish" / "Neutral" with a simple confidence score 0..3.
    Returns dict with keys: vwap, last, ema9, ema21, rsi14, bias, confidence
    If unavailable, returns dict with all None.
    """
    result = { 'vwap': None, 'last': None, 'ema9': None, 'ema21': None, 'rsi14': None, 'bias': None, 'confidence': None }
    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        return result
    try:
        from kiteconnect import KiteConnect
    except Exception:
        return result
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(KITE_ACCESS_TOKEN)
        instrument_token = 256265
        from datetime import datetime, time as dtime
        today = datetime.now().date()
        start_dt = datetime.combine(today, dtime(9, 15))
        end_dt = datetime.now()
        data = kite.historical_data(instrument_token, start_dt, end_dt, interval="minute", continuous=False, oi=False)
        closes = []
        candles = []
        for item in data:
            c = item.get('close')
            v = item.get('volume')
            if c is None or v in (None, 0):
                continue
            try:
                c = float(c)
                v = float(v)
            except Exception:
                continue
            closes.append(c)
            candles.append({'close': c, 'volume': v})
        vwap, last_price = compute_intraday_vwap(candles)
        ema9 = _compute_ema(closes, 9)
        ema21 = _compute_ema(closes, 21)
        rsi14 = _compute_rsi(closes, 14)
        # Aggregate bias
        votes = 0
        total = 0
        if vwap is not None and last_price is not None:
            total += 1
            votes += 1 if last_price > vwap else -1
        if ema9 is not None and ema21 is not None:
            total += 1
            votes += 1 if ema9 > ema21 else -1
        if rsi14 is not None:
            total += 1
            if rsi14 >= 60:
                votes += 1
            elif rsi14 <= 40:
                votes -= 1
        bias = None
        if total > 0:
            if votes >= 2:
                bias = "Bullish"
            elif votes <= -2:
                bias = "Bearish"
            else:
                bias = "Neutral"
        result.update({'vwap': vwap, 'last': last_price, 'ema9': ema9, 'ema21': ema21, 'rsi14': rsi14, 'bias': bias, 'confidence': abs(votes) if total > 0 else None})
        return result
    except Exception:
        return result

def get_daily_bias_via_kite():
    """
    Compute daily timeframe bias using Kite daily candles for NIFTY (token 256265).
    Indicators: EMA20, EMA50, EMA200, RSI14. Bias votes across:
      - Price vs EMA200 (trend filter)
      - EMA20 vs EMA50 (trend direction)
      - RSI14 (>55 bullish, <45 bearish)
    Returns dict: { 'ema20', 'ema50', 'ema200', 'rsi14', 'last', 'bias', 'confidence' }
    If unavailable, values are None.
    """
    result = { 'ema20': None, 'ema50': None, 'ema200': None, 'rsi14': None, 'last': None, 'bias': None, 'confidence': None }
    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        return result
    try:
        from kiteconnect import KiteConnect
    except Exception:
        return result
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(KITE_ACCESS_TOKEN)
        instrument_token = 256265
        from datetime import datetime, timedelta
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=400)  # enough history for EMA200
        data = kite.historical_data(instrument_token, start_dt, end_dt, interval="day", continuous=False, oi=False)
        closes = []
        for item in data:
            c = item.get('close')
            if c is None:
                continue
            try:
                c = float(c)
            except Exception:
                continue
            closes.append(c)
        if not closes:
            return result
        ema20 = _compute_ema(closes, 20)
        ema50 = _compute_ema(closes, 50)
        ema200 = _compute_ema(closes, 200)
        rsi14 = _compute_rsi(closes, 14)
        last_close = closes[-1]
        # Voting
        votes = 0
        total = 0
        if ema200 is not None:
            total += 1
            votes += 1 if last_close > ema200 else -1
        if ema20 is not None and ema50 is not None:
            total += 1
            votes += 1 if ema20 > ema50 else -1
        if rsi14 is not None:
            total += 1
            if rsi14 >= 55:
                votes += 1
            elif rsi14 <= 45:
                votes -= 1
        bias = None
        if total > 0:
            if votes >= 2:
                bias = "Bullish"
            elif votes <= -2:
                bias = "Bearish"
            else:
                bias = "Neutral"
        result.update({'ema20': ema20, 'ema50': ema50, 'ema200': ema200, 'rsi14': rsi14, 'last': last_close, 'bias': bias, 'confidence': abs(votes) if total > 0 else None})
        return result
    except Exception:
        return result

def fetch_nifty_option_chain():
    """
    Fetches Nifty 50 option chain data from NSE website.
    Returns the data as a list of dictionaries.
    """
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    # Try via nsepython first (handles NSE cookies/anti-bot better)
    try:
        from nsepython import nsefetch
        data = nsefetch(url)
        if not isinstance(data, dict):
            raise ValueError("Unexpected response type from nsefetch")
    except Exception:
        # Fallback to hardened requests session
        data = None
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Origin": "https://www.nseindia.com",
        "Referer": "https://www.nseindia.com/option-chain",
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if data is None:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=[401, 403, 429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        try:
            session.get("https://www.nseindia.com", headers=base_headers, timeout=10)
            time.sleep(0.8)
            session.get("https://www.nseindia.com/option-chain", headers=base_headers, timeout=10)
            time.sleep(0.8)
            response = session.get(url, headers=base_headers, timeout=10)
        except Exception as e:
            raise Exception(f"Failed to fetch data from NSE (network error): {e}")
        if response.status_code != 200:
            raise Exception("Failed to fetch data from NSE. Status code: {}".format(response.status_code))
        try:
            data = response.json()
        except Exception:
            text_preview = response.text[:300].replace("\n", " ")
            raise Exception(f"NSE response not JSON. Preview: {text_preview}")
    from datetime import datetime, timedelta

    def parse_expiry(exp_str):
        try:
            return datetime.strptime(exp_str, "%d-%b-%Y").date()
        except Exception:
            return None

    def last_thursday(year, month):
        if month == 12:
            next_month = datetime(year + 1, 1, 1).date()
        else:
            next_month = datetime(year, month + 1, 1).date()
        day = next_month - timedelta(days=1)
        while day.weekday() != 3:  # 0=Mon, 3=Thu
            day -= timedelta(days=1)
        return day

    expiry_dates = data.get('records', {}).get('expiryDates', [])
    parsed_expiries = [(e, parse_expiry(e)) for e in expiry_dates]
    parsed_expiries = [t for t in parsed_expiries if t[1] is not None]
    parsed_expiries.sort(key=lambda t: t[1])

    monthly_candidates = [e for e, d in parsed_expiries if d == last_thursday(d.year, d.month)]
    weekly_candidates = [e for e, d in parsed_expiries if d != last_thursday(d.year, d.month)]
    nearest_monthly = monthly_candidates[0] if monthly_candidates else (parsed_expiries[0][0] if parsed_expiries else None)
    nearest_weekly = weekly_candidates[0] if weekly_candidates else (parsed_expiries[0][0] if parsed_expiries else None)

    nifty_spot = data.get('records', {}).get('underlyingValue')
    # Try to compute intraday indicators via Kite (best effort)
    bias_info = get_intraday_bias_via_kite()
    daily_info = get_daily_bias_via_kite()
    nifty_vwap = bias_info.get('vwap')
    nifty_last_from_kite = bias_info.get('last')
    nifty_ema9 = bias_info.get('ema9')
    nifty_ema21 = bias_info.get('ema21')
    nifty_rsi14 = bias_info.get('rsi14')
    intraday_bias = bias_info.get('bias')
    bias_conf = bias_info.get('confidence')
    daily_bias = daily_info.get('bias')
    daily_conf = daily_info.get('confidence')
    # Prefer Kite last if available
    if nifty_last_from_kite is not None:
        try:
            nifty_spot = float(nifty_last_from_kite)
        except Exception:
            pass

    records_all = []
    for item in data['records']['data']:
        strike = item.get('strikePrice')
        ce = item.get('CE', {})
        pe = item.get('PE', {})
        expiry = item.get('expiryDate')
        record = {
            'expiryDate': expiry,
            'strikePrice': strike,
            'CE_OI': ce.get('openInterest'),
            'CE_Chg_OI': ce.get('changeinOpenInterest'),
            'CE_LTP': ce.get('lastPrice'),
            'PE_OI': pe.get('openInterest'),
            'PE_Chg_OI': pe.get('changeinOpenInterest'),
            'PE_LTP': pe.get('lastPrice'),
        }
        records_all.append(record)

    weekly_records = [r for r in records_all if r.get('expiryDate') == nearest_weekly]
    monthly_records = [r for r in records_all if r.get('expiryDate') == nearest_monthly]

    # Sort by strike for readability
    weekly_records.sort(key=lambda r: (r.get('strikePrice') is None, r.get('strikePrice')))
    monthly_records.sort(key=lambda r: (r.get('strikePrice') is None, r.get('strikePrice')))

    return {
        'weekly_expiry': nearest_weekly,
        'monthly_expiry': nearest_monthly,
        'weekly_records': weekly_records,
        'monthly_records': monthly_records,
        'nifty_spot': nifty_spot,
        'nifty_vwap': nifty_vwap,
        'nifty_ema9': nifty_ema9,
        'nifty_ema21': nifty_ema21,
        'nifty_rsi14': nifty_rsi14,
        'intraday_bias': intraday_bias,
        'bias_confidence': bias_conf,
        'daily_bias': daily_bias,
        'daily_confidence': daily_conf,
    }

def write_to_gsheet_with_formulas(records, sheet_name, worksheet_name):
    """
    Writes the option chain records to a Google Sheet and inserts prebuilt indicator logic for RSI, EMA, VWAP.
    Also sets up conditional formatting for trading signals.
    """
    try:
        import gspread
    except ImportError as e:
        print("The 'gspread' module is not installed. Please install it using 'pip install gspread' and try again.")
        raise
    try:
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError as e:
        print("The 'oauth2client' module is not installed. Please install it using 'pip install oauth2client' and try again.")
        raise
    # Define the scope and credentials file
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'credentials.json')
    if not os.path.exists(cred_path):
        raise FileNotFoundError(
            "Google Sheets credentials file not found. Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON path or place 'credentials.json' in the project root."
        )
    creds = ServiceAccountCredentials.from_json_keyfile_name(cred_path, scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name)
    try:
        worksheet = sheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=worksheet_name, rows="100", cols="30")
    # Prepare header and data
    header = [
        'strikePrice', 'CE_OI', 'CE_Chg_OI', 'CE_LTP', 'PE_OI', 'PE_Chg_OI', 'PE_LTP',
        'CE_EMA_5', 'CE_EMA_10', 'CE_RSI_14', 'CE_VWAP',
        'CE_EMA_20', 'CE_MACD', 'CE_MACD_SIGNAL', 'CE_MACD_HIST',
        'RSI_TREND', 'PRICE_VS_VWAP', 'EMA20_CROSS', 'SUPER_TREND',
        'RULE_CALL', 'RULE_PUT', 'OI_CONFIRM_CALL', 'OI_CONFIRM_PUT', 'FINAL_SIGNAL'
    ]
    data = [header]
    for i, rec in enumerate(records):
        row = [
            rec['strikePrice'],
            rec['CE_OI'],
            rec['CE_Chg_OI'],
            rec['CE_LTP'],
            rec['PE_OI'],
            rec['PE_Chg_OI'],
            rec['PE_LTP'],
            '', '', '', '',  # H, I, J, K populated by formulas
            '', '', '', '',  # L, M, N, O populated by formulas
            '', '', '', '',  # P, Q, R, S populated by formulas
            '', '', '', '',  # T, U, V, W populated by formulas
        ]
        data.append(row)
    worksheet.clear()
    worksheet.update('A1', data)

    # Insert prebuilt indicator logic for EMA, RSI, VWAP, MACD, and composite Signals
    num_rows = len(records)
    # EMA 5 and EMA 10 for CE_LTP (column D)
    for i in range(2, num_rows + 2):
        # Prebuilt indicator logic using Google Sheets add-on functions if available
        # If not available, fallback to simple moving average for EMA and custom RSI
        # EMA (using prebuilt or fallback)
        ema5_formula = f"=IFERROR(EMA(D$2:D{i}, 5), IF(COUNT(D$2:D{i})>=5, AVERAGE(D{i-4}:D{i}), \"\"))"
        ema10_formula = f"=IFERROR(EMA(D$2:D{i}, 10), IF(COUNT(D$2:D{i})>=10, AVERAGE(D{i-9}:D{i}), \"\"))"
        # RSI (using prebuilt or fallback)
        rsi14_formula = (
            f"=IFERROR(RSI(D$2:D{i}, 14), "
            f"IF(COUNT(D$2:D{i})>=14, "
            f"LET("
            f"r,D$2:D{i},"
            f"chg,ARRAYFORMULA(r-IFERROR(LAG(r),r)),"
            f"up,ARRAYFORMULA(IF(chg>0,chg,0)),"
            f"dn,ARRAYFORMULA(IF(chg<0,-chg,0)),"
            f"avgUp,AVERAGE(OFFSET(up,COUNT(up)-14,0,14,1)),"
            f"avgDn,AVERAGE(OFFSET(dn,COUNT(dn)-14,0,14,1)),"
            f"IF(avgDn=0,100,100-(100/(1+avgUp/avgDn)))"
            f"), \"\"))"
        )
        # VWAP (prebuilt logic: SUMPRODUCT(Price, Volume)/SUM(Volume))
        vwap_formula = f"=IF(SUM(B$2:B{i})=0, \"\", SUMPRODUCT(D$2:D{i}, B$2:B{i})/SUM(B$2:B{i}))"
        # EMA 20
        ema20_formula = f"=IFERROR(EMA(D$2:D{i}, 20), IF(COUNT(D$2:D{i})>=20, AVERAGE(D{i-19}:D{i}), \"\"))"
        # MACD line, Signal, Histogram with fallback to simple moving averages when EMA not available
        macd_line = (
            f"=IFERROR(MACD(D$2:D{i},12,26,9), "
            f"IF(COUNT(D$2:D{i})>=26, "
            f"(IFERROR(EMA(D$2:D{i},12), AVERAGE(D{max(2,i-11)}:D{i})) - IFERROR(EMA(D$2:D{i},26), AVERAGE(D{max(2,i-25)}:D{i}))), \"\"))"
        )
        macd_signal = (
            f"=IFERROR(INDEX(MACD(D$2:D{i},12,26,9),,2), "
            f"IF(COUNT(M$2:M{i})>=9, AVERAGE(M{max(2,i-8)}:M{i}), \"\"))"
        )
        macd_hist = f"=IF(AND(M{i}\"\"<>\"\",N{i}\"\"<>\"\"), M{i}-N{i}, \"\")"

        # RSI trend: above 50 and rising
        rsi_trend = f"=IF(J{i}=\"\", \"\", IF(AND(J{i}>50, J{i}>IFERROR(J{i-1}, J{i})), TRUE, FALSE))"
        # Price vs VWAP: price above VWAP
        price_vs_vwap = f"=IF(OR(D{i}=\"\", K{i}=\"\"), \"\", IF(D{i}>K{i}, TRUE, FALSE))"
        # EMA20 cross: price crossing above/below EMA20 at current row
        ema20_cross = (
            f"=IF(OR(D{i}=\"\", L{i}=\"\"), \"\", "
            f"IF(AND(D{i}>L{i}, IFERROR(D{i-1}, D{i})<=IFERROR(L{i-1}, L{i})), \"CROSS_UP\", "
            f"IF(AND(D{i}<L{i}, IFERROR(D{i-1}, D{i})>=IFERROR(L{i-1}, L{i})), \"CROSS_DOWN\", \"NO_CROSS\")))"
        )
        # MACD crossover using histogram sign change
        macd_cross = (
            f"=IF(O{i}=\"\", \"\", IF(AND(O{i}>0, IFERROR(O{i-1}, O{i})<=0), \"CROSS_UP\", "
            f"IF(AND(O{i}<0, IFERROR(O{i-1}, O{i})>=0), \"CROSS_DOWN\", \"NO_CROSS\")))"
        )
        # Supertrend placeholder: manual or add-on; default blank/NA
        super_trend = "=\"\""

        # Strategy rules
        rule_call = (
            f"=IF(AND({price_vs_vwap.replace('=','')}, {rsi_trend.replace('=','')}, R{i}=\"CROSS_UP\", S{i}=\"\"+\"\" OR S{i}=\"GREEN\", R{i}=\"CROSS_UP\"), TRUE, "
            f"IF(AND({price_vs_vwap.replace('=','')}, {rsi_trend.replace('=','')}, R{i}=\"CROSS_UP\", Q{i}=\"CROSS_UP\"), TRUE, FALSE))"
        )
        rule_put = (
            f"=IF(AND(NOT({price_vs_vwap.replace('=','')}), IF(J{i}=\"\", FALSE, AND(J{i}<50, J{i}<IFERROR(J{i-1}, J{i}))), R{i}=\"CROSS_DOWN\", S{i}=\"RED\"), TRUE, "
            f"IF(AND(NOT({price_vs_vwap.replace('=','')}), IF(J{i}=\"\", FALSE, AND(J{i}<50, J{i}<IFERROR(J{i-1}, J{i}))), Q{i}=\"CROSS_DOWN\"), TRUE, FALSE))"
        )
        # Optional OI confirmation (based on change in OI and simple price momentum across rows)
        oi_confirm_call = f"=IF(AND(C{i}>0, IFERROR(D{i}-D{i-1}, 0)>0), TRUE, FALSE)"
        oi_confirm_put = f"=IF(AND(F{i}>0, IFERROR(D{i}-D{i-1}, 0)<0), TRUE, FALSE)"
        # Final signal
        final_signal = (
            f"=IF(T{i}, IF(U{i}, \"BUY CALL (conf)\", \"BUY CALL\"), IF(U{i}, \"BUY CALL (conf)\", IF(V{i}, IF(W{i}, \"BUY PUT (conf)\", \"BUY PUT\"), \"HOLD\")))"
        )
        worksheet.update_acell(f'H{i}', ema5_formula)
        worksheet.update_acell(f'I{i}', ema10_formula)
        worksheet.update_acell(f'J{i}', rsi14_formula)
        worksheet.update_acell(f'K{i}', vwap_formula)
        worksheet.update_acell(f'L{i}', ema20_formula)
        worksheet.update_acell(f'M{i}', macd_line)
        worksheet.update_acell(f'N{i}', macd_signal)
        worksheet.update_acell(f'O{i}', macd_hist)
        worksheet.update_acell(f'P{i}', rsi_trend)
        worksheet.update_acell(f'Q{i}', price_vs_vwap)
        worksheet.update_acell(f'R{i}', ema20_cross)
        worksheet.update_acell(f'S{i}', super_trend)
        worksheet.update_acell(f'T{i}', rule_call)
        worksheet.update_acell(f'U{i}', rule_put)
        worksheet.update_acell(f'V{i}', oi_confirm_call)
        worksheet.update_acell(f'W{i}', oi_confirm_put)
        worksheet.update_acell(f'X{i}', final_signal)

    # Set up conditional formatting for Signal column (L)
    # This must be done manually in the Google Sheet UI or via Google Sheets API (not gspread).
    # Instructions for user:
    print("\n--- ACTION REQUIRED ---")
    print("To set up conditional formatting for trading signals in your Google Sheet:")
    print("1. Select column 'L' (Signal).")
    print("2. Go to Format > Conditional formatting.")
    print("3. Add rules:")
    print("   - If cell text is 'BUY', set background to green.")
    print("   - If cell text is 'SELL', set background to red.")
    print("   - If cell text is 'HOLD', set background to yellow.")
    print("4. Click 'Done'.")
    print("You can also set up notifications for changes using Google Sheets' built-in notification rules.")

    print("\n--- FORMULA NOTES ---")
    print("If EMA or RSI formulas are not recognized, you can use Google Sheets' built-in functions or add-ons like 'Technical Analysis Functions' from the Google Workspace Marketplace.")
    print("For EMA: Use =EMA(range, period) if available, or use =AVERAGE for a simple moving average as a fallback.")
    print("For RSI: Use =RSI(range, period) if available, or use a custom formula (see fallback in code).")
    print("VWAP is approximated as SUMPRODUCT(Price, Volume)/SUM(Volume).")

def render_option_chain_html(records_or_bundle, output_path="option_chain.html", open_in_browser=True):
    """
    Renders the option chain records to a local HTML file and opens it in the default browser.
    """
    columns = [
        'strikePrice', 'CE_OI', 'CE_Chg_OI', 'CE_LTP',
        'PE_OI', 'PE_Chg_OI', 'PE_LTP',
        'Entry_Price', 'Exit_Price', 'Stoploss',
        'Entry_Window', 'Exit_Window',
        'Used_Indicators', 'Signal'
    ]
    decimals_by_column = {
        'strikePrice': 0, 'CE_OI': 0, 'CE_Chg_OI': 0, 'PE_OI': 0, 'PE_Chg_OI': 0,
        'CE_LTP': 2, 'PE_LTP': 2, 'Entry_Price': 2, 'Exit_Price': 2, 'Stoploss': 2,
    }
    def format_number(value, decimals):
        if value in (None, ""):
            return ""
        try:
            number = float(value)
        except Exception:
            return value
        if decimals == 0:
            return f"{int(round(number)):,}"
        return f"{number:,.{decimals}f}"

    # Normalize input (old path returned list, new path returns bundle with weekly/monthly)
    if isinstance(records_or_bundle, dict) and 'weekly_records' in records_or_bundle:
        weekly_expiry = records_or_bundle.get('weekly_expiry') or "Weekly"
        monthly_expiry = records_or_bundle.get('monthly_expiry') or "Monthly"
        weekly_records_src = records_or_bundle.get('weekly_records') or []
        monthly_records_src = records_or_bundle.get('monthly_records') or []
        nifty_spot = records_or_bundle.get('nifty_spot')
        nifty_vwap = records_or_bundle.get('nifty_vwap')
        nifty_ema9 = records_or_bundle.get('nifty_ema9')
        nifty_ema21 = records_or_bundle.get('nifty_ema21')
        nifty_rsi14 = records_or_bundle.get('nifty_rsi14')
        intraday_bias = records_or_bundle.get('intraday_bias')
        bias_confidence = records_or_bundle.get('bias_confidence')
        daily_bias = records_or_bundle.get('daily_bias')
        daily_confidence = records_or_bundle.get('daily_confidence')
    else:
        weekly_expiry = "Weekly"
        monthly_expiry = "Monthly"
        weekly_records_src = records_or_bundle or []
        monthly_records_src = []
        nifty_spot = None
        nifty_vwap = None
        nifty_ema9 = None
        nifty_ema21 = None
        nifty_rsi14 = None
        intraday_bias = None
        bias_confidence = None
        daily_bias = None
        daily_confidence = None
    # Indicator status (Active/Partial/Off) for toolbar badge
    has_vwap = (nifty_vwap is not None)
    has_ema = (nifty_ema9 is not None and nifty_ema21 is not None)
    has_rsi = (nifty_rsi14 is not None)
    ind_count = (1 if has_vwap else 0) + (1 if has_ema else 0) + (1 if has_rsi else 0)
    if ind_count == 3:
        ind_status_text = "Indicators: Active"
        ind_status_class = "pill pill-ok"
    elif ind_count > 0:
        ind_status_text = "Indicators: Partial"
        ind_status_class = "pill pill-partial"
    else:
        ind_status_text = "Indicators: Off"
        ind_status_class = "pill pill-off"
    # Heading style class for indicators panel
    if ind_count == 3:
        ind_h3_class = "ind-h3 ok"
    elif ind_count > 0:
        ind_h3_class = "ind-h3 partial"
    else:
        ind_h3_class = "ind-h3 off"
    # Choose CE/PE decision based on bias
    choose_text = ""
    choose_class = ""
    if intraday_bias == 'Bullish':
        choose_text = "Choose: CE"
        choose_class = "pill pill-ce"
    elif intraday_bias == 'Bearish':
        choose_text = "Choose: PE"
        choose_class = "pill pill-pe"

    def build_rows(recs):
        html_rows_local = []
        for rec in recs:
            ce_ltp = rec.get('CE_LTP') or 0
            entry_price = ce_ltp if ce_ltp else ""
            exit_price = round(ce_ltp * 1.05, 2) if ce_ltp else ""
            stoploss = round(ce_ltp * 0.97, 2) if ce_ltp else ""
            ce_chg_oi = rec.get('CE_Chg_OI')
            pe_chg_oi = rec.get('PE_Chg_OI')

            signal = "HOLD"
            row_class = ""
            used_indicators = "-"
            vwap_note = None
            vwap_bias_ok_ce = None
            if nifty_spot is not None and nifty_vwap is not None:
                try:
                    vwap_note = "VWAP: above" if float(nifty_spot) > float(nifty_vwap) else "VWAP: below"
                    vwap_bias_ok_ce = (float(nifty_spot) > float(nifty_vwap))
                except Exception:
                    vwap_note = None
                    vwap_bias_ok_ce = None
            try:
                ce_chg = float(ce_chg_oi) if ce_chg_oi is not None else 0.0
                pe_chg = float(pe_chg_oi) if pe_chg_oi is not None else 0.0
                if ce_chg > 0 and pe_chg <= 0:
                    # Gate by VWAP if available
                    if vwap_bias_ok_ce is None or vwap_bias_ok_ce:
                        signal = "BUY CE"
                        row_class = "row-buy-ce"
                    used_indicators = "OI↑(CE), OI↓/flat(PE)"
                    if vwap_note:
                        used_indicators += ", " + vwap_note
                elif pe_chg > 0 and ce_chg <= 0:
                    # Gate by VWAP if available (PE expects below)
                    if vwap_bias_ok_ce is None or (vwap_bias_ok_ce is False):
                        signal = "BUY PE"
                        row_class = "row-buy-pe"
                    used_indicators = "OI↑(PE), OI↓/flat(CE)"
                    if vwap_note:
                        used_indicators += ", " + vwap_note
            except Exception:
                pass

            # Recommended time windows
            entry_window = "09:30–11:00; 13:30–15:00"
            exit_window = "Hit SL/Target; else 15:10–15:20"

            raw_values = {
                'strikePrice': rec.get('strikePrice', ""),
                'CE_OI': rec.get('CE_OI', ""),
                'CE_Chg_OI': rec.get('CE_Chg_OI', ""),
                'CE_LTP': rec.get('CE_LTP', ""),
                'PE_OI': rec.get('PE_OI', ""),
                'PE_Chg_OI': rec.get('PE_Chg_OI', ""),
                'PE_LTP': rec.get('PE_LTP', ""),
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Stoploss': stoploss,
                'Entry_Window': entry_window,
                'Exit_Window': exit_window,
                'Used_Indicators': used_indicators,
                'Signal': signal,
            }

            cells_markup = []
            for col in columns:
                value = raw_values.get(col, "")
                formatted = value if col in ('Signal','Entry_Window','Exit_Window','Used_Indicators') else format_number(value, decimals_by_column.get(col, 2))
                cell_class = ""
                if col in ("CE_Chg_OI", "PE_Chg_OI"):
                    try:
                        num = float(value)
                        if num > 0:
                            cell_class = " class=\"pos\""
                        elif num < 0:
                            cell_class = " class=\"neg\""
                    except Exception:
                        pass
                if col == 'Signal' and value:
                    if value == 'BUY CE':
                        cell_class = " class=\"pos\""
                    elif value == 'BUY PE':
                        cell_class = " class=\"neg\""
                cells_markup.append(f"<td{cell_class}>{formatted}</td>")

            html_rows_local.append(f"<tr class=\"{row_class}\">{''.join(cells_markup)}</tr>")
        return html_rows_local

    weekly_rows = build_rows(weekly_records_src)
    monthly_rows = build_rows(monthly_records_src)
    table_header = ''.join(f"<th>{h}</th>" for h in columns)

    def rank_top_suggestions(recs, spot):
        if not recs or spot is None:
            return [], []
        def ce_score(rec):
            try:
                chg = float(rec.get('CE_Chg_OI') or 0)
                dist = abs(float(rec.get('strikePrice') or 0) - float(spot))
                return (chg / 1000.0) - (dist / 25.0)
            except Exception:
                return -1e9
        def pe_score(rec):
            try:
                chg = float(rec.get('PE_Chg_OI') or 0)
                dist = abs(float(rec.get('strikePrice') or 0) - float(spot))
                return (chg / 1000.0) - (dist / 25.0)
            except Exception:
                return -1e9
        ce_sorted = sorted(recs, key=ce_score, reverse=True)
        pe_sorted = sorted(recs, key=pe_score, reverse=True)
        # Keep only positive-score ideas
        ce_top = [(r, ce_score(r)) for r in ce_sorted[:20] if ce_score(r) > 0][:5]
        pe_top = [(r, pe_score(r)) for r in pe_sorted[:20] if pe_score(r) > 0][:5]
        return ce_top, pe_top

    weekly_ce_top, weekly_pe_top = rank_top_suggestions(weekly_records_src, nifty_spot)
    monthly_ce_top, monthly_pe_top = rank_top_suggestions(monthly_records_src, nifty_spot)

    def build_top_list(items, side, lot_size_val):
        # items: list of (record, score)
        li = []
        for rec, sc in items:
            strike = rec.get('strikePrice', '')
            raw_ltp = rec.get('CE_LTP' if side == 'CE' else 'PE_LTP', 0) or 0
            entry = raw_ltp if raw_ltp else ""
            target = round(raw_ltp * 1.05, 2) if raw_ltp else ""
            stop = round(raw_ltp * 0.97, 2) if raw_ltp else ""
            li.append(
                f"<li><span class=\"strike\">{format_number(strike, 0)}</span> · {side} Entry: {format_number(entry, 2)} · Exit: {format_number(target, 2)} · SL: {format_number(stop, 2)} · Lot: {lot_size_val} · score: {format_number(sc, 2)}</li>"
            )
        return "".join(li) or "<li>No strong setups right now</li>"

    lot_size_now = get_lot_size()
    weekly_top_html = f"""
    <div class=\"topbox\">
      <div class=\"topcol\">
        <div class=\"toptitle pos\">Best CE trades (weekly)</div>
        <ul>{build_top_list(weekly_ce_top, 'CE', lot_size_now)}</ul>
      </div>
      <div class=\"topcol\">
        <div class=\"toptitle neg\">Best PE trades (weekly)</div>
        <ul>{build_top_list(weekly_pe_top, 'PE', lot_size_now)}</ul>
      </div>
    </div>
    """
    monthly_top_html = f"""
    <div class=\"topbox\">
      <div class=\"topcol\">
        <div class=\"toptitle pos\">Best CE trades (monthly)</div>
        <ul>{build_top_list(monthly_ce_top, 'CE', lot_size_now)}</ul>
      </div>
      <div class=\"topcol\">
        <div class=\"toptitle neg\">Best PE trades (monthly)</div>
        <ul>{build_top_list(monthly_pe_top, 'PE', lot_size_now)}</ul>
      </div>
    </div>
    """
    # Intraday Best Calls box from intraday bias and weekly ranking
    def build_intraday_best_calls():
        # Show only when indicators are fully active (VWAP, EMA9/21, RSI14)
        try:
            if ind_count != 3:
                return ""
        except Exception:
            return ""
        # Require strong alignment and confidence: intraday & daily bias match and both high confidence
        try:
            if not intraday_bias or not daily_bias or intraday_bias != daily_bias:
                return ""
            if bias_confidence is None or daily_confidence is None or int(bias_confidence) < 2 or int(daily_confidence) < 2:
                return ""
        except Exception:
            return ""
        def li_for(rec, side, score):
            strike = rec.get('strikePrice', '')
            raw_ltp = rec.get('CE_LTP' if side == 'CE' else 'PE_LTP', 0) or 0
            entry = raw_ltp if raw_ltp else ""
            target = round(raw_ltp * 1.05, 2) if raw_ltp else ""
            stop = round(raw_ltp * 0.97, 2) if raw_ltp else ""
            return f"<li><span class=\"strike\">{format_number(strike, 0)}</span> · {side} Entry: {format_number(entry, 2)} · Exit: {format_number(target, 2)} · SL: {format_number(stop, 2)} · Lot: {lot_size_now} · score: {format_number(score, 2)}</li>"
        title = "High confidence intraday calls"
        klass = "info"
        lis = []
        if intraday_bias == 'Bullish' and weekly_ce_top:
            title = "High confidence intraday calls (Bullish)"
            klass = "pos"
            lis = [li_for(rec, 'CE', sc) for rec, sc in weekly_ce_top[:3]]
        elif intraday_bias == 'Bearish' and weekly_pe_top:
            title = "High confidence intraday calls (Bearish)"
            klass = "neg"
            lis = [li_for(rec, 'PE', sc) for rec, sc in weekly_pe_top[:3]]
        else:
            # Neutral or missing bias: mix top 2 from each if present
            mix = []
            for i in range(0, 2):
                if i < len(weekly_ce_top):
                    mix.append(('CE', weekly_ce_top[i]))
                if i < len(weekly_pe_top):
                    mix.append(('PE', weekly_pe_top[i]))
            lis = [li_for(rec, side, sc) for side, (rec, sc) in mix]
        if not lis:
            return ""
        return f"<div class=\"best-calls {klass}\"><div class=\"toptitle\">{title}</div><ul>{''.join(lis)}</ul></div>"

    intraday_best_html = build_intraday_best_calls()

    # Dedicated Best CE trades (intraday) regardless of bias, shown only when indicators are active
    def build_intraday_best_ce():
        try:
            if ind_count != 3:
                return ""
        except Exception:
            return ""
        if not weekly_ce_top:
            return ""
        def li_for(rec, score):
            strike = rec.get('strikePrice', '')
            raw_ltp = rec.get('CE_LTP', 0) or 0
            entry = raw_ltp if raw_ltp else ""
            target = round(raw_ltp * 1.05, 2) if raw_ltp else ""
            stop = round(raw_ltp * 0.97, 2) if raw_ltp else ""
            return f"<li><span class=\"strike\">{format_number(strike, 0)}</span> · CE Entry: {format_number(entry, 2)} · Exit: {format_number(target, 2)} · SL: {format_number(stop, 2)} · Lot: {lot_size_now} · score: {format_number(score, 2)}</li>"
        lis = [li_for(rec, sc) for rec, sc in weekly_ce_top[:5]]
        if not lis:
            return ""
        return f"<div class=\"best-calls pos\"><div class=\"toptitle\">Best CE trades (intraday)</div><ul>{''.join(lis)}</ul></div>"

    intraday_best_ce_html = build_intraday_best_ce()

    # Top intraday call (single highlighted pick) based on bias and ranking
    def build_intraday_top_pick():
        pick = None
        side = None
        if intraday_bias == 'Bullish' and weekly_ce_top:
            side = 'CE'
            pick = weekly_ce_top[0]
        elif intraday_bias == 'Bearish' and weekly_pe_top:
            side = 'PE'
            pick = weekly_pe_top[0]
        else:
            # Choose the best available by score between CE and PE tops
            cand = []
            if weekly_ce_top:
                cand.append(('CE', weekly_ce_top[0]))
            if weekly_pe_top:
                cand.append(('PE', weekly_pe_top[0]))
            if not cand:
                return ""
            # pick with max score
            side, pick = max(cand, key=lambda t: t[1][1])
        if not pick:
            return ""
        rec, sc = pick
        strike = rec.get('strikePrice', '')
        raw_ltp = rec.get('CE_LTP' if side == 'CE' else 'PE_LTP', 0) or 0
        entry = raw_ltp if raw_ltp else ""
        target = round(raw_ltp * 1.05, 2) if raw_ltp else ""
        stop = round(raw_ltp * 0.97, 2) if raw_ltp else ""
        klass = 'pos' if side == 'CE' else 'neg'
        return f"<div class=\"top-pick {klass}\"><div class=\"toptitle\">Top intraday call</div><div><span class=\"strike\">{format_number(strike, 0)}</span> · {side} · Entry: {format_number(entry, 2)} · Exit: {format_number(target, 2)} · SL: {format_number(stop, 2)} · score: {format_number(sc, 2)}</div></div>"

    intraday_top_pick_html = build_intraday_top_pick()

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Nifty 50 Option Chain</title>
  <style>
    :root {{ --bg: #0b1020; --card: #0f172a; --muted: #94a3b8; --border: #1f2937; --header: #111827; --text: #e2e8f0; --accent: #2563eb; --pos: #16a34a; --neg: #ef4444; }}
    html, body {{ height: 100%; }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Noto Sans, \"Apple Color Emoji\", \"Segoe UI Emoji\"; margin: 0; background: var(--bg); color: var(--text); }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    .subtitle {{ margin-top: 6px; color: var(--muted); }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; box-shadow: 0 10px 22px rgba(0,0,0,0.25); overflow: hidden; }}
    .toolbar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 16px; background: var(--header); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 2; }}
    .btn {{ background: var(--accent); color: white; border: 0; padding: 8px 12px; border-radius: 6px; cursor: pointer; }}
    .pill {{ display: inline-block; background: #1e293b; color: #93c5fd; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-left: 8px; }}
    .pill-ok {{ background: rgba(16,185,129,0.14); color: #22c55e; border: 1px solid rgba(34,197,94,0.35); }}
    .pill-partial {{ background: rgba(234,179,8,0.14); color: #f59e0b; border: 1px solid rgba(245,158,11,0.35); }}
    .pill-off {{ background: rgba(75,85,99,0.14); color: #9ca3af; border: 1px solid rgba(107,114,128,0.35); }}
    .pill-ce {{ background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(16,185,129,0.35); }}
    .pill-pe {{ background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.35); }}
    .price {{ display:inline-block; margin-top:6px; padding: 2px 10px; border-radius: 999px; background: #0b1220; border: 1px solid var(--border); color: #e2e8f0; font-variant-numeric: tabular-nums; }}
    select {{ background: #0b1220; color: #e2e8f0; border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; }}
    .topwrap {{ padding: 12px 16px; background: #0b1220; border-bottom: 1px solid var(--border); }}
    .topbox {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .topcol {{ background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }}
    .toptitle {{ font-weight: 600; margin-bottom: 8px; }}
    .topcol ul {{ margin: 0; padding-left: 18px; }}
    .strike {{ font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    thead th {{ position: sticky; top: 56px; background: var(--header); color: #cbd5e1; text-align: right; font-weight: 600; padding: 10px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    thead th:first-child, tbody td:first-child {{ text-align: right; }}
    tbody td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: right; font-variant-numeric: tabular-nums; }}
    tbody tr:nth-child(2n) {{ background: rgba(255,255,255,0.02); }}
    tbody tr:hover {{ background: rgba(37, 99, 235, 0.08); }}
    .pos {{ color: var(--pos); }}
    .neg {{ color: var(--neg); }}
      .row-buy-ce {{ background: rgba(22, 163, 74, 0.10); }}
      .row-buy-pe {{ background: rgba(239, 68, 68, 0.10); }}
     .hidden {{ display: none; }}
    .indicators {{ padding: 16px; }}
    .indicators h3 {{ margin: 12px 0; font-size: 16px; color: #cbd5e1; }}
    .indicators ul {{ margin: 0; padding-left: 18px; color: #94a3b8; }}
    /* Toast notification */
    .toast-wrap {{ position: fixed; right: 16px; top: 16px; z-index: 9999; display: flex; flex-direction: column; gap: 10px; }}
    .toast {{ background: #0f172a; border: 1px solid var(--pos); color: #e2e8f0; padding: 10px 12px; border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.35); min-width: 260px; position: relative; }}
    .toast h4 {{ margin: 0 0 6px 0; color: var(--pos); font-size: 14px; padding-right: 18px; }}
    .toast .close {{ position: absolute; top: 8px; right: 10px; cursor: pointer; color: #94a3b8; }}
    .toast .small {{ color: #94a3b8; font-size: 12px; }}
    .toast.pe {{ border-color: var(--neg); }}
    .toast.pe h4 {{ color: var(--neg); }}
    .toast.info {{ border-color: #4b5563; }}
    .toast.info h4 {{ color: #9ca3af; }}
    /* Intraday best calls */
    .best-calls {{ margin: 10px 16px; background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }}
    .best-calls.pos {{ border-color: rgba(22, 163, 74, 0.55); }}
    .best-calls.neg {{ border-color: rgba(239, 68, 68, 0.55); }}
    .best-calls.info {{ border-color: #4b5563; }}
    /* Top intraday pick */
    .top-pick {{ margin: 10px 16px; background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }}
    .top-pick.pos {{ border-color: rgba(22, 163, 74, 0.65); }}
    .top-pick.neg {{ border-color: rgba(239, 68, 68, 0.65); }}
    /* Scroll-to-top button */
    .to-top {{ position: fixed; right: 18px; bottom: 18px; z-index: 10000; background: #0f172a; color: #e2e8f0; border: 1px solid var(--border); border-radius: 999px; padding: 8px 12px; cursor: pointer; box-shadow: 0 6px 18px rgba(0,0,0,0.35); }}
    .to-top:hover {{ background: #111827; }}
    /* Indicators heading state */
    .ind-h3 {{ margin: 12px 0; font-size: 16px; }}
    .ind-h3.ok {{ color: #34d399; }}
    .ind-h3.partial {{ color: #f59e0b; }}
    .ind-h3.off {{ color: #9ca3af; }}
  </style>
  <script>
    function refreshPage() {{ location.reload(); }}
    // Give popups time to show before auto reload
    setTimeout(refreshPage, 12000);

    function getAlertedSet(key) {{
      try {{ return new Set(JSON.parse(localStorage.getItem(key) || '[]')); }} catch(e) {{ return new Set(); }}
    }}
    function saveAlertedSet(key, set) {{
      localStorage.setItem(key, JSON.stringify(Array.from(set)));
    }}
    function clearAlerted() {{
      localStorage.removeItem('alerted_buy_ce');
      localStorage.removeItem('alerted_buy_pe');
    }}
    function showToast(html, type) {{
      const wrap = document.querySelector('.toast-wrap') || (function(){{
        const d = document.createElement('div'); d.className = 'toast-wrap'; document.body.appendChild(d); return d;
      }})();
      let klass = 'toast';
      if (type==='PE') klass += ' pe';
      if (type==='INFO') klass += ' info';
      const t = document.createElement('div'); t.className = klass;
      t.innerHTML = '<span class="close" title="Close">&times;</span>' + html;
      const closeBtn = t.querySelector('.close');
      closeBtn.addEventListener('click', function(){{ t.remove(); }});
      wrap.appendChild(t);
      setTimeout(()=> {{ t.remove(); }}, 7000);
    }}
    function runScan(force) {{
      if (force) clearAlerted();
      scanForBuyCEAlerts();
    }}
    function getHeaderIndexMap() {{
      const ths = document.querySelectorAll('#weeklyTable thead th');
      const map = {{}};
      ths.forEach((th, i) => {{ map[(th.innerText||'').trim()] = i; }});
      return map;
    }}
    function scanForBuyCEAlerts() {{
      const rows = document.querySelectorAll('#weeklyTable tbody tr');
      if (!rows.length) return;
      const idx = getHeaderIndexMap();
      const strikeIdx = idx['strikePrice'] ?? 0;
      const sigIdx = idx['Signal'];
      const entryIdx = idx['Entry_Price'];
      const exitIdx = idx['Exit_Price'];
      const slIdx = idx['Stoploss'];
      const usedIdx = idx['Used_Indicators'];
      const alertedCE = getAlertedSet('alerted_buy_ce');
      const alertedPE = getAlertedSet('alerted_buy_pe');
      let alerts = 0;
      rows.forEach((tr) => {{
        const tds = tr.querySelectorAll('td');
        if (!tds.length || sigIdx == null) return;
        const strike = (tds[strikeIdx]?.innerText || '').trim();
        const signal = (tds[sigIdx]?.innerText || '').trim();
        const entry = (tds[entryIdx]?.innerText || '').trim();
        const exitp = (tds[exitIdx]?.innerText || '').trim();
        const sl = (tds[slIdx]?.innerText || '').trim();
        const used = (tds[usedIdx]?.innerText || '').trim();
        if (signal === 'BUY CE') {{
          const key = 'CE|' + strike;
          if (!alertedCE.has(key) && alerts < 5) {{
            showToast('<h4>BUY CE alert</h4>' +
                      '<div>Strike: <b>'+strike+'</b></div>'+
                      '<div class="small">Entry: '+entry+' · Exit: '+exitp+' · SL: '+sl+'</div>'+
                      '<div class="small">Indicators: '+used+'</div>', 'CE');
            alertedCE.add(key);
            alerts++;
          }}
        }} else if (signal === 'BUY PE') {{
          const key = 'PE|' + strike;
          if (!alertedPE.has(key) && alerts < 5) {{
            showToast('<h4>BUY PE alert</h4>' +
                      '<div>Strike: <b>'+strike+'</b></div>'+
                      '<div class="small">Entry: '+entry+' · Exit: '+exitp+' · SL: '+sl+'</div>'+
                      '<div class="small">Indicators: '+used+'</div>', 'PE');
            alertedPE.add(key);
            alerts++;
          }}
        }}
      }});
      if (alerts > 0) {{
        saveAlertedSet('alerted_buy_ce', alertedCE);
        saveAlertedSet('alerted_buy_pe', alertedPE);
      }} else {{
        showToast('<h4>No new signals</h4><div class="small">Try again later or press Refresh to force scan.</div>', 'INFO');
      }}
    }}
    // Delegated close handler for reliability
    document.addEventListener('click', function(e){{
      const target = e.target;
      if (target && target.classList && target.classList.contains('close')) {{
        const t = target.closest('.toast');
        if (t) t.remove();
      }}
    }});
    // ESC to close all toasts
    document.addEventListener('keydown', function(e){{
      if (e.key === 'Escape') {{
        document.querySelectorAll('.toast').forEach((t)=> t.remove());
      }}
    }});
    window.addEventListener('DOMContentLoaded', function(){{
      setTimeout(()=> runScan(false), 150);
      // Show/hide Top button
      const topBtn = document.getElementById('toTopBtn');
      function toggleTopBtn() {{
        if (!topBtn) return;
        if (window.scrollY > 200) topBtn.classList.remove('hidden'); else topBtn.classList.add('hidden');
      }}
      window.addEventListener('scroll', toggleTopBtn);
      if (topBtn) {{
        topBtn.addEventListener('click', function() {{ window.scrollTo({{ top: 0, behavior: 'smooth' }}); }});
      }}
      toggleTopBtn();
    }});
  </script>
  </head>
  <body>
    <div class=\"container\">
      <div class=\"card\">
        <div class=\"toolbar\">
          <div>
            <h1>Nifty 50 Option Chain <span class=\"pill\">Live snapshot</span></h1>
            <div class=\"subtitle\">Strike, OI, change in OI, LTP, entry/exit/SL</div>
            <div class=\"price\">NIFTY: {format_number(nifty_spot, 2) if nifty_spot is not None else ''}</div>
            <div class=\"price\">VWAP: {format_number(nifty_vwap, 2) if nifty_vwap is not None else 'N/A'}</div>
            <div class=\"price\">EMA9/21: {(format_number(nifty_ema9, 2) if nifty_ema9 is not None else 'N/A')} / {(format_number(nifty_ema21, 2) if nifty_ema21 is not None else 'N/A')}</div>
            <div class=\"price\">RSI14: {(format_number(nifty_rsi14, 2) if nifty_rsi14 is not None else 'N/A')}</div>
            <div class=\"price\">Bias: {(intraday_bias if intraday_bias else 'N/A')} {(f"(conf {int(bias_confidence)}/3)" if bias_confidence is not None else '')}</div>
            <div class=\"price\">Lot: {lot_size_now}</div>
          </div>
          <div style="display:flex;gap:10px;align-items:center;">
            <span class="pill">Weekly: {weekly_expiry}</span>
            <span class="{ind_status_class}">{ind_status_text}</span>
            {f'<span class="{choose_class}">{choose_text}</span>' if choose_text else ''}
            {f'<span class="pill {"pill-ce" if daily_bias=="Bullish" else ("pill-pe" if daily_bias=="Bearish" else "")}">Daily: {daily_bias} {(f"(conf {int(daily_confidence)}/3)" if daily_confidence is not None else "")}</span>' if daily_bias else ''}
            <button class="btn" onclick="runScan(true)">Refresh</button>
          </div>
        </div>
        {intraday_top_pick_html}
        {intraday_best_html}
        {intraday_best_ce_html}
        <div id=\"weeklyTop\" class=\"topwrap\">{weekly_top_html}</div>
        <table id=\"weeklyTable\">
          <thead><tr>{table_header}</tr></thead>
          <tbody>
            {''.join(weekly_rows)}
          </tbody>
        </table>
        <div class=\"indicators\">
          <h3 class=\"{ind_h3_class}\">Indicators (intraday)</h3>
          <ul>
            <li><b>VWAP</b>: Above = bullish bias (prefer CE); Below = bearish (prefer PE)</li>
            <li><b>EMA 9/21</b>: 9&gt;21 bullish; 9&lt;21 bearish; use with VWAP</li>
            <li><b>RSI 14</b>: &gt;60 strong up; &lt;40 strong down; divergences for reversal</li>
            <li><b>Support/Resistance</b>: Validate breakouts/breakdowns and set targets</li>
            <li><b>Open Interest (OI)</b>: Rising CE OI + price up strengthens CE trend; rising PE OI + price down strengthens PE trend</li>
            <li><b>Supertrend 10,3</b> (optional): Green align CE; Red align PE</li>
            <li><b>Risk</b>: SL 20–30% of premium; partials 40–60%; close by 15:10–15:20</li>
          </ul>
        </div>
      </div>
    </div>
    <div class=\"toast-wrap\"></div>
    <button id="toTopBtn" class="to-top hidden" title="Scroll to top">Top</button>
  </body>
</html>
"""
    out_path = Path(output_path).resolve()
    out_path.write_text(html, encoding="utf-8")
    if open_in_browser:
        webbrowser.open(out_path.as_uri())

if __name__ == "__main__":
    has_opened = False
    try:
        while True:
            try:
                records = fetch_nifty_option_chain()
            except Exception as e:
                print("Error fetching option chain:", e)
                records = []
            if records:
                render_option_chain_html(records, open_in_browser=(not has_opened))
                if not has_opened:
                    print("Opened Nifty 50 option chain snapshot in your browser.")
                    has_opened = True
            else:
                print("No data to write.")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped refreshing.")
