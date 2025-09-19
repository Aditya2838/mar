"""
Nifty 50 Options Trading System
Provides comprehensive functionality for trading Nifty 50 index options
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Option type enumeration"""
    CALL = "CE"
    PUT = "PE"

class OptionExpiry(Enum):
    """Option expiry periods"""
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"

class ExitStrategy(Enum):
    """Exit strategy types"""
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_BASED = "TIME_BASED"
    MANUAL = "MANUAL"
    TRAILING_STOP = "TRAILING_STOP"

@dataclass
class TradeSetup:
    """Trade setup with entry, stop loss, and target"""
    entry_price: float
    stop_loss: float
    target_price: float
    quantity: int
    risk_reward_ratio: float
    max_loss: float
    max_profit: float
    
    def __post_init__(self):
        """Calculate risk-reward metrics"""
        self.risk = abs(self.entry_price - self.stop_loss)
        self.reward = abs(self.target_price - self.entry_price)
        self.risk_reward_ratio = self.reward / self.risk if self.risk > 0 else 0
        self.max_loss = self.risk * self.quantity * 50  # 50 is lot size
        self.max_profit = self.reward * self.quantity * 50

@dataclass
class OptionContract:
    """Option contract data structure"""
    symbol: str
    strike_price: float
    option_type: OptionType
    expiry_date: date
    lot_size: int = 50
    underlying: str = "NIFTY50"
    
    def __post_init__(self):
        self.contract_id = f"{self.underlying}_{self.strike_price}_{self.option_type.value}_{self.expiry_date.strftime('%Y%m%d')}"
        self.display_name = f"{self.strike_price} {self.option_type.value} {self.expiry_date.strftime('%d-%b-%Y')}"

@dataclass
class OptionQuote:
    """Option quote data structure"""
    contract: OptionContract
    bid_price: float
    ask_price: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    timestamp: datetime
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask_price - self.bid_price
    
    @property
    def spread_percentage(self) -> float:
        """Calculate bid-ask spread percentage"""
        return (self.spread / self.mid_price) * 100 if self.mid_price > 0 else 0

@dataclass
class OptionTrade:
    """Option trade data structure"""
    contract: OptionContract
    action: str  # "BUY" or "SELL"
    quantity: int
    price: float
    timestamp: datetime
    trade_id: str
    status: str = "PENDING"  # PENDING, EXECUTED, CANCELLED
    trade_setup: Optional[TradeSetup] = None
    exit_strategy: Optional[ExitStrategy] = None
    
    @property
    def total_value(self) -> float:
        """Calculate total trade value"""
        return self.quantity * self.price * self.contract.lot_size

class Nifty50OptionsTrader:
    """
    Main class for Nifty 50 options trading
    """
    
    def __init__(self, database_path: str = "nifty50_options.db"):
        """
        Initialize the options trader
        
        Args:
            database_path (str): Path to SQLite database
        """
        self.database_path = database_path
        self.initialize_database()
        self.current_positions = {}
        self.trade_history = []
        self.account_balance = 100000  # Default account balance
        self.active_trades = {}  # Track active trades with setups
        
        # Nifty 50 current level (will be updated)
        self.nifty50_current_level = 25000
        
        # Available strike prices (around current Nifty level)
        self.available_strikes = self._generate_strike_prices()
        
        # Available expiry dates
        self.expiry_dates = self._get_expiry_dates()
        
        # Risk management settings
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_portfolio_risk = 0.10  # 10% max portfolio risk
        self.trailing_stop_percentage = 0.05  # 5% trailing stop
        
        logger.info("Nifty50OptionsTrader initialized successfully")
    
    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create options contracts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS options_contracts (
                    contract_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    option_type TEXT NOT NULL,
                    expiry_date TEXT NOT NULL,
                    lot_size INTEGER NOT NULL,
                    underlying TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create options quotes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS options_quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT NOT NULL,
                    bid_price REAL NOT NULL,
                    ask_price REAL NOT NULL,
                    last_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    open_interest INTEGER NOT NULL,
                    implied_volatility REAL,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (contract_id) REFERENCES options_contracts (contract_id)
                )
            ''')
            
            # Create trades table with enhanced fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    contract_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    entry_price REAL,
                    stop_loss REAL,
                    target_price REAL,
                    exit_strategy TEXT,
                    exit_price REAL,
                    exit_timestamp TIMESTAMP,
                    pnl REAL,
                    FOREIGN KEY (contract_id) REFERENCES options_contracts (contract_id)
                )
            ''')
            
            # Create positions table with stop loss and target
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    contract_id TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    average_price REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    target_price REAL,
                    exit_strategy TEXT,
                    open_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (contract_id) REFERENCES options_contracts (contract_id)
                )
            ''')
            
            # Create trade setups table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_setups (
                    setup_id TEXT PRIMARY KEY,
                    trade_id TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    risk_reward_ratio REAL,
                    max_loss REAL,
                    max_profit REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades (trade_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _generate_strike_prices(self) -> List[float]:
        """Generate available strike prices around current Nifty level"""
        current_level = self.nifty50_current_level
        strikes = []
        
        # Generate strikes from 2000 points below to 2000 points above current level
        for i in range(-40, 41):  # -2000 to +2000 in steps of 50
            strike = current_level + (i * 50)
            if strike > 0:
                strikes.append(strike)
        
        return sorted(strikes)
    
    def _get_expiry_dates(self) -> List[date]:
        """Get available expiry dates for options"""
        today = date.today()
        expiry_dates = []
        
        # Weekly expiries (next 4 weeks)
        for i in range(1, 5):
            # Thursday expiry (Nifty options expire on Thursday)
            days_until_thursday = (3 - today.weekday()) % 7
            if days_until_thursday == 0:
                days_until_thursday = 7
            expiry_date = today + timedelta(days=days_until_thursday + (i-1)*7)
            expiry_dates.append(expiry_date)
        
        # Monthly expiry (next 3 months)
        for i in range(1, 4):
            # Last Thursday of the month
            next_month = today.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)
            last_day = next_month - timedelta(days=1)
            days_until_thursday = (3 - last_day.weekday()) % 7
            if days_until_thursday == 0:
                days_until_thursday = 7
            monthly_expiry = last_day - timedelta(days=days_until_thursday)
            if monthly_expiry > today:
                expiry_dates.append(monthly_expiry)
        
        # Quarterly expiry (next 2 quarters)
        for i in range(1, 3):
            quarter_month = 3 * i
            quarter_year = today.year
            if today.month > quarter_month:
                quarter_year += 1
            
            quarter_date = date(quarter_year, quarter_month, 1)
            last_day = quarter_date - timedelta(days=1)
            days_until_thursday = (3 - last_day.weekday()) % 7
            if days_until_thursday == 0:
                days_until_thursday = 7
            quarterly_expiry = last_day - timedelta(days=days_until_thursday)
            if quarterly_expiry > today:
                expiry_dates.append(quarterly_expiry)
        
        return sorted(list(set(expiry_dates)))
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, risk_amount: float) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            entry_price (float): Entry price of the option
            stop_loss (float): Stop loss price
            risk_amount (float): Maximum amount willing to risk
            
        Returns:
            int: Number of lots to trade
        """
        try:
            # Calculate risk per lot
            risk_per_lot = abs(entry_price - stop_loss) * 50  # 50 is lot size
            
            if risk_per_lot <= 0:
                return 0
            
            # Calculate position size based on risk
            position_size = int(risk_amount / risk_per_lot)
            
            # Ensure minimum position size
            if position_size < 1:
                position_size = 1
            
            # Check against account balance
            max_lots_by_balance = int(self.account_balance * 0.1 / (entry_price * 50))  # Max 10% of balance
            position_size = min(position_size, max_lots_by_balance)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1
    
    def place_option_order_with_setup(self, 
                                    contract: OptionContract, 
                                    action: str, 
                                    entry_price: float,
                                    stop_loss: float,
                                    target_price: float,
                                    quantity: Optional[int] = None,
                                    risk_amount: Optional[float] = None,
                                    order_type: str = "MARKET",
                                    limit_price: Optional[float] = None) -> Optional[OptionTrade]:
        """
        Place an option order with entry, stop loss, and target setup
        
        Args:
            contract (OptionContract): Option contract
            action (str): "BUY" or "SELL"
            entry_price (float): Entry price for the trade
            stop_loss (float): Stop loss price
            target_price (float): Target profit price
            quantity (int): Number of lots (if None, calculated from risk)
            risk_amount (float): Risk amount in rupees (if None, uses default)
            order_type (str): "MARKET" or "LIMIT"
            limit_price (float): Limit price for limit orders
            
        Returns:
            OptionTrade: Trade object if successful
        """
        try:
            # Validate setup
            if action == "BUY":
                if stop_loss >= entry_price:
                    raise ValueError("Stop loss must be below entry price for long positions")
                if target_price <= entry_price:
                    raise ValueError("Target price must be above entry price for long positions")
            else:  # SELL
                if stop_loss <= entry_price:
                    raise ValueError("Stop loss must be above entry price for short positions")
                if target_price >= entry_price:
                    raise ValueError("Target price must be below entry price for short positions")
            
            # Calculate position size if not provided
            if quantity is None:
                if risk_amount is None:
                    risk_amount = self.account_balance * self.max_risk_per_trade
                quantity = self.calculate_position_size(entry_price, stop_loss, risk_amount)
            
            # Create trade setup
            trade_setup = TradeSetup(
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                quantity=quantity,
                risk_reward_ratio=0,
                max_loss=0,
                max_profit=0
            )
            
            # Place the order
            trade = self.place_option_order(
                contract=contract,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price
            )
            
            if trade:
                # Add trade setup
                trade.trade_setup = trade_setup
                trade.exit_strategy = ExitStrategy.STOP_LOSS
                
                # Store in active trades
                self.active_trades[trade.trade_id] = {
                    'trade': trade,
                    'setup': trade_setup,
                    'entry_time': datetime.now(),
                    'last_check': datetime.now()
                }
                
                # Save setup to database
                self._save_trade_setup_to_db(trade, trade_setup)
                
                logger.info(f"Trade setup created: Entry: {entry_price}, SL: {stop_loss}, Target: {target_price}")
                return trade
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing option order with setup: {e}")
            return None
    
    def check_stop_loss_and_target(self, current_price: float) -> List[Dict]:
        """
        Check if any positions hit stop loss or target
        
        Args:
            current_price (float): Current market price
            
        Returns:
            List[Dict]: List of positions that need action
        """
        positions_to_exit = []
        
        for trade_id, trade_info in self.active_trades.items():
            trade = trade_info['trade']
            setup = trade_info['setup']
            
            if not setup:
                continue
            
            # Check stop loss
            if self._should_exit_stop_loss(trade, setup, current_price):
                positions_to_exit.append({
                    'trade_id': trade_id,
                    'reason': 'STOP_LOSS',
                    'current_price': current_price,
                    'setup': setup
                })
            
            # Check target
            elif self._should_exit_target(trade, setup, current_price):
                positions_to_exit.append({
                    'trade_id': trade_id,
                    'reason': 'TARGET_HIT',
                    'current_price': current_price,
                    'setup': setup
                })
        
        return positions_to_exit
    
    def _should_exit_stop_loss(self, trade: OptionTrade, setup: TradeSetup, current_price: float) -> bool:
        """Check if stop loss should trigger"""
        if trade.action == "BUY":
            return current_price <= setup.stop_loss
        else:  # SELL
            return current_price >= setup.stop_loss
    
    def _should_exit_target(self, trade: OptionTrade, setup: TradeSetup, current_price: float) -> bool:
        """Check if target should trigger"""
        if trade.action == "BUY":
            return current_price >= setup.target_price
        else:  # SELL
            return current_price <= setup.target_price
    
    def auto_exit_positions(self, current_price: float) -> List[OptionTrade]:
        """
        Automatically exit positions based on stop loss and target
        
        Args:
            current_price (float): Current market price
            
        Returns:
            List[OptionTrade]: List of exited trades
        """
        exited_trades = []
        positions_to_exit = self.check_stop_loss_and_target(current_price)
        
        for position in positions_to_exit:
            trade_id = position['trade_id']
            reason = position['reason']
            
            if self.close_position_by_trade_id(trade_id, reason=reason):
                trade_info = self.active_trades[trade_id]
                trade = trade_info['trade']
                trade.exit_strategy = ExitStrategy.STOP_LOSS if reason == 'STOP_LOSS' else ExitStrategy.TAKE_PROFIT
                exited_trades.append(trade)
                
                # Remove from active trades
                del self.active_trades[trade_id]
                
                logger.info(f"Auto-exited position {trade_id} due to {reason}")
        
        return exited_trades
    
    def close_position_by_trade_id(self, trade_id: str, reason: str = "MANUAL") -> bool:
        """
        Close position by trade ID
        
        Args:
            trade_id (str): Trade ID to close
            reason (str): Reason for closing
            
        Returns:
            bool: True if successful
        """
        try:
            if trade_id not in self.active_trades:
                return False
            
            trade_info = self.active_trades[trade_id]
            trade = trade_info['trade']
            
            # Get current quote for exit price
            current_quote = self.get_option_quote(trade.contract)
            if not current_quote:
                return False
            
            # Create exit trade
            exit_trade = OptionTrade(
                contract=trade.contract,
                action="SELL" if trade.action == "BUY" else "BUY",
                quantity=trade.quantity,
                price=current_quote.mid_price,
                timestamp=datetime.now(),
                trade_id=f"EXIT_{trade_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                exit_strategy=ExitStrategy.STOP_LOSS if reason == "STOP_LOSS" else ExitStrategy.TAKE_PROFIT
            )
            
            # Execute exit
            if self._execute_trade(exit_trade):
                exit_trade.status = "EXECUTED"
                self.trade_history.append(exit_trade)
                
                # Update original trade
                trade.exit_strategy = exit_trade.exit_strategy
                trade.status = "CLOSED"
                
                # Save exit details to database
                self._save_exit_details_to_db(trade, exit_trade, reason)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position by trade ID: {e}")
            return False
    
    def _save_trade_setup_to_db(self, trade: OptionTrade, setup: TradeSetup):
        """Save trade setup to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_setups 
                (setup_id, trade_id, entry_price, stop_loss, target_price, quantity, 
                 risk_reward_ratio, max_loss, max_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"SETUP_{trade.trade_id}",
                trade.trade_id,
                setup.entry_price,
                setup.stop_loss,
                setup.target_price,
                setup.quantity,
                setup.risk_reward_ratio,
                setup.max_loss,
                setup.max_profit
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade setup to database: {e}")
    
    def _save_exit_details_to_db(self, trade: OptionTrade, exit_trade: OptionTrade, reason: str):
        """Save exit details to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Calculate P&L
            if trade.action == "BUY":
                pnl = (exit_trade.price - trade.price) * trade.quantity * trade.contract.lot_size
            else:
                pnl = (trade.price - exit_trade.price) * trade.quantity * trade.contract.lot_size
            
            cursor.execute('''
                UPDATE trades 
                SET exit_strategy = ?, exit_price = ?, exit_timestamp = ?, pnl = ?
                WHERE trade_id = ?
            ''', (
                exit_trade.exit_strategy.value if exit_trade.exit_strategy else reason,
                exit_trade.price,
                exit_trade.timestamp.isoformat(),
                pnl,
                trade.trade_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving exit details to database: {e}")
    
    def get_trade_setup_summary(self) -> List[Dict]:
        """Get summary of all trade setups"""
        summary = []
        
        for trade_id, trade_info in self.active_trades.items():
            trade = trade_info['trade']
            setup = trade_info['setup']
            
            if setup:
                summary.append({
                    'trade_id': trade_id,
                    'contract': trade.contract.display_name,
                    'action': trade.action,
                    'entry_price': setup.entry_price,
                    'stop_loss': setup.stop_loss,
                    'target_price': setup.target_price,
                    'quantity': setup.quantity,
                    'risk_reward_ratio': setup.risk_reward_ratio,
                    'max_loss': setup.max_loss,
                    'max_profit': setup.max_profit,
                    'entry_time': trade_info['entry_time']
                })
        
        return summary
    
    def update_stop_loss(self, trade_id: str, new_stop_loss: float) -> bool:
        """
        Update stop loss for an active trade
        
        Args:
            trade_id (str): Trade ID to update
            new_stop_loss (float): New stop loss price
            
        Returns:
            bool: True if successful
        """
        try:
            if trade_id not in self.active_trades:
                return False
            
            trade_info = self.active_trades[trade_id]
            trade = trade_info['trade']
            setup = trade_info['setup']
            
            # Validate new stop loss
            if trade.action == "BUY":
                if new_stop_loss >= setup.entry_price:
                    raise ValueError("New stop loss must be below entry price for long positions")
            else:
                if new_stop_loss <= setup.entry_price:
                    raise ValueError("New stop loss must be above entry price for short positions")
            
            # Update stop loss
            setup.stop_loss = new_stop_loss
            
            # Update database
            self._update_stop_loss_in_db(trade_id, new_stop_loss)
            
            logger.info(f"Updated stop loss for trade {trade_id} to {new_stop_loss}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            return False
    
    def _update_stop_loss_in_db(self, trade_id: str, new_stop_loss: float):
        """Update stop loss in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE trade_setups 
                SET stop_loss = ?
                WHERE trade_id = ?
            ''', (new_stop_loss, trade_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating stop loss in database: {e}")
    
    def get_available_contracts(self, 
                               strike_range: Optional[Tuple[float, float]] = None,
                               expiry_filter: Optional[OptionExpiry] = None) -> List[OptionContract]:
        """
        Get available option contracts
        
        Args:
            strike_range (Tuple[float, float]): Min and max strike prices
            expiry_filter (OptionExpiry): Filter by expiry type
            
        Returns:
            List[OptionContract]: List of available contracts
        """
        contracts = []
        
        strikes = self.available_strikes
        if strike_range:
            min_strike, max_strike = strike_range
            strikes = [s for s in strikes if min_strike <= s <= max_strike]
        
        expiries = self.expiry_dates
        if expiry_filter:
            if expiry_filter == OptionExpiry.WEEKLY:
                expiries = [e for e in expiries if e <= date.today() + timedelta(days=30)]
            elif expiry_filter == OptionExpiry.MONTHLY:
                expiries = [e for e in expiries if date.today() + timedelta(days=30) < e <= date.today() + timedelta(days=90)]
            elif expiry_filter == OptionExpiry.QUARTERLY:
                expiries = [e for e in expiries if e > date.today() + timedelta(days=90)]
        
        for strike in strikes:
            for expiry in expiries:
                for option_type in [OptionType.CALL, OptionType.PUT]:
                    contract = OptionContract(
                        symbol=f"NIFTY{strike}{option_type.value}",
                        strike_price=strike,
                        option_type=option_type,
                        expiry_date=expiry
                    )
                    contracts.append(contract)
        
        return contracts
    
    def get_option_quote(self, contract: OptionContract) -> Optional[OptionQuote]:
        """
        Get option quote for a specific contract
        
        Args:
            contract (OptionContract): Option contract
            
        Returns:
            OptionQuote: Option quote data
        """
        try:
            # In a real implementation, this would fetch from a broker API
            # For demo purposes, we'll generate synthetic data
            
            # Calculate synthetic option price using Black-Scholes approximation
            spot_price = self.nifty50_current_level
            strike_price = contract.strike_price
            time_to_expiry = (contract.expiry_date - date.today()).days / 365
            
            # Simple option pricing (for demonstration)
            if contract.option_type == OptionType.CALL:
                intrinsic_value = max(0, spot_price - strike_price)
            else:
                intrinsic_value = max(0, strike_price - spot_price)
            
            # Add time value (simplified)
            time_value = max(0.1, intrinsic_value * 0.1 + time_to_expiry * 0.5)
            option_price = intrinsic_value + time_value
            
            # Generate synthetic quote
            spread = option_price * 0.05  # 5% spread
            bid_price = max(0.05, option_price - spread/2)
            ask_price = option_price + spread/2
            
            quote = OptionQuote(
                contract=contract,
                bid_price=round(bid_price, 2),
                ask_price=round(ask_price, 2),
                last_price=round(option_price, 2),
                volume=np.random.randint(100, 1000),
                open_interest=np.random.randint(500, 5000),
                implied_volatility=0.25 + np.random.random() * 0.2,
                delta=0.5 + np.random.random() * 0.4,
                gamma=0.01 + np.random.random() * 0.02,
                theta=-0.1 - np.random.random() * 0.2,
                vega=0.5 + np.random.random() * 0.5,
                timestamp=datetime.now()
            )
            
            return quote
            
        except Exception as e:
            logger.error(f"Error getting option quote: {e}")
            return None
    
    def place_option_order(self, 
                          contract: OptionContract, 
                          action: str, 
                          quantity: int, 
                          order_type: str = "MARKET",
                          limit_price: Optional[float] = None) -> Optional[OptionTrade]:
        """
        Place an option order
        
        Args:
            contract (OptionContract): Option contract
            action (str): "BUY" or "SELL"
            quantity (int): Number of lots
            order_type (str): "MARKET" or "LIMIT"
            limit_price (float): Limit price for limit orders
            
        Returns:
            OptionTrade: Trade object if successful
        """
        try:
            # Validate order
            if action not in ["BUY", "SELL"]:
                raise ValueError("Action must be 'BUY' or 'SELL'")
            
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            # Check account balance for buy orders
            if action == "BUY":
                quote = self.get_option_quote(contract)
                if quote:
                    required_amount = quantity * quote.ask_price * contract.lot_size
                    if required_amount > self.account_balance:
                        raise ValueError(f"Insufficient balance. Required: {required_amount}, Available: {self.account_balance}")
            
            # Generate trade ID
            trade_id = f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
            
            # Get execution price
            if order_type == "MARKET":
                quote = self.get_option_quote(contract)
                if quote:
                    execution_price = quote.ask_price if action == "BUY" else quote.bid_price
                else:
                    raise ValueError("Unable to get quote for market order")
            else:  # LIMIT order
                if limit_price is None:
                    raise ValueError("Limit price required for limit orders")
                execution_price = limit_price
            
            # Create trade
            trade = OptionTrade(
                contract=contract,
                action=action,
                quantity=quantity,
                price=execution_price,
                timestamp=datetime.now(),
                trade_id=trade_id
            )
            
            # Execute trade
            if self._execute_trade(trade):
                trade.status = "EXECUTED"
                self.trade_history.append(trade)
                self._update_positions(trade)
                self._save_trade_to_db(trade)
                logger.info(f"Trade executed: {trade}")
                return trade
            else:
                trade.status = "FAILED"
                logger.error(f"Trade execution failed: {trade}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing option order: {e}")
            return None
    
    def _execute_trade(self, trade: OptionTrade) -> bool:
        """Execute a trade (simulated)"""
        try:
            # Simulate trade execution
            if trade.action == "BUY":
                cost = trade.total_value
                if cost <= self.account_balance:
                    self.account_balance -= cost
                    return True
                else:
                    return False
            else:  # SELL
                # For selling, we need to check if we have the position
                position_key = trade.contract.contract_id
                if position_key in self.current_positions:
                    position = self.current_positions[position_key]
                    if position['quantity'] >= trade.quantity:
                        revenue = trade.total_value
                        self.account_balance += revenue
                        return True
                    else:
                        return False
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def _update_positions(self, trade: OptionTrade):
        """Update current positions after trade execution"""
        position_key = trade.contract.contract_id
        
        if trade.action == "BUY":
            if position_key in self.current_positions:
                # Update existing position
                position = self.current_positions[position_key]
                total_quantity = position['quantity'] + trade.quantity
                total_cost = (position['quantity'] * position['average_price'] + 
                            trade.quantity * trade.price)
                position['quantity'] = total_quantity
                position['average_price'] = total_cost / total_quantity
                position['last_update'] = datetime.now()
            else:
                # Create new position
                self.current_positions[position_key] = {
                    'contract': trade.contract,
                    'quantity': trade.quantity,
                    'average_price': trade.price,
                    'open_timestamp': trade.timestamp,
                    'last_update': trade.timestamp
                }
        
        else:  # SELL
            if position_key in self.current_positions:
                position = self.current_positions[position_key]
                position['quantity'] -= trade.quantity
                position['last_update'] = datetime.now()
                
                # Remove position if quantity becomes 0
                if position['quantity'] <= 0:
                    del self.current_positions[position_key]
    
    def _save_trade_to_db(self, trade: OptionTrade):
        """Save trade to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Save contract if not exists
            cursor.execute('''
                INSERT OR IGNORE INTO options_contracts 
                (contract_id, symbol, strike_price, option_type, expiry_date, lot_size, underlying)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.contract.contract_id,
                trade.contract.symbol,
                trade.contract.strike_price,
                trade.contract.option_type.value,
                trade.contract.expiry_date.isoformat(),
                trade.contract.lot_size,
                trade.contract.underlying
            ))
            
            # Save trade
            cursor.execute('''
                INSERT INTO trades 
                (trade_id, contract_id, action, quantity, price, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.trade_id,
                trade.contract.contract_id,
                trade.action,
                trade.quantity,
                trade.price,
                trade.timestamp.isoformat(),
                trade.status
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current open positions"""
        return self.current_positions
    
    def get_trade_history(self) -> List[OptionTrade]:
        """Get trade history"""
        return self.trade_history
    
    def get_account_summary(self) -> Dict[str, Union[float, int]]:
        """Get account summary"""
        total_positions = len(self.current_positions)
        total_trades = len(self.trade_history)
        
        # Calculate P&L for open positions
        unrealized_pnl = 0
        for position_key, position in self.current_positions.items():
            contract = position['contract']
            current_quote = self.get_option_quote(contract)
            if current_quote:
                current_value = position['quantity'] * current_quote.mid_price * contract.lot_size
                cost_basis = position['quantity'] * position['average_price'] * contract.lot_size
                unrealized_pnl += current_value - cost_basis
        
        return {
            'account_balance': self.account_balance,
            'total_positions': total_positions,
            'total_trades': total_trades,
            'unrealized_pnl': round(unrealized_pnl, 2),
            'total_value': self.account_balance + unrealized_pnl
        }
    
    def close_position(self, contract_id: str, quantity: Optional[int] = None) -> bool:
        """
        Close a position
        
        Args:
            contract_id (str): Contract ID to close
            quantity (int): Quantity to close (None for full position)
            
        Returns:
            bool: True if successful
        """
        try:
            if contract_id not in self.current_positions:
                raise ValueError(f"Position not found: {contract_id}")
            
            position = self.current_positions[contract_id]
            close_quantity = quantity if quantity else position['quantity']
            
            if close_quantity > position['quantity']:
                raise ValueError(f"Invalid quantity to close: {close_quantity}")
            
            # Create sell order
            trade = OptionTrade(
                contract=position['contract'],
                action="SELL",
                quantity=close_quantity,
                price=0,  # Will be set by market order
                timestamp=datetime.now(),
                trade_id=f"CLOSE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
            )
            
            # Execute close
            if self._execute_trade(trade):
                trade.status = "EXECUTED"
                self.trade_history.append(trade)
                self._update_positions(trade)
                self._save_trade_to_db(trade)
                logger.info(f"Position closed: {trade}")
                return True
            else:
                logger.error(f"Failed to close position: {contract_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_options_chain(self, expiry_date: Optional[date] = None) -> Dict[str, List[OptionQuote]]:
        """
        Get options chain for a specific expiry
        
        Args:
            expiry_date (date): Expiry date (None for all expiries)
            
        Returns:
            Dict[str, List[OptionQuote]]: Options chain organized by strike
        """
        options_chain = {}
        
        if expiry_date:
            contracts = [c for c in self.get_available_contracts() if c.expiry_date == expiry_date]
        else:
            contracts = self.get_available_contracts()
        
        for contract in contracts:
            quote = self.get_option_quote(contract)
            if quote:
                strike_key = str(contract.strike_price)
                if strike_key not in options_chain:
                    options_chain[strike_key] = []
                options_chain[strike_key].append(quote)
        
        return options_chain
    
    def calculate_payoff(self, 
                        contracts: List[OptionContract], 
                        quantities: List[int], 
                        spot_range: Tuple[float, float] = (24000, 26000),
                        spot_step: float = 100) -> pd.DataFrame:
        """
        Calculate payoff diagram for option positions
        
        Args:
            contracts (List[OptionContract]): List of option contracts
            quantities (List[int]): List of quantities (positive for long, negative for short)
            spot_range (Tuple[float, float]): Range of spot prices
            spot_step (float): Step size for spot prices
            
        Returns:
            pd.DataFrame: Payoff data
        """
        try:
            spot_prices = np.arange(spot_range[0], spot_range[1] + spot_step, spot_step)
            payoffs = []
            
            for spot in spot_prices:
                total_payoff = 0
                
                for contract, quantity in zip(contracts, quantities):
                    if contract.option_type == OptionType.CALL:
                        payoff = max(0, spot - contract.strike_price) * quantity * contract.lot_size
                    else:  # PUT
                        payoff = max(0, contract.strike_price - spot) * quantity * contract.lot_size
                    
                    total_payoff += payoff
                
                payoffs.append({
                    'spot_price': spot,
                    'payoff': total_payoff,
                    'breakeven': total_payoff == 0
                })
            
            return pd.DataFrame(payoffs)
            
        except Exception as e:
            logger.error(f"Error calculating payoff: {e}")
            return pd.DataFrame()

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize trader
    trader = Nifty50OptionsTrader()
    
    # Get available contracts
    contracts = trader.get_available_contracts(
        strike_range=(24800, 25200),
        expiry_filter=OptionExpiry.WEEKLY
    )
    
    print(f"Available contracts: {len(contracts)}")
    
    # Example: Buy 25000 CE with setup
    if contracts:
        # Find 25000 CE contract
        target_contract = None
        for contract in contracts:
            if (contract.strike_price == 25000 and 
                contract.option_type == OptionType.CALL):
                target_contract = contract
                break
        
        if target_contract:
            print(f"Found target contract: {target_contract.display_name}")
            
            # Get quote
            quote = trader.get_option_quote(target_contract)
            if quote:
                print(f"Quote: Bid: {quote.bid_price}, Ask: {quote.ask_price}")
                
                # Place order with setup (entry, stop loss, target)
                trade = trader.place_option_order_with_setup(
                    contract=target_contract,
                    action="BUY",
                    entry_price=quote.mid_price,
                    stop_loss=quote.mid_price * 0.8,  # 20% below entry
                    target_price=quote.mid_price * 1.5,  # 50% above entry
                    risk_amount=2000  # Risk ₹2000
                )
                
                if trade:
                    print(f"Trade executed with setup: {trade}")
                    print(f"Trade setup: {trade.trade_setup}")
                    
                    # Get trade setup summary
                    setup_summary = trader.get_trade_setup_summary()
                    print(f"Active trade setups: {len(setup_summary)}")
                    
                    # Simulate price movement and check stop loss/target
                    print("\nSimulating price movement...")
                    trader.nifty50_current_level = 25200  # Nifty moves up
                    
                    # Check if any positions need to exit
                    current_quote = trader.get_option_quote(target_contract)
                    if current_quote:
                        positions_to_exit = trader.check_stop_loss_and_target(current_quote.mid_price)
                        print(f"Positions to exit: {len(positions_to_exit)}")
                        
                        # Auto-exit positions
                        exited_trades = trader.auto_exit_positions(current_quote.mid_price)
                        print(f"Exited trades: {len(exited_trades)}")
                    
                    # Get account summary
                    summary = trader.get_account_summary()
                    print(f"Account summary: {summary}")
                    
                    # Get current positions
                    positions = trader.get_current_positions()
                    print(f"Current positions: {len(positions)}")
                    
                    # Close position manually
                    if trader.close_position(target_contract.contract_id):
                        print("Position closed successfully")
                    
                    # Final account summary
                    final_summary = trader.get_account_summary()
                    print(f"Final account summary: {final_summary}")
                else:
                    print("Trade failed")
            else:
                print("Unable to get quote")
        else:
            print("Target contract not found")
    
    print("Enhanced options trading demonstration completed")
