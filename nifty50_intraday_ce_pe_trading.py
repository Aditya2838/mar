"""
Enhanced Intraday Nifty 50 CE & PE Options Trading System
Specialized for both CALL and PUT options with entry/exit strategies and stop loss management
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time
from typing import Dict, List, Tuple, Optional, Union
import logging
import sqlite3
from dataclasses import dataclass
from enum import Enum

# Import base trading system
from nifty50_options_trading import (
    Nifty50OptionsTrader, 
    OptionContract, 
    OptionType, 
    OptionExpiry,
    OptionQuote,
    ExitStrategy,
    TradeSetup
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntradayStrategy(Enum):
    """Intraday trading strategies for CE & PE options"""
    MOMENTUM_BREAKOUT = "MOMENTUM_BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    GAP_TRADING = "GAP_TRADING"
    NEWS_BASED = "NEWS_BASED"
    TECHNICAL_BREAKOUT = "TECHNICAL_BREAKOUT"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    STRADDLE = "STRADDLE"                    # Buy both CE & PE
    STRANGLE = "STRANGLE"                    # Buy OTM CE & PE
    BUTTERFLY = "BUTTERFLY"                  # Complex spread
    IRON_CONDOR = "IRON_CONDOR"              # Sell spreads

class IntradayTimeSlot(Enum):
    """Intraday time slots"""
    PRE_MARKET = "PRE_MARKET"      # 9:00 - 9:15
    OPENING = "OPENING"            # 9:15 - 9:30
    MORNING = "MORNING"            # 9:30 - 11:00
    MID_DAY = "MID_DAY"            # 11:00 - 14:00
    AFTERNOON = "AFTERNOON"        # 14:00 - 15:00
    CLOSING = "CLOSING"            # 15:00 - 15:30

@dataclass
class CE_PE_TradeSetup:
    """CE & PE options trade setup with specific parameters"""
    entry_price: float
    stop_loss: float
    target_price: float
    quantity: int
    strategy: IntradayStrategy
    time_slot: IntradayTimeSlot
    entry_time: datetime
    max_holding_time: timedelta
    exit_time: datetime
    risk_reward_ratio: float
    max_loss: float
    max_profit: float
    intraday_stop_loss: float
    trailing_stop: bool
    trailing_stop_percentage: float
    option_type: OptionType  # CE or PE
    strike_price: float
    is_spread_trade: bool = False
    spread_legs: List[Dict] = None  # For complex strategies
    
    def __post_init__(self):
        """Calculate intraday metrics"""
        self.risk = abs(self.entry_price - self.stop_loss)
        self.reward = abs(self.target_price - self.entry_price)
        self.risk_reward_ratio = self.reward / self.risk if self.risk > 0 else 0
        self.max_loss = self.risk * self.quantity * 50
        self.max_profit = self.reward * self.quantity * 50
        
        if not self.exit_time:
            self.exit_time = self.entry_time + self.max_holding_time

class EnhancedIntradayNifty50Trader(Nifty50OptionsTrader):
    """
    Enhanced Nifty 50 options trader for intraday CE & PE trading
    """
    
    def __init__(self, database_path: str = "enhanced_intraday_nifty50.db"):
        """
        Initialize enhanced intraday trader
        
        Args:
            database_path (str): Path to SQLite database
        """
        super().__init__(database_path)
        
        # Market timing
        self.market_open_time = time(9, 15)
        self.market_close_time = time(15, 30)
        self.pre_market_start = time(9, 0)
        self.post_market_end = time(16, 0)
        
        # Enhanced risk management
        self.max_intraday_risk = 0.03            # 3% max risk per trade
        self.max_intraday_positions = 5          # Increased for CE & PE
        self.intraday_stop_loss_percentage = 0.15  # 15% stop loss
        self.intraday_target_percentage = 0.30    # 30% target
        self.max_holding_hours = 6
        self.max_spread_positions = 2            # Max complex strategies
        
        # CE & PE specific settings
        self.ce_pe_balance = True               # Balance CE & PE exposure
        self.max_ce_positions = 3               # Max CE positions
        self.max_pe_positions = 3               # Max PE positions
        self.spread_risk_multiplier = 1.5       # Higher risk for spreads
        
        # Tracking
        self.intraday_positions = {}
        self.intraday_trades = []
        self.daily_pnl = 0
        self.ce_positions = 0
        self.pe_positions = 0
        self.spread_positions = 0
        
        # Initialize enhanced database
        self.initialize_enhanced_intraday_database()
        
        logger.info("EnhancedIntradayNifty50Trader initialized successfully")
    
    def initialize_enhanced_intraday_database(self):
        """Initialize database with enhanced CE & PE tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Enhanced intraday trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_intraday_trades (
                    trade_id TEXT PRIMARY KEY,
                    contract_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target_price REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    time_slot TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    is_spread_trade BOOLEAN DEFAULT FALSE,
                    spread_legs TEXT,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl REAL,
                    holding_duration REAL,
                    FOREIGN KEY (contract_id) REFERENCES options_contracts (contract_id)
                )
            ''')
            
            # CE & PE positions tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ce_pe_positions (
                    position_id TEXT PRIMARY KEY,
                    trade_id TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    entry_time TIMESTAMP NOT NULL,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'ACTIVE',
                    FOREIGN KEY (trade_id) REFERENCES enhanced_intraday_trades (trade_id),
                    FOREIGN KEY (contract_id) REFERENCES options_contracts (contract_id)
                )
            ''')
            
            # Spread trades tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spread_trades (
                    spread_id TEXT PRIMARY KEY,
                    trade_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    legs TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    total_pnl REAL,
                    status TEXT DEFAULT 'ACTIVE',
                    FOREIGN KEY (trade_id) REFERENCES enhanced_intraday_trades (trade_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Enhanced intraday database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced database: {e}")
            raise
    
    def get_current_time_slot(self) -> IntradayTimeSlot:
        """Get current intraday time slot"""
        current_time = datetime.now().time()
        
        if self.pre_market_start <= current_time < self.market_open_time:
            return IntradayTimeSlot.PRE_MARKET
        elif self.market_open_time <= current_time < time(9, 30):
            return IntradayTimeSlot.OPENING
        elif time(9, 30) <= current_time < time(11, 0):
            return IntradayTimeSlot.MORNING
        elif time(11, 0) <= current_time < time(14, 0):
            return IntradayTimeSlot.MID_DAY
        elif time(14, 0) <= current_time < time(15, 0):
            return IntradayTimeSlot.AFTERNOON
        elif time(15, 0) <= current_time <= self.market_close_time:
            return IntradayTimeSlot.CLOSING
        else:
            return IntradayTimeSlot.PRE_MARKET
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        current_time = datetime.now().time()
        return self.market_open_time <= current_time <= self.market_close_time
    
    def can_place_ce_pe_trade(self, option_type: OptionType, is_spread: bool = False) -> Tuple[bool, str]:
        """Check if we can place a CE or PE trade"""
        if not self.is_market_open():
            return False, "Market is closed"
        
        if len(self.intraday_positions) >= self.max_intraday_positions:
            return False, f"Maximum intraday positions ({self.max_intraday_positions}) reached"
        
        # Check CE/PE balance
        if option_type == OptionType.CALL and self.ce_positions >= self.max_ce_positions:
            return False, f"Maximum CE positions ({self.max_ce_positions}) reached"
        elif option_type == OptionType.PUT and self.pe_positions >= self.max_pe_positions:
            return False, f"Maximum PE positions ({self.max_pe_positions}) reached"
        
        # Check spread limits
        if is_spread and self.spread_positions >= self.max_spread_positions:
            return False, f"Maximum spread positions ({self.max_spread_positions}) reached"
        
        # Check daily risk limit
        if abs(self.daily_pnl) > self.account_balance * 0.05:
            return False, "Daily loss limit reached"
        
        return True, "OK"
    
    def place_ce_pe_intraday_trade(self,
                                  contract: OptionContract,
                                  action: str,
                                  strategy: IntradayStrategy,
                                  time_slot: IntradayTimeSlot,
                                  entry_price: float,
                                  stop_loss_percentage: Optional[float] = None,
                                  target_percentage: Optional[float] = None,
                                  quantity: Optional[int] = None,
                                  risk_amount: Optional[float] = None,
                                  max_holding_hours: Optional[int] = None,
                                  is_spread: bool = False) -> Optional[Dict]:
        """
        Place an intraday CE or PE option trade
        
        Args:
            contract (OptionContract): Option contract (CE or PE)
            action (str): "BUY" or "SELL"
            strategy (IntradayStrategy): Trading strategy
            time_slot (IntradayTimeSlot): Entry time slot
            entry_price (float): Entry price
            stop_loss_percentage (float): Stop loss percentage
            target_percentage (float): Target percentage
            quantity (int): Number of lots
            risk_amount (float): Risk amount in rupees
            max_holding_hours (int): Maximum holding time
            is_spread (bool): Is this a spread trade
            
        Returns:
            Dict: Trade details if successful
        """
        try:
            # Validate trade placement
            can_trade, message = self.can_place_ce_pe_trade(contract.option_type, is_spread)
            if not can_trade:
                logger.warning(f"Cannot place CE/PE trade: {message}")
                return None
            
            # Set default values
            if stop_loss_percentage is None:
                stop_loss_percentage = self.intraday_stop_loss_percentage
            if target_percentage is None:
                target_percentage = self.intraday_target_percentage
            if max_holding_hours is None:
                max_holding_hours = self.max_holding_hours
            
            # Calculate stop loss and target based on option type
            if action == "BUY":
                if contract.option_type == OptionType.CALL:
                    stop_loss = entry_price * (1 - stop_loss_percentage)
                    target_price = entry_price * (1 + target_percentage)
                else:  # PUT
                    stop_loss = entry_price * (1 - stop_loss_percentage)
                    target_price = entry_price * (1 + target_percentage)
            else:  # SELL
                if contract.option_type == OptionType.CALL:
                    stop_loss = entry_price * (1 + stop_loss_percentage)
                    target_price = entry_price * (1 - target_percentage)
                else:  # PUT
                    stop_loss = entry_price * (1 + stop_loss_percentage)
                    target_price = entry_price * (1 - target_percentage)
            
            # Calculate position size
            if quantity is None:
                if risk_amount is None:
                    risk_amount = self.account_balance * self.max_intraday_risk
                    if is_spread:
                        risk_amount *= self.spread_risk_multiplier
                quantity = self.calculate_position_size(entry_price, stop_loss, risk_amount)
            
            # Create enhanced trade setup
            trade_setup = CE_PE_TradeSetup(
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                quantity=quantity,
                strategy=strategy,
                time_slot=time_slot,
                entry_time=datetime.now(),
                max_holding_time=timedelta(hours=max_holding_hours),
                exit_time=datetime.now() + timedelta(hours=max_holding_hours),
                risk_reward_ratio=0,
                max_loss=0,
                max_profit=0,
                intraday_stop_loss=stop_loss,
                trailing_stop=True,
                trailing_stop_percentage=0.05,
                option_type=contract.option_type,
                strike_price=contract.strike_price,
                is_spread_trade=is_spread
            )
            
            # Place the order
            trade = self.place_option_order(
                contract=contract,
                action=action,
                quantity=quantity,
                order_type="MARKET"
            )
            
            if trade:
                # Create enhanced trade record
                enhanced_trade = {
                    'trade_id': trade.trade_id,
                    'contract': contract,
                    'action': action,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_price': target_price,
                    'strategy': strategy,
                    'time_slot': time_slot,
                    'entry_time': datetime.now(),
                    'setup': trade_setup,
                    'is_spread': is_spread
                }
                
                # Store in intraday positions
                self.intraday_positions[trade.trade_id] = enhanced_trade
                
                # Update CE/PE counters
                if contract.option_type == OptionType.CALL:
                    self.ce_positions += 1
                else:
                    self.pe_positions += 1
                
                if is_spread:
                    self.spread_positions += 1
                
                # Save to database
                self._save_enhanced_intraday_trade_to_db(enhanced_trade)
                
                logger.info(f"Enhanced CE/PE trade placed: {action} {quantity} lots of {contract.display_name}")
                return enhanced_trade
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing enhanced CE/PE trade: {e}")
            return None
    
    def place_straddle_trade(self,
                           strike_price: float,
                           expiry_date: date,
                           time_slot: IntradayTimeSlot,
                           entry_price_ce: float,
                           entry_price_pe: float,
                           stop_loss_percentage: float = 0.15,
                           target_percentage: float = 0.30,
                           quantity: int = 1,
                           risk_amount: Optional[float] = None) -> Optional[Dict]:
        """
        Place a straddle trade (buy both CE & PE at same strike)
        
        Args:
            strike_price (float): Strike price for both options
            expiry_date (date): Expiry date
            time_slot (IntradayTimeSlot): Entry time slot
            entry_price_ce (float): CE entry price
            entry_price_pe (float): PE entry price
            stop_loss_percentage (float): Stop loss percentage
            target_percentage (float): Target percentage
            quantity (int): Number of lots
            risk_amount (float): Risk amount
            
        Returns:
            Dict: Straddle trade details
        """
        try:
            # Create CE contract
            ce_contract = OptionContract(
                symbol=f"NIFTY{strike_price}CE",
                strike_price=strike_price,
                option_type=OptionType.CALL,
                expiry_date=expiry_date
            )
            
            # Create PE contract
            pe_contract = OptionContract(
                symbol=f"NIFTY{strike_price}PE",
                strike_price=strike_price,
                option_type=OptionType.PUT,
                expiry_date=expiry_date
            )
            
            # Place CE trade
            ce_trade = self.place_ce_pe_intraday_trade(
                contract=ce_contract,
                action="BUY",
                strategy=IntradayStrategy.STRADDLE,
                time_slot=time_slot,
                entry_price=entry_price_ce,
                stop_loss_percentage=stop_loss_percentage,
                target_percentage=target_percentage,
                quantity=quantity,
                risk_amount=risk_amount,
                is_spread=True
            )
            
            # Place PE trade
            pe_trade = self.place_ce_pe_intraday_trade(
                contract=pe_contract,
                action="BUY",
                strategy=IntradayStrategy.STRADDLE,
                time_slot=time_slot,
                entry_price=entry_price_pe,
                stop_loss_percentage=stop_loss_percentage,
                target_percentage=target_percentage,
                quantity=quantity,
                risk_amount=risk_amount,
                is_spread=True
            )
            
            if ce_trade and pe_trade:
                straddle_trade = {
                    'straddle_id': f"STRADDLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'strike_price': strike_price,
                    'ce_trade': ce_trade,
                    'pe_trade': pe_trade,
                    'strategy': IntradayStrategy.STRADDLE,
                    'entry_time': datetime.now(),
                    'total_risk': (ce_trade['setup'].max_loss + pe_trade['setup'].max_loss),
                    'total_reward': (ce_trade['setup'].max_profit + pe_trade['setup'].max_profit)
                }
                
                logger.info(f"Straddle trade placed successfully: Strike {strike_price}")
                return straddle_trade
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing straddle trade: {e}")
            return None
    
    def place_strangle_trade(self,
                           ce_strike: float,
                           pe_strike: float,
                           expiry_date: date,
                           time_slot: IntradayTimeSlot,
                           entry_price_ce: float,
                           entry_price_pe: float,
                           stop_loss_percentage: float = 0.15,
                           target_percentage: float = 0.30,
                           quantity: int = 1,
                           risk_amount: Optional[float] = None) -> Optional[Dict]:
        """
        Place a strangle trade (buy OTM CE & PE)
        
        Args:
            ce_strike (float): CE strike price (OTM)
            pe_strike (float): PE strike price (OTM)
            expiry_date (date): Expiry date
            time_slot (IntradayTimeSlot): Entry time slot
            entry_price_ce (float): CE entry price
            entry_price_pe (float): PE entry price
            stop_loss_percentage (float): Stop loss percentage
            target_percentage (float): Target percentage
            quantity (int): Number of lots
            risk_amount (float): Risk amount
            
        Returns:
            Dict: Strangle trade details
        """
        try:
            # Create CE contract
            ce_contract = OptionContract(
                symbol=f"NIFTY{ce_strike}CE",
                strike_price=ce_strike,
                option_type=OptionType.CALL,
                expiry_date=expiry_date
            )
            
            # Create PE contract
            pe_contract = OptionContract(
                symbol=f"NIFTY{pe_strike}PE",
                strike_price=pe_strike,
                option_type=OptionType.PUT,
                expiry_date=expiry_date
            )
            
            # Place CE trade
            ce_trade = self.place_ce_pe_intraday_trade(
                contract=ce_contract,
                action="BUY",
                strategy=IntradayStrategy.STRANGLE,
                time_slot=time_slot,
                entry_price=entry_price_ce,
                stop_loss_percentage=stop_loss_percentage,
                target_percentage=target_percentage,
                quantity=quantity,
                risk_amount=risk_amount,
                is_spread=True
            )
            
            # Place PE trade
            pe_trade = self.place_ce_pe_intraday_trade(
                contract=pe_contract,
                action="BUY",
                strategy=IntradayStrategy.STRANGLE,
                time_slot=time_slot,
                entry_price=entry_price_pe,
                stop_loss_percentage=stop_loss_percentage,
                target_percentage=target_percentage,
                quantity=quantity,
                risk_amount=risk_amount,
                is_spread=True
            )
            
            if ce_trade and pe_trade:
                strangle_trade = {
                    'strangle_id': f"STRANGLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'ce_strike': ce_strike,
                    'pe_strike': pe_strike,
                    'ce_trade': ce_trade,
                    'pe_trade': pe_trade,
                    'strategy': IntradayStrategy.STRANGLE,
                    'entry_time': datetime.now(),
                    'total_risk': (ce_trade['setup'].max_loss + pe_trade['setup'].max_loss),
                    'total_reward': (ce_trade['setup'].max_profit + pe_trade['setup'].max_profit)
                }
                
                logger.info(f"Strangle trade placed successfully: CE {ce_strike}, PE {pe_strike}")
                return strangle_trade
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing strangle trade: {e}")
            return None
    
    def monitor_ce_pe_positions(self, current_price: float) -> List[Dict]:
        """
        Monitor all CE & PE positions for exit conditions
        
        Args:
            current_price (float): Current market price
            
        Returns:
            List[Dict]: Positions that need to exit
        """
        positions_to_exit = []
        current_time = datetime.now()
        
        for trade_id, position in self.intraday_positions.items():
            setup = position['setup']
            
            # Check stop loss
            if self._should_exit_ce_pe_stop_loss(position, current_price):
                positions_to_exit.append({
                    'trade_id': trade_id,
                    'reason': 'STOP_LOSS',
                    'current_price': current_price,
                    'position': position
                })
            
            # Check target
            elif self._should_exit_ce_pe_target(position, current_price):
                positions_to_exit.append({
                    'trade_id': trade_id,
                    'reason': 'TARGET_HIT',
                    'current_price': current_price,
                    'position': position
                })
            
            # Check time-based exit
            elif self._should_exit_time_based(position, current_time):
                positions_to_exit.append({
                    'trade_id': trade_id,
                    'reason': 'TIME_BASED',
                    'current_price': current_price,
                    'position': position
                })
            
            # Check trailing stop
            elif self._should_exit_trailing_stop(position, current_price):
                positions_to_exit.append({
                    'trade_id': trade_id,
                    'reason': 'TRAILING_STOP',
                    'current_price': current_price,
                    'position': position
                })
        
        return positions_to_exit
    
    def _should_exit_ce_pe_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if CE/PE stop loss should trigger"""
        setup = position['setup']
        if position['action'] == "BUY":
            return current_price <= setup.stop_loss
        else:  # SELL
            return current_price >= setup.stop_loss
    
    def _should_exit_ce_pe_target(self, position: Dict, current_price: float) -> bool:
        """Check if CE/PE target should trigger"""
        setup = position['setup']
        if position['action'] == "BUY":
            return current_price >= setup.target_price
        else:  # SELL
            return current_price <= setup.target_price
    
    def _should_exit_time_based(self, position: Dict, current_time: datetime) -> bool:
        """Check if time-based exit should trigger"""
        setup = position['setup']
        return current_time >= setup.exit_time
    
    def _should_exit_trailing_stop(self, position: Dict, current_price: float) -> bool:
        """Check if trailing stop should trigger"""
        if not position['setup'].trailing_stop:
            return False
        
        setup = position['setup']
        if position['action'] == "BUY":
            trailing_stop = current_price * (1 - setup.trailing_stop_percentage)
            return current_price <= trailing_stop
        else:
            trailing_stop = current_price * (1 + setup.trailing_stop_percentage)
            return current_price >= trailing_stop
    
    def exit_ce_pe_position(self, trade_id: str, reason: str, exit_price: float) -> bool:
        """
        Exit a CE or PE position
        
        Args:
            trade_id (str): Trade ID to exit
            reason (str): Reason for exit
            exit_price (float): Exit price
            
        Returns:
            bool: True if successful
        """
        try:
            if trade_id not in self.intraday_positions:
                return False
            
            position = self.intraday_positions[trade_id]
            contract = position['contract']
            
            # Calculate P&L
            if position['action'] == "BUY":
                pnl = (exit_price - position['entry_price']) * position['quantity'] * 50
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity'] * 50
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Update CE/PE counters
            if contract.option_type == OptionType.CALL:
                self.ce_positions = max(0, self.ce_positions - 1)
            else:
                self.pe_positions = max(0, self.pe_positions - 1)
            
            if position.get('is_spread', False):
                self.spread_positions = max(0, self.spread_positions - 1)
            
            # Create exit record
            exit_record = {
                'trade_id': trade_id,
                'exit_time': datetime.now(),
                'exit_price': exit_price,
                'exit_reason': reason,
                'pnl': pnl,
                'holding_duration': (datetime.now() - position['entry_time']).total_seconds() / 3600
            }
            
            # Save exit details
            self._save_enhanced_intraday_exit_to_db(exit_record)
            
            # Remove from active positions
            del self.intraday_positions[trade_id]
            
            # Add to trade history
            self.intraday_trades.append({**position, **exit_record})
            
            logger.info(f"CE/PE position exited: {trade_id}, Reason: {reason}, P&L: ₹{pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error exiting CE/PE position: {e}")
            return False
    
    def auto_exit_ce_pe_positions(self, current_price: float) -> List[Dict]:
        """
        Automatically exit CE & PE positions based on conditions
        
        Args:
            current_price (float): Current market price
            
        Returns:
            List[Dict]: Exited positions
        """
        exited_positions = []
        positions_to_exit = self.monitor_ce_pe_positions(current_price)
        
        for position_info in positions_to_exit:
            trade_id = position_info['trade_id']
            reason = position_info['reason']
            current_price = position_info['current_price']
            
            if self.exit_ce_pe_position(trade_id, reason, current_price):
                exited_positions.append(position_info)
        
        return exited_positions
    
    def get_enhanced_intraday_summary(self) -> Dict:
        """Get enhanced summary of CE & PE intraday trading"""
        active_positions = len(self.intraday_positions)
        total_trades = len(self.intraday_trades)
        
        # Calculate winning/losing trades
        winning_trades = len([t for t in self.intraday_trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.intraday_trades if t.get('pnl', 0) < 0])
        
        # Calculate average P&L
        total_pnl = sum(t.get('pnl', 0) for t in self.intraday_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = min([t.get('pnl', 0) for t in self.intraday_trades]) if self.intraday_trades else 0
        
        return {
            'active_positions': active_positions,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'average_pnl': avg_pnl,
            'max_drawdown': max_drawdown,
            'current_time_slot': self.get_current_time_slot().value,
            'market_open': self.is_market_open(),
            'ce_positions': self.ce_positions,
            'pe_positions': self.pe_positions,
            'spread_positions': self.spread_positions,
            'ce_pe_balance': self.ce_positions - self.pe_positions
        }
    
    def get_ce_pe_strategy_recommendations(self, current_time_slot: IntradayTimeSlot) -> List[Dict]:
        """
        Get CE & PE trading strategy recommendations based on time slot
        
        Args:
            current_time_slot (IntradayTimeSlot): Current time slot
            
        Returns:
            List[Dict]: Strategy recommendations
        """
        recommendations = []
        
        if current_time_slot == IntradayTimeSlot.OPENING:
            recommendations.extend([
                {
                    'strategy': IntradayStrategy.STRADDLE,
                    'description': 'Buy both CE & PE at same strike for gap trading',
                    'risk_level': 'HIGH',
                    'suitable_for': 'Experienced traders',
                    'strike_selection': 'At-the-money (ATM)'
                },
                {
                    'strategy': IntradayStrategy.MOMENTUM_BREAKOUT,
                    'description': 'Breakout trading in first 15 minutes',
                    'risk_level': 'MEDIUM',
                    'suitable_for': 'All traders',
                    'strike_selection': 'Near-the-money (NTM)'
                }
            ])
        
        elif current_time_slot == IntradayTimeSlot.MORNING:
            recommendations.extend([
                {
                    'strategy': IntradayStrategy.TECHNICAL_BREAKOUT,
                    'description': 'Technical breakout patterns',
                    'risk_level': 'MEDIUM',
                    'suitable_for': 'Technical traders',
                    'strike_selection': 'Support/Resistance levels'
                },
                {
                    'strategy': IntradayStrategy.STRANGLE,
                    'description': 'Buy OTM CE & PE for volatility expansion',
                    'risk_level': 'MEDIUM',
                    'suitable_for': 'Options traders',
                    'strike_selection': 'Out-of-the-money (OTM)'
                }
            ])
        
        elif current_time_slot == IntradayTimeSlot.MID_DAY:
            recommendations.extend([
                {
                    'strategy': IntradayStrategy.MEAN_REVERSION,
                    'description': 'Mean reversion trades',
                    'risk_level': 'LOW',
                    'suitable_for': 'Conservative traders',
                    'strike_selection': 'Moving average levels'
                },
                {
                    'strategy': IntradayStrategy.VOLATILITY_EXPANSION,
                    'description': 'Volatility-based trades',
                    'risk_level': 'MEDIUM',
                    'suitable_for': 'Options traders',
                    'strike_selection': 'Volatility bands'
                }
            ])
        
        elif current_time_slot == IntradayTimeSlot.CLOSING:
            recommendations.extend([
                {
                    'strategy': IntradayStrategy.MEAN_REVERSION,
                    'description': 'End-of-day mean reversion',
                    'risk_level': 'LOW',
                    'suitable_for': 'Conservative traders',
                    'strike_selection': 'Daily pivot points'
                }
            ])
        
        return recommendations
    
    def _save_enhanced_intraday_trade_to_db(self, trade: Dict):
        """Save enhanced intraday trade to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO enhanced_intraday_trades 
                (trade_id, contract_id, action, quantity, entry_price, stop_loss, target_price,
                 strategy, time_slot, option_type, strike_price, is_spread_trade, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['trade_id'],
                trade['contract'].contract_id,
                trade['action'],
                trade['quantity'],
                trade['entry_price'],
                trade['stop_loss'],
                trade['target_price'],
                trade['strategy'].value,
                trade['time_slot'].value,
                trade['contract'].option_type.value,
                trade['contract'].strike_price,
                trade.get('is_spread', False),
                trade['entry_time'].isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving enhanced intraday trade to database: {e}")
    
    def _save_enhanced_intraday_exit_to_db(self, exit_record: Dict):
        """Save enhanced intraday exit details to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE enhanced_intraday_trades 
                SET exit_time = ?, exit_price = ?, exit_reason = ?, pnl = ?, holding_duration = ?
                WHERE trade_id = ?
            ''', (
                exit_record['exit_time'].isoformat(),
                exit_record['exit_price'],
                exit_record['exit_reason'],
                exit_record['pnl'],
                exit_record['holding_duration'],
                exit_record['trade_id']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving enhanced intraday exit to database: {e}")

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize enhanced intraday trader
    trader = EnhancedIntradayNifty50Trader()
    
    print("🚀 Enhanced Intraday Nifty 50 CE & PE Options Trading System")
    print("=" * 70)
    
    # Check market status
    print(f"Market Open: {trader.is_market_open()}")
    print(f"Current Time Slot: {trader.get_current_time_slot().value}")
    
    # Get strategy recommendations
    current_slot = trader.get_current_time_slot()
    recommendations = trader.get_ce_pe_strategy_recommendations(current_slot)
    
    print(f"\nCE & PE Strategy Recommendations for {current_slot.value}:")
    for rec in recommendations:
        print(f"  {rec['strategy'].value}: {rec['description']}")
        print(f"    Risk: {rec['risk_level']} | Strike: {rec['strike_selection']}")
    
    # Get available contracts
    contracts = trader.get_available_contracts(
        strike_range=(24800, 25200),
        expiry_filter=OptionExpiry.WEEKLY
    )
    
    if contracts:
        # Find 25000 CE and PE contracts
        ce_contract = None
        pe_contract = None
        
        for contract in contracts:
            if contract.strike_price == 25000:
                if contract.option_type == OptionType.CALL:
                    ce_contract = contract
                elif contract.option_type == OptionType.PUT:
                    pe_contract = contract
        
        if ce_contract and pe_contract:
            print(f"\nFound contracts:")
            print(f"  CE: {ce_contract.display_name}")
            print(f"  PE: {pe_contract.display_name}")
            
            # Get quotes
            ce_quote = trader.get_option_quote(ce_contract)
            pe_quote = trader.get_option_quote(pe_contract)
            
            if ce_quote and pe_quote:
                print(f"\nCurrent Quotes:")
                print(f"  CE: ₹{ce_quote.mid_price:.2f}")
                print(f"  PE: ₹{pe_quote.mid_price:.2f}")
                
                # Place straddle trade
                print(f"\nPlacing Straddle Trade...")
                straddle = trader.place_straddle_trade(
                    strike_price=25000,
                    expiry_date=ce_contract.expiry_date,
                    time_slot=trader.get_current_time_slot(),
                    entry_price_ce=ce_quote.mid_price,
                    entry_price_pe=pe_quote.mid_price,
                    stop_loss_percentage=0.15,
                    target_percentage=0.30,
                    quantity=1,
                    risk_amount=5000
                )
                
                if straddle:
                    print(f"✅ Straddle trade placed successfully!")
                    print(f"   Strike: {straddle['strike_price']}")
                    print(f"   Total Risk: ₹{straddle['total_risk']:,.2f}")
                    print(f"   Total Reward: ₹{straddle['total_reward']:,.2f}")
                    
                    # Monitor positions
                    print(f"\nMonitoring positions...")
                    positions_to_exit = trader.monitor_ce_pe_positions(ce_quote.mid_price)
                    print(f"Positions to exit: {len(positions_to_exit)}")
                    
                    # Get summary
                    summary = trader.get_enhanced_intraday_summary()
                    print(f"\nEnhanced Summary:")
                    print(f"   CE Positions: {summary['ce_positions']}")
                    print(f"   PE Positions: {summary['pe_positions']}")
                    print(f"   Spread Positions: {summary['spread_positions']}")
                    print(f"   CE-PE Balance: {summary['ce_pe_balance']}")
                else:
                    print("❌ Straddle trade placement failed")
            else:
                print("❌ Unable to get quotes")
        else:
            print("❌ 25000 CE or PE contracts not found")
    
    print("\n🎯 Enhanced CE & PE intraday trading demonstration completed")
