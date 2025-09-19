"""
Intraday Nifty 50 Index Options Trading System
Specialized for intraday trading with entry/exit strategies and stop loss management
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
    """Intraday trading strategies"""
    MOMENTUM_BREAKOUT = "MOMENTUM_BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    GAP_TRADING = "GAP_TRADING"
    NEWS_BASED = "NEWS_BASED"
    TECHNICAL_BREAKOUT = "TECHNICAL_BREAKOUT"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"

class IntradayTimeSlot(Enum):
    """Intraday time slots"""
    PRE_MARKET = "PRE_MARKET"      # 9:00 - 9:15
    OPENING = "OPENING"            # 9:15 - 9:30
    MORNING = "MORNING"            # 9:30 - 11:00
    MID_DAY = "MID_DAY"            # 11:00 - 14:00
    AFTERNOON = "AFTERNOON"        # 14:00 - 15:00
    CLOSING = "CLOSING"            # 15:00 - 15:30

@dataclass
class IntradayTradeSetup:
    """Intraday trade setup with specific intraday parameters"""
    entry_price: float
    stop_loss: float
    target_price: float
    quantity: int
    strategy: IntradayStrategy
    time_slot: IntradayTimeSlot
    entry_time: datetime
    max_holding_time: timedelta  # Maximum time to hold position
    exit_time: datetime          # Planned exit time
    risk_reward_ratio: float
    max_loss: float
    max_profit: float
    intraday_stop_loss: float    # Intraday specific stop loss
    trailing_stop: bool          # Enable trailing stop
    trailing_stop_percentage: float
    
    def __post_init__(self):
        """Calculate intraday metrics"""
        self.risk = abs(self.entry_price - self.stop_loss)
        self.reward = abs(self.target_price - self.entry_price)
        self.risk_reward_ratio = self.reward / self.risk if self.risk > 0 else 0
        self.max_loss = self.risk * self.quantity * 50
        self.max_profit = self.reward * self.quantity * 50
        
        # Calculate exit time if not provided
        if not self.exit_time:
            self.exit_time = self.entry_time + self.max_holding_time

class IntradayNifty50Trader(Nifty50OptionsTrader):
    """
    Enhanced Nifty 50 options trader for intraday trading
    """
    
    def __init__(self, database_path: str = "intraday_nifty50.db"):
        """
        Initialize intraday trader
        
        Args:
            database_path (str): Path to SQLite database
        """
        super().__init__(database_path)
        
        # Intraday specific settings
        self.market_open_time = time(9, 15)      # 9:15 AM
        self.market_close_time = time(15, 30)    # 3:30 PM
        self.pre_market_start = time(9, 0)       # 9:00 AM
        self.post_market_end = time(16, 0)       # 4:00 PM
        
        # Intraday risk management
        self.max_intraday_risk = 0.03            # 3% max risk per intraday trade
        self.max_intraday_positions = 3          # Maximum concurrent intraday positions
        self.intraday_stop_loss_percentage = 0.15  # 15% stop loss for intraday
        self.intraday_target_percentage = 0.30    # 30% target for intraday
        self.max_holding_hours = 6               # Maximum holding time in hours
        
        # Intraday tracking
        self.intraday_positions = {}             # Active intraday positions
        self.intraday_trades = []                # Intraday trade history
        self.daily_pnl = 0                       # Daily P&L tracking
        
        # Initialize intraday database
        self.initialize_intraday_database()
        
        logger.info("IntradayNifty50Trader initialized successfully")
    
    def initialize_intraday_database(self):
        """Initialize database with intraday specific tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create intraday trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intraday_trades (
                    trade_id TEXT PRIMARY KEY,
                    contract_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target_price REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    time_slot TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl REAL,
                    holding_duration REAL,
                    FOREIGN KEY (contract_id) REFERENCES options_contracts (contract_id)
                )
            ''')
            
            # Create intraday positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intraday_positions (
                    position_id TEXT PRIMARY KEY,
                    trade_id TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    entry_time TIMESTAMP NOT NULL,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'ACTIVE',
                    FOREIGN KEY (trade_id) REFERENCES intraday_trades (trade_id),
                    FOREIGN KEY (contract_id) REFERENCES options_contracts (contract_id)
                )
            ''')
            
            # Create daily summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    max_risk_exposure REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Intraday database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing intraday database: {e}")
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
            return IntradayTimeSlot.PRE_MARKET  # Default for after hours
    
    def is_market_open(self) -> bool:
        """Check if market is currently open for trading"""
        current_time = datetime.now().time()
        return self.market_open_time <= current_time <= self.market_close_time
    
    def can_place_intraday_trade(self) -> Tuple[bool, str]:
        """Check if we can place an intraday trade"""
        if not self.is_market_open():
            return False, "Market is closed"
        
        if len(self.intraday_positions) >= self.max_intraday_positions:
            return False, f"Maximum intraday positions ({self.max_intraday_positions}) reached"
        
        # Check daily risk limit
        if abs(self.daily_pnl) > self.account_balance * 0.05:  # 5% daily loss limit
            return False, "Daily loss limit reached"
        
        return True, "OK"
    
    def place_intraday_option_order(self,
                                  contract: OptionContract,
                                  action: str,
                                  strategy: IntradayStrategy,
                                  time_slot: IntradayTimeSlot,
                                  entry_price: float,
                                  stop_loss_percentage: Optional[float] = None,
                                  target_percentage: Optional[float] = None,
                                  quantity: Optional[int] = None,
                                  risk_amount: Optional[float] = None,
                                  max_holding_hours: Optional[int] = None) -> Optional[Dict]:
        """
        Place an intraday option order with enhanced risk management
        
        Args:
            contract (OptionContract): Option contract
            action (str): "BUY" or "SELL"
            strategy (IntradayStrategy): Trading strategy
            time_slot (IntradayTimeSlot): Entry time slot
            entry_price (float): Entry price
            stop_loss_percentage (float): Stop loss percentage (default: 15%)
            target_percentage (float): Target percentage (default: 30%)
            quantity (int): Number of lots
            risk_amount (float): Risk amount in rupees
            max_holding_hours (int): Maximum holding time in hours
            
        Returns:
            Dict: Trade details if successful
        """
        try:
            # Validate trade placement
            can_trade, message = self.can_place_intraday_trade()
            if not can_trade:
                logger.warning(f"Cannot place intraday trade: {message}")
                return None
            
            # Set default values
            if stop_loss_percentage is None:
                stop_loss_percentage = self.intraday_stop_loss_percentage
            if target_percentage is None:
                target_percentage = self.intraday_target_percentage
            if max_holding_hours is None:
                max_holding_hours = self.max_holding_hours
            
            # Calculate stop loss and target
            if action == "BUY":
                stop_loss = entry_price * (1 - stop_loss_percentage)
                target_price = entry_price * (1 + target_percentage)
            else:  # SELL
                stop_loss = entry_price * (1 + stop_loss_percentage)
                target_price = entry_price * (1 - target_percentage)
            
            # Calculate position size
            if quantity is None:
                if risk_amount is None:
                    risk_amount = self.account_balance * self.max_intraday_risk
                quantity = self.calculate_position_size(entry_price, stop_loss, risk_amount)
            
            # Create intraday trade setup
            intraday_setup = IntradayTradeSetup(
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
                trailing_stop_percentage=0.05
            )
            
            # Place the order
            trade = self.place_option_order(
                contract=contract,
                action=action,
                quantity=quantity,
                order_type="MARKET"
            )
            
            if trade:
                # Create intraday trade record
                intraday_trade = {
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
                    'setup': intraday_setup
                }
                
                # Store in intraday positions
                self.intraday_positions[trade.trade_id] = intraday_trade
                
                # Save to database
                self._save_intraday_trade_to_db(intraday_trade)
                
                logger.info(f"Intraday trade placed: {action} {quantity} lots of {contract.display_name}")
                return intraday_trade
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing intraday option order: {e}")
            return None
    
    def monitor_intraday_positions(self, current_price: float) -> List[Dict]:
        """
        Monitor all intraday positions for exit conditions
        
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
            if self._should_exit_intraday_stop_loss(position, current_price):
                positions_to_exit.append({
                    'trade_id': trade_id,
                    'reason': 'STOP_LOSS',
                    'current_price': current_price,
                    'position': position
                })
            
            # Check target
            elif self._should_exit_intraday_target(position, current_price):
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
    
    def _should_exit_intraday_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if intraday stop loss should trigger"""
        setup = position['setup']
        if position['action'] == "BUY":
            return current_price <= setup.stop_loss
        else:  # SELL
            return current_price >= setup.stop_loss
    
    def _should_exit_intraday_target(self, position: Dict, current_price: float) -> bool:
        """Check if intraday target should trigger"""
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
        # Calculate trailing stop based on highest/lowest price reached
        # This is a simplified version - in real implementation, track highest/lowest
        
        if position['action'] == "BUY":
            # For long positions, trailing stop moves up
            trailing_stop = current_price * (1 - setup.trailing_stop_percentage)
            return current_price <= trailing_stop
        else:
            # For short positions, trailing stop moves down
            trailing_stop = current_price * (1 + setup.trailing_stop_percentage)
            return current_price >= trailing_stop
    
    def exit_intraday_position(self, trade_id: str, reason: str, exit_price: float) -> bool:
        """
        Exit an intraday position
        
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
            
            # Calculate P&L
            if position['action'] == "BUY":
                pnl = (exit_price - position['entry_price']) * position['quantity'] * 50
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity'] * 50
            
            # Update daily P&L
            self.daily_pnl += pnl
            
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
            self._save_intraday_exit_to_db(exit_record)
            
            # Remove from active positions
            del self.intraday_positions[trade_id]
            
            # Add to trade history
            self.intraday_trades.append({**position, **exit_record})
            
            logger.info(f"Intraday position exited: {trade_id}, Reason: {reason}, P&L: ₹{pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error exiting intraday position: {e}")
            return False
    
    def auto_exit_intraday_positions(self, current_price: float) -> List[Dict]:
        """
        Automatically exit intraday positions based on conditions
        
        Args:
            current_price (float): Current market price
            
        Returns:
            List[Dict]: Exited positions
        """
        exited_positions = []
        positions_to_exit = self.monitor_intraday_positions(current_price)
        
        for position_info in positions_to_exit:
            trade_id = position_info['trade_id']
            reason = position_info['reason']
            current_price = position_info['current_price']
            
            if self.exit_intraday_position(trade_id, reason, current_price):
                exited_positions.append(position_info)
        
        return exited_positions
    
    def get_intraday_summary(self) -> Dict:
        """Get summary of intraday trading activity"""
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
            'market_open': self.is_market_open()
        }
    
    def force_close_all_intraday_positions(self, reason: str = "MARKET_CLOSE") -> List[Dict]:
        """
        Force close all intraday positions (e.g., at market close)
        
        Args:
            reason (str): Reason for force close
            
        Returns:
            List[Dict]: Closed positions
        """
        closed_positions = []
        
        for trade_id in list(self.intraday_positions.keys()):
            position = self.intraday_positions[trade_id]
            
            # Get current quote for exit price
            current_quote = self.get_option_quote(position['contract'])
            if current_quote:
                exit_price = current_quote.mid_price
                
                if self.exit_intraday_position(trade_id, reason, exit_price):
                    closed_positions.append({
                        'trade_id': trade_id,
                        'reason': reason,
                        'exit_price': exit_price
                    })
        
        return closed_positions
    
    def _save_intraday_trade_to_db(self, trade: Dict):
        """Save intraday trade to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO intraday_trades 
                (trade_id, contract_id, action, quantity, entry_price, stop_loss, target_price,
                 strategy, time_slot, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                trade['entry_time'].isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving intraday trade to database: {e}")
    
    def _save_intraday_exit_to_db(self, exit_record: Dict):
        """Save intraday exit details to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE intraday_trades 
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
            logger.error(f"Error saving intraday exit to database: {e}")
    
    def get_intraday_strategy_recommendations(self, current_time_slot: IntradayTimeSlot) -> List[Dict]:
        """
        Get trading strategy recommendations based on time slot
        
        Args:
            current_time_slot (IntradayTimeSlot): Current time slot
            
        Returns:
            List[Dict]: Strategy recommendations
        """
        recommendations = []
        
        if current_time_slot == IntradayTimeSlot.OPENING:
            recommendations.extend([
                {
                    'strategy': IntradayStrategy.GAP_TRADING,
                    'description': 'Trade gaps from previous day close',
                    'risk_level': 'HIGH',
                    'suitable_for': 'Experienced traders'
                },
                {
                    'strategy': IntradayStrategy.MOMENTUM_BREAKOUT,
                    'description': 'Breakout trading in first 15 minutes',
                    'risk_level': 'MEDIUM',
                    'suitable_for': 'All traders'
                }
            ])
        
        elif current_time_slot == IntradayTimeSlot.MORNING:
            recommendations.extend([
                {
                    'strategy': IntradayStrategy.TECHNICAL_BREAKOUT,
                    'description': 'Technical breakout patterns',
                    'risk_level': 'MEDIUM',
                    'suitable_for': 'Technical traders'
                },
                {
                    'strategy': IntradayStrategy.MOMENTUM_BREAKOUT,
                    'description': 'Momentum continuation trades',
                    'risk_level': 'MEDIUM',
                    'suitable_for': 'All traders'
                }
            ])
        
        elif current_time_slot == IntradayTimeSlot.MID_DAY:
            recommendations.extend([
                {
                    'strategy': IntradayStrategy.MEAN_REVERSION,
                    'description': 'Mean reversion trades',
                    'risk_level': 'LOW',
                    'suitable_for': 'Conservative traders'
                },
                {
                    'strategy': IntradayStrategy.VOLATILITY_EXPANSION,
                    'description': 'Volatility-based trades',
                    'risk_level': 'MEDIUM',
                    'suitable_for': 'Options traders'
                }
            ])
        
        elif current_time_slot == IntradayTimeSlot.CLOSING:
            recommendations.extend([
                {
                    'strategy': IntradayStrategy.MEAN_REVERSION,
                    'description': 'End-of-day mean reversion',
                    'risk_level': 'LOW',
                    'suitable_for': 'Conservative traders'
                }
            ])
        
        return recommendations

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize intraday trader
    trader = IntradayNifty50Trader()
    
    print("🚀 Intraday Nifty 50 Options Trading System")
    print("=" * 60)
    
    # Check market status
    print(f"Market Open: {trader.is_market_open()}")
    print(f"Current Time Slot: {trader.get_current_time_slot().value}")
    
    # Get strategy recommendations
    current_slot = trader.get_current_time_slot()
    recommendations = trader.get_intraday_strategy_recommendations(current_slot)
    
    print(f"\nStrategy Recommendations for {current_slot.value}:")
    for rec in recommendations:
        print(f"  {rec['strategy'].value}: {rec['description']} (Risk: {rec['risk_level']})")
    
    # Get available contracts
    contracts = trader.get_available_contracts(
        strike_range=(24800, 25200),
        expiry_filter=OptionExpiry.WEEKLY
    )
    
    if contracts:
        # Find 25000 CE contract
        target_contract = None
        for contract in contracts:
            if (contract.strike_price == 25000 and 
                contract.option_type == OptionType.CALL):
                target_contract = contract
                break
        
        if target_contract:
            print(f"\nTarget Contract: {target_contract.display_name}")
            
            # Get quote
            quote = trader.get_option_quote(target_contract)
            if quote:
                print(f"Current Quote: ₹{quote.mid_price:.2f}")
                
                # Place intraday trade
                trade = trader.place_intraday_option_order(
                    contract=target_contract,
                    action="BUY",
                    strategy=IntradayStrategy.MOMENTUM_BREAKOUT,
                    time_slot=trader.get_current_time_slot(),
                    entry_price=quote.mid_price,
                    stop_loss_percentage=0.15,
                    target_percentage=0.30,
                    risk_amount=3000
                )
                
                if trade:
                    print(f"✅ Intraday trade placed successfully!")
                    print(f"   Trade ID: {trade['trade_id']}")
                    print(f"   Strategy: {trade['strategy'].value}")
                    print(f"   Entry: ₹{trade['entry_price']:.2f}")
                    print(f"   SL: ₹{trade['stop_loss']:.2f}")
                    print(f"   Target: ₹{trade['target_price']:.2f}")
                    
                    # Monitor positions
                    print("\nMonitoring positions...")
                    positions_to_exit = trader.monitor_intraday_positions(quote.mid_price)
                    print(f"Positions to exit: {len(positions_to_exit)}")
                    
                    # Get summary
                    summary = trader.get_intraday_summary()
                    print(f"\nIntraday Summary:")
                    for key, value in summary.items():
                        print(f"   {key}: {value}")
                else:
                    print("❌ Trade placement failed")
            else:
                print("❌ Unable to get quote")
        else:
            print("❌ Target contract not found")
    
    print("\n🎯 Intraday trading demonstration completed")
