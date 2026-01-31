"""
Trade Audit Logging Module

Institutional-grade trade logging with:
- Structured JSON logs
- Full audit trail
- Decision reasoning capture
- Compliance-ready formatting
- Async-safe logging
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import threading
from queue import Queue
import atexit

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    """Trade action types."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"
    REBALANCE = "rebalance"
    TRIM = "trim"
    ADD = "add"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DecisionSource(Enum):
    """Source of trade decision."""
    MODEL_SIGNAL = "model_signal"
    RISK_LIMIT = "risk_limit"
    REBALANCE = "rebalance"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MANUAL = "manual"
    LIQUIDATION = "liquidation"


@dataclass
class SignalContext:
    """Context about the signal that triggered the trade."""
    model_name: str
    signal_value: float
    signal_timestamp: datetime
    confidence: Optional[float] = None
    raw_scores: Dict[str, float] = field(default_factory=dict)
    features_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["signal_timestamp"] = self.signal_timestamp.isoformat()
        return d


@dataclass
class RiskContext:
    """Risk metrics at time of trade."""
    portfolio_value: float
    position_weight_before: float
    position_weight_after: float
    sector_exposure: Optional[float] = None
    portfolio_beta: Optional[float] = None
    var_contribution: Optional[float] = None
    drawdown_current: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeDecision:
    """Reasoning behind trade decision."""
    source: DecisionSource
    rationale: str
    factors_considered: List[str] = field(default_factory=list)
    alternatives_rejected: List[str] = field(default_factory=list)
    constraints_active: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["source"] = self.source.value
        return d


@dataclass
class TradeRecord:
    """Complete trade record for audit trail."""
    # Identifiers
    trade_id: str
    order_id: Optional[str] = None
    parent_trade_id: Optional[str] = None  # For linked trades
    
    # Timestamps
    decision_timestamp: datetime = field(default_factory=datetime.now)
    submission_timestamp: Optional[datetime] = None
    execution_timestamp: Optional[datetime] = None
    
    # Trade details
    ticker: str = ""
    action: TradeAction = TradeAction.BUY
    quantity: float = 0
    price: Optional[float] = None
    notional: Optional[float] = None
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    
    # Context
    signal_context: Optional[SignalContext] = None
    risk_context: Optional[RiskContext] = None
    decision: Optional[TradeDecision] = None
    
    # Metadata
    strategy: str = "default"
    account: str = "primary"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "parent_trade_id": self.parent_trade_id,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "submission_timestamp": self.submission_timestamp.isoformat() if self.submission_timestamp else None,
            "execution_timestamp": self.execution_timestamp.isoformat() if self.execution_timestamp else None,
            "ticker": self.ticker,
            "action": self.action.value,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "status": self.status.value,
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "commission": self.commission,
            "slippage": self.slippage,
            "signal_context": self.signal_context.to_dict() if self.signal_context else None,
            "risk_context": self.risk_context.to_dict() if self.risk_context else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "strategy": self.strategy,
            "account": self.account,
            "tags": self.tags,
            "notes": self.notes,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }
        return d
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class TradeAuditLogger:
    """
    Institutional-grade trade audit logger.
    
    Features:
    - Structured JSON logging
    - Async-safe write queue
    - Automatic log rotation
    - Compliance-ready format
    - Real-time and batch modes
    
    Usage:
        audit = TradeAuditLogger(log_dir="./logs/trades")
        
        # Log a trade
        record = TradeRecord(
            trade_id=audit.generate_trade_id(),
            ticker="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price=150.00,
            decision=TradeDecision(
                source=DecisionSource.MODEL_SIGNAL,
                rationale="Strong early adopter signal with high confidence"
            )
        )
        audit.log_trade(record)
        
        # Update status
        audit.update_status(record.trade_id, OrderStatus.FILLED, fill_price=150.05)
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "./logs/trades",
        rotate_daily: bool = True,
        async_write: bool = True,
        max_queue_size: int = 1000,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.rotate_daily = rotate_daily
        self.async_write = async_write
        
        # Trade cache for updates
        self._trades: Dict[str, TradeRecord] = {}
        self._lock = threading.Lock()
        
        # Async write queue
        if async_write:
            self._queue: Queue = Queue(maxsize=max_queue_size)
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
            atexit.register(self._flush_queue)
        
        # Current log file
        self._current_log_file: Optional[Path] = None
        self._current_date: Optional[str] = None
        
        logger.info(f"TradeAuditLogger initialized: {self.log_dir}")
    
    def generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique = uuid.uuid4().hex[:8]
        return f"TRD-{timestamp}-{unique}"
    
    def log_trade(self, record: TradeRecord) -> str:
        """
        Log a trade record.
        
        Returns:
            trade_id for reference
        """
        # Cache the record
        with self._lock:
            self._trades[record.trade_id] = record
        
        # Write to log
        if self.async_write:
            self._queue.put(("trade", record))
        else:
            self._write_record(record)
        
        logger.info(f"Trade logged: {record.trade_id} {record.action.value} {record.ticker}")
        
        return record.trade_id
    
    def update_status(
        self,
        trade_id: str,
        status: OrderStatus,
        fill_price: Optional[float] = None,
        fill_quantity: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> Optional[TradeRecord]:
        """
        Update trade status.
        
        Returns:
            Updated TradeRecord or None if not found
        """
        with self._lock:
            record = self._trades.get(trade_id)
            if not record:
                logger.warning(f"Trade not found for update: {trade_id}")
                return None
            
            record.status = status
            
            if status == OrderStatus.FILLED:
                record.execution_timestamp = datetime.now()
            
            if fill_price is not None:
                record.fill_price = fill_price
                if record.price:
                    record.slippage = fill_price - record.price
            
            if fill_quantity is not None:
                record.fill_quantity = fill_quantity
            
            if error_message:
                record.error_message = error_message
        
        # Log the update
        if self.async_write:
            self._queue.put(("update", record))
        else:
            self._write_update(record)
        
        logger.info(f"Trade updated: {trade_id} -> {status.value}")
        
        return record
    
    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Get trade record by ID."""
        with self._lock:
            return self._trades.get(trade_id)
    
    def get_trades_by_ticker(self, ticker: str) -> List[TradeRecord]:
        """Get all trades for a ticker."""
        with self._lock:
            return [t for t in self._trades.values() if t.ticker == ticker]
    
    def get_trades_by_date(self, date: datetime) -> List[TradeRecord]:
        """Get all trades for a date."""
        target_date = date.date()
        with self._lock:
            return [
                t for t in self._trades.values()
                if t.decision_timestamp.date() == target_date
            ]
    
    def _get_log_file(self) -> Path:
        """Get current log file path."""
        if self.rotate_daily:
            date_str = datetime.now().strftime("%Y-%m-%d")
            if date_str != self._current_date:
                self._current_date = date_str
                self._current_log_file = self.log_dir / f"trades_{date_str}.jsonl"
        else:
            if self._current_log_file is None:
                self._current_log_file = self.log_dir / "trades.jsonl"
        
        return self._current_log_file
    
    def _write_record(self, record: TradeRecord):
        """Write trade record to log file."""
        log_file = self._get_log_file()
        
        entry = {
            "event": "trade",
            "timestamp": datetime.now().isoformat(),
            "data": record.to_dict(),
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def _write_update(self, record: TradeRecord):
        """Write trade update to log file."""
        log_file = self._get_log_file()
        
        entry = {
            "event": "update",
            "timestamp": datetime.now().isoformat(),
            "trade_id": record.trade_id,
            "status": record.status.value,
            "fill_price": record.fill_price,
            "fill_quantity": record.fill_quantity,
            "slippage": record.slippage,
            "error": record.error_message,
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def _writer_loop(self):
        """Background writer thread."""
        while True:
            try:
                event_type, record = self._queue.get()
                
                if event_type == "trade":
                    self._write_record(record)
                elif event_type == "update":
                    self._write_update(record)
                elif event_type == "shutdown":
                    break
                
                self._queue.task_done()
            except Exception as e:
                logger.error(f"Writer thread error: {e}")
    
    def _flush_queue(self):
        """Flush pending writes on shutdown."""
        if self.async_write:
            self._queue.put(("shutdown", None))
            self._queue.join()
    
    def export_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json",
    ) -> Union[str, List[Dict]]:
        """
        Export trades for compliance/reporting.
        
        Args:
            start_date: Start of period
            end_date: End of period
            format: 'json' or 'csv'
            
        Returns:
            Exported data
        """
        with self._lock:
            trades = list(self._trades.values())
        
        # Filter by date
        if start_date:
            trades = [t for t in trades if t.decision_timestamp >= start_date]
        if end_date:
            trades = [t for t in trades if t.decision_timestamp <= end_date]
        
        # Sort by timestamp
        trades.sort(key=lambda t: t.decision_timestamp)
        
        if format == "json":
            return json.dumps([t.to_dict() for t in trades], indent=2)
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if trades:
                writer = csv.DictWriter(output, fieldnames=trades[0].to_dict().keys())
                writer.writeheader()
                for t in trades:
                    writer.writerow(t.to_dict())
            
            return output.getvalue()
        else:
            return [t.to_dict() for t in trades]
    
    def generate_daily_summary(self, date: Optional[datetime] = None) -> Dict:
        """Generate daily trade summary."""
        date = date or datetime.now()
        trades = self.get_trades_by_date(date)
        
        summary = {
            "date": date.strftime("%Y-%m-%d"),
            "total_trades": len(trades),
            "buys": len([t for t in trades if t.action == TradeAction.BUY]),
            "sells": len([t for t in trades if t.action == TradeAction.SELL]),
            "filled": len([t for t in trades if t.status == OrderStatus.FILLED]),
            "rejected": len([t for t in trades if t.status == OrderStatus.REJECTED]),
            "total_notional": sum(t.notional or 0 for t in trades),
            "total_commission": sum(t.commission or 0 for t in trades),
            "avg_slippage": (
                sum(t.slippage or 0 for t in trades) / len(trades) if trades else 0
            ),
            "tickers_traded": list(set(t.ticker for t in trades)),
            "by_source": {},
        }
        
        # Count by decision source
        for t in trades:
            if t.decision:
                source = t.decision.source.value
                summary["by_source"][source] = summary["by_source"].get(source, 0) + 1
        
        return summary


# Convenience function for quick logging
_default_logger: Optional[TradeAuditLogger] = None


def get_trade_logger() -> TradeAuditLogger:
    """Get or create default trade logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = TradeAuditLogger()
    return _default_logger


def log_trade(
    ticker: str,
    action: TradeAction,
    quantity: float,
    price: float,
    rationale: str,
    signal_value: Optional[float] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Convenience function to log a trade.
    
    Returns:
        trade_id
    """
    audit = get_trade_logger()
    
    signal_context = None
    if signal_value is not None and model_name:
        signal_context = SignalContext(
            model_name=model_name,
            signal_value=signal_value,
            signal_timestamp=datetime.now(),
        )
    
    record = TradeRecord(
        trade_id=audit.generate_trade_id(),
        ticker=ticker,
        action=action,
        quantity=quantity,
        price=price,
        notional=quantity * price,
        signal_context=signal_context,
        decision=TradeDecision(
            source=DecisionSource.MODEL_SIGNAL if model_name else DecisionSource.MANUAL,
            rationale=rationale,
        ),
    )
    
    return audit.log_trade(record)
