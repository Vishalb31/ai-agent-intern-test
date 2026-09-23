import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    name: str
    quantity: int
    final_sale: bool = False


class SafeOrderResult(BaseModel):
    order_id: str
    status: str
    membership_tier: Optional[str] = None
    placed_at: Optional[str] = None
    status_updated_at: Optional[str] = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    items: List[OrderItem] = Field(default_factory=list)
    requires_human_handoff: bool = False
    action_supported: bool = False  # Dataset is strictly read-only lookup


class OrderLookupService:
    def __init__(self, data_path: Path | str = "data/orders.json"):
        self.data_path = Path(data_path)
        self._load_data()

    def _load_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Orders file not found at {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        self.snapshot_at: str = self.raw_data.get("snapshot_at", "")
        # Index orders by normalized order_id for O(1) retrieval
        orders_list = self.raw_data.get("orders", [])
        self.orders: Dict[str, Dict[str, Any]] = {
            self.normalize_id(o["order_id"]): o for o in orders_list if "order_id" in o
        }

    @staticmethod
    def normalize_id(order_id: Optional[str]) -> str:
        """Strip surrounding spaces, lowercase variants, and non-alphanumeric punctuation."""
        if not order_id:
            return ""
        cleaned = re.sub(r"[^A-Za-z0-9-]", "", order_id.strip())
        return cleaned.upper()

    def lookup_order(self, order_id: Optional[str]) -> Dict[str, Any]:
        """
        Public lookup function returning only customer-safe fields.
        Returns a dictionary suitable for LLM tool consumption.
        """
        if not order_id:
            return {
                "found": False,
                "error": "Missing order ID. Please provide a valid order ID (e.g., ORD-1007)."
            }

        norm_id = self.normalize_id(order_id)
        order = self.orders.get(norm_id)

        if not order:
            return {
                "found": False,
                "error": f"Order {norm_id} was not found in the system."
            }

        status = str(order.get("status", "")).lower()

        # Handle stale delivery dates on terminal statuses
        is_terminal = status in ["cancelled", "returned"]
        estimated_delivery = None if is_terminal else order.get("estimated_delivery")
        carrier = None if is_terminal else order.get("carrier")
        tracking_number = None if is_terminal else order.get("tracking_number")

        # Map customer-safe items
        items = [
            OrderItem(
                name=item.get("name", "Unknown Item"),
                quantity=item.get("quantity", 1),
                final_sale=bool(item.get("final_sale", False))
            )
            for item in order.get("items", [])
        ]

        # Status: exception requires human intervention
        requires_handoff = (status == "exception")

        safe_result = SafeOrderResult(
            order_id=order["order_id"],
            status=order.get("status", "unknown"),
            membership_tier=order.get("membership_tier"),
            placed_at=order.get("placed_at"),
            status_updated_at=order.get("status_updated_at"),
            shipped_at=order.get("shipped_at"),
            delivered_at=order.get("delivered_at"),
            carrier=carrier,
            tracking_number=tracking_number,
            estimated_delivery=estimated_delivery,
            customer_safe_message=order.get("customer_safe_message"),
            items=items,
            requires_human_handoff=requires_handoff,
            action_supported=False
        )

        return {
            "found": True,
            "order": safe_result.model_dump(),
            "snapshot_at": self.snapshot_at,
            "notice": "Lookup only. No order changes, cancellations, or refunds can be executed."
        }