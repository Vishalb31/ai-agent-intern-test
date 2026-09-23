import pytest
from src.tools.order_tool import OrderLookupService


@pytest.fixture
def service():
    return OrderLookupService("data/orders.json")


def test_order_id_normalization(service):
    # Test lowercase and leading/trailing whitespace
    res = service.lookup_order("  ord-1001  ")
    assert res["found"] is True
    assert res["order"]["order_id"] == "ORD-1001"


def test_missing_and_invalid_id(service):
    res_none = service.lookup_order(None)
    assert res_none["found"] is False
    assert "Missing order ID" in res_none["error"]

    res_fake = service.lookup_order("ORD-999999")
    assert res_fake["found"] is False
    assert "not found" in res_fake["error"]


def test_pii_and_internal_fields_never_leaked(service):
    for _, raw_order in service.orders.items():
        res = service.lookup_order(raw_order["order_id"])
        if res["found"]:
            order_dict = res["order"]
            # Ensure sensitive keys are absent from top-level and nested structure
            assert "customer" not in order_dict
            assert "email" not in order_dict
            assert "shipping_address" not in order_dict
            assert "internal" not in order_dict
            assert "risk_score" not in order_dict
            assert "notes" not in order_dict