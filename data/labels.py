"""
Label taxonomy shared by data prep, training, and the agent.

The classifier predicts one of Bitext's 27 fine-grained intents; routing
rules operate on the coarse category each intent maps to. Keeping the
mapping here (dependency-free) lets the agent import it without pulling in
`datasets`.
"""

INTENT_MAP = {
    "get_refund": "refund",
    "track_refund": "refund",
    "check_refund_policy": "refund",
    "cancel_order": "billing",
    "change_order": "billing",
    "payment_issue": "billing",
    "check_invoice": "billing",
    "get_invoice": "billing",
    "check_payment_methods": "billing",
    "check_cancellation_fee": "billing",
    "delivery_options": "shipping",
    "delivery_period": "shipping",
    "track_order": "shipping",
    "place_order": "shipping",
    "change_shipping_address": "shipping",
    "set_up_shipping_address": "shipping",
    "create_account": "account",
    "delete_account": "account",
    "edit_account": "account",
    "recover_password": "account",
    "switch_account": "account",
    "registration_problems": "account",
    "contact_customer_service": "technical",
    "contact_human_agent": "urgent_escalate",
    "complaint": "urgent_escalate",
    "review": "technical",
    "newsletter_subscription": "technical",
}
DEFAULT_CATEGORY = "technical"
CATEGORIES = ["billing", "refund", "technical", "account", "shipping", "urgent_escalate"]


def category_for(intent: str) -> str:
    return INTENT_MAP.get(intent, DEFAULT_CATEGORY)
