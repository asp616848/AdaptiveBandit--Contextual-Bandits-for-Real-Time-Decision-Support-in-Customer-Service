from __future__ import annotations

import csv
import json
import random
from pathlib import Path

root = Path(__file__).resolve().parents[2]
rag_dir = root / "Simulation_4" / "rag"
subflow_csv = root / "Simulation_4" / "artifacts" / "phase 1" / "extract3_subflow_stats.csv"

subflows: list[str] = []
with subflow_csv.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sf = row["subflow"].strip()
        if sf and sf not in subflows:
            subflows.append(sf)

mapping: dict[str, dict] = {
    "recover_username": {
        "lumo_label": "account_login_help",
        "display_name": "Account Login Help",
        "query_expansion": "login account access forgotten email workspace URL sign in",
        "primary_policy_sections": ["4.3 Password Policies", "4.1 Two-Factor Authentication"],
        "variants": [
            "forgotten email address used to sign up",
            "wrong workspace URL, cannot find workspace",
            "email address changed since account creation",
        ],
    },
    "recover_password": {
        "lumo_label": "password_reset",
        "display_name": "Password Reset",
        "query_expansion": "password reset email not received magic link SSO login credentials",
        "primary_policy_sections": ["4.3 Password Policies", "4.2 Single Sign-On"],
        "variants": [
            "password reset email not arriving in inbox",
            "SSO org password reset goes through IdP",
            "magic link expired before customer clicked it",
        ],
    },
    "reset_2fa": {
        "lumo_label": "two_factor_auth_locked",
        "display_name": "2FA Locked Out",
        "query_expansion": "two factor authentication locked out authenticator app SMS phone lost new device",
        "primary_policy_sections": ["4.1 Two-Factor Authentication"],
        "variants": [
            "lost phone with authenticator app",
            "new phone and authenticator app did not transfer",
            "SMS 2FA codes not being received",
        ],
    },
    "manage_dispute_bill": {
        "lumo_label": "billing_dispute",
        "display_name": "Billing Dispute",
        "query_expansion": "billing dispute overcharged wrong amount refund credit incorrect invoice",
        "primary_policy_sections": ["1.1 Standard Refund Window", "1.2 Courtesy Refunds", "1.3 Duplicate Charges"],
        "variants": [
            "still charged Business+ after downgrade",
            "charged for seats that were removed",
            "duplicate charge in same billing period",
        ],
    },
    "manage_cancel": {
        "lumo_label": "cancel_subscription",
        "display_name": "Cancel Subscription",
        "query_expansion": "cancel subscription plan end account close workspace stop paying",
        "primary_policy_sections": ["2.4 Cancellations", "5.1 Data Retention"],
        "variants": [
            "too expensive want to cancel",
            "switching tools and need cancellation",
            "want export before cancelling",
        ],
    },
    "slow_speed": {
        "lumo_label": "performance_issue",
        "display_name": "Performance Issue",
        "query_expansion": "slow loading performance lag timeout not responding crashing speed",
        "primary_policy_sections": ["11. Known Issues & Platform Limitations"],
        "variants": ["desktop app extremely slow", "huddle audio lagging", "search timing out"],
    },
    "refund_initiate": {
        "lumo_label": "refund_request",
        "display_name": "Refund Request",
        "query_expansion": "refund money back charge cancel payment return first month",
        "primary_policy_sections": ["1.1 Standard Refund Window", "1.2 Courtesy Refunds"],
        "variants": [
            "within 30 days wants full refund",
            "accidental workspace creation",
            "outside window refund appeal",
        ],
    },
    "refund_status": {
        "lumo_label": "refund_status_check",
        "display_name": "Refund Status Check",
        "query_expansion": "refund status pending waiting where is my refund processing",
        "primary_policy_sections": ["1.1 Standard Refund Window"],
        "variants": ["refund not visible on statement", "bank no record yet", "amount mismatch concern"],
    },
    "refund_update": {
        "lumo_label": "refund_follow_up",
        "display_name": "Refund Follow-Up",
        "query_expansion": "refund overdue not received follow up escalate bank",
        "primary_policy_sections": ["1.1 Standard Refund Window", "1.6 Refund Method"],
        "variants": ["it has been over 10 days", "bank says no incoming refund", "partial refund only"],
    },
}

for sf in subflows:
    if sf in mapping:
        continue
    if sf.startswith("manage_"):
        label = sf.replace("manage_", "account_")
        mapping[sf] = {
            "lumo_label": label,
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": sf.replace("_", " ") + " billing account plan support",
            "primary_policy_sections": ["2. Plans & Pricing", "3. Billing Operations"],
            "variants": [f"{sf} issue variant A", f"{sf} issue variant B"],
        }
    elif sf.startswith("status_") or sf == "status":
        mapping[sf] = {
            "lumo_label": "order_status_question",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": sf.replace("_", " ") + " status tracking timeline details",
            "primary_policy_sections": ["8. Order and Ticket Status"],
            "variants": [f"{sf} clarification request", f"{sf} discrepancy request"],
        }
    elif sf.startswith("return_"):
        mapping[sf] = {
            "lumo_label": "return_request",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": sf.replace("_", " ") + " return replacement policy",
            "primary_policy_sections": ["9. Returns & Replacement"],
            "variants": [f"{sf} return scenario one", f"{sf} return scenario two"],
        }
    elif sf.startswith("promo_code_"):
        mapping[sf] = {
            "lumo_label": "promo_code_issue",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": "promo code discount coupon not working expired",
            "primary_policy_sections": ["10.2 Promotional Codes"],
            "variants": [f"{sf} invalid code case", f"{sf} eligibility mismatch case"],
        }
    elif sf.startswith("membership_"):
        mapping[sf] = {
            "lumo_label": "membership_access_issue",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": "membership access entitlement confirmation account benefits",
            "primary_policy_sections": ["2. Plans & Pricing", "4. User Access and Authentication"],
            "variants": [f"{sf} membership activation delay", f"{sf} membership entitlement mismatch"],
        }
    elif sf.startswith("policy_"):
        mapping[sf] = {
            "lumo_label": "policy_clarification",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": "policy clarification rules eligibility compliance",
            "primary_policy_sections": ["11. Known Issues & Platform Limitations", "10. Promotional & Discount Policies"],
            "variants": [f"{sf} policy interpretation request", f"{sf} policy exception request"],
        }
    elif sf.startswith("pricing_") or sf in {"cost", "bad_price_competitor", "bad_price_yesterday"}:
        mapping[sf] = {
            "lumo_label": "pricing_question",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": "pricing increase competitor comparison seats annual monthly",
            "primary_policy_sections": ["2. Plans & Pricing", "10. Promotional & Discount Policies"],
            "variants": [f"{sf} pricing concern case", f"{sf} renewal pricing question"],
        }
    elif sf.startswith("timing_") or sf.startswith("mistimed_"):
        mapping[sf] = {
            "lumo_label": "billing_timing_issue",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": "billing timing charge date renewal proration unexpected timing",
            "primary_policy_sections": ["3. Billing Operations", "1.3 Duplicate Charges"],
            "variants": [f"{sf} charge timing mismatch", f"{sf} renewal timing mismatch"],
        }
    elif any(sf.startswith(x) for x in ["boots_", "jacket_", "jeans_", "shirt_"]):
        mapping[sf] = {
            "lumo_label": "product_info_request",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": "product information details compatibility recommendation usage",
            "primary_policy_sections": ["3. Feature Reference", "11. Known Issues & Platform Limitations"],
            "variants": [f"{sf} detail inquiry", f"{sf} support guidance request"],
        }
    elif sf == "search_results":
        mapping[sf] = {
            "lumo_label": "search_not_working",
            "display_name": "Search Not Working",
            "query_expansion": "search results missing messages not found search broken",
            "primary_policy_sections": ["3.8 Search"],
            "variants": ["cannot find old messages", "search filters not behaving as expected"],
        }
    elif sf in {"out_of_stock_general", "out_of_stock_one_item"}:
        mapping[sf] = {
            "lumo_label": "feature_availability_issue",
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": "feature unavailable plan requirement integration availability",
            "primary_policy_sections": ["2. Plans & Pricing", "6. Integrations"],
            "variants": ["feature not in current plan", "integration access restricted"],
        }
    elif sf == "credit_card":
        mapping[sf] = {
            "lumo_label": "payment_failed",
            "display_name": "Payment Failed",
            "query_expansion": "credit card declined failed payment billing",
            "primary_policy_sections": ["3.1 Accepted Payment Methods", "3.2 Failed Payments"],
            "variants": ["card declined by issuer", "card expired needs update"],
        }
    elif sf == "missing":
        mapping[sf] = {
            "lumo_label": "missing_invoice",
            "display_name": "Missing Invoice",
            "query_expansion": "invoice missing receipt billing history tax document",
            "primary_policy_sections": ["3.3 Invoice & Receipt Requests"],
            "variants": ["invoice not received by finance", "need historical invoices for audit"],
        }
    else:
        mapping[sf] = {
            "lumo_label": sf,
            "display_name": sf.replace("_", " ").title(),
            "query_expansion": sf.replace("_", " ") + " support help",
            "primary_policy_sections": ["General Support"],
            "variants": [f"{sf} case variant A", f"{sf} case variant B"],
        }

for sf in subflows:
    assert sf in mapping

mapping_path = rag_dir / "subflow_mapping.json"
mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

workspace_names = [
    "NovaBridge",
    "Thalia Studio",
    "Crestline Analytics",
    "Fieldstone Capital",
    "Vertex Systems",
    "Bloom Retail",
    "Ironclad Legal",
    "Solaris Ventures",
    "Greenpath",
    "Meridian Ops",
    "Coastal Media",
    "Pinnacle HR",
    "Hartwell Group",
    "Cascade Studio",
    "Northgate Software",
    "Brightline Consulting",
]
people = ["Jordan Malik", "Priya Nair", "Marcus Webb", "Dana Okonkwo", "Sam Hendricks", "Cleo Vasquez"]


def mk_variant(label: str, idx: int) -> dict:
    ws = workspace_names[(abs(hash(label)) + idx) % len(workspace_names)]
    slug = ws.lower().replace(" ", "")
    return {
        "variant_id": f"{label}_v{idx}",
        "description": f"{label.replace('_', ' ')} scenario {idx}",
        "slots": {
            "workspace_name": ws,
            "workspace_url": f"lumo.com/{slug}",
            "customer_name": people[(abs(hash(label + str(idx))) % len(people))],
            "plan": random.choice(["Pro", "Business+", "Free"]),
            "member_role": random.choice(["Member", "Admin", "Owner"]),
            "email": f"user{idx}@{slug}.com",
            "billing_email": f"billing@{slug}.com",
            "charge_amount": random.choice(["$65.25", "$130.50", "$217.50", "$375.00", "$450.00"]),
            "error_message": random.choice([
                "Verification code required",
                "Invalid email or password",
                "Payment failed",
                "Request timed out",
            ]),
            "issue_detail": f"Customer reports {label.replace('_', ' ')} and needs support resolution.",
        },
    }

labels = sorted({entry["lumo_label"] for entry in mapping.values()})
templates = {label: {"variants": [mk_variant(label, 1), mk_variant(label, 2)]} for label in labels}

if "account_login_help" in templates:
    templates["account_login_help"]["variants"][0]["slots"].update(
        {
            "issue_detail": "forgot which email address was used to sign up, joined workspace 8 months ago",
            "error_message": "No account found with that email address",
        }
    )
if "password_reset" in templates:
    templates["password_reset"]["variants"][0]["slots"].update(
        {
            "issue_detail": "requested password reset 3 times, no email received",
            "error_message": "Reset email not received",
        }
    )
if "billing_dispute" in templates:
    templates["billing_dispute"]["variants"][0]["slots"].update(
        {
            "charge_amount": "$450.00",
            "expected_amount": "$217.50",
            "invoice_number": "INV-20260301-4821",
        }
    )

templates_path = rag_dir / "scenario_templates.json"
templates_path.write_text(json.dumps(templates, indent=2), encoding="utf-8")

print(f"wrote {mapping_path}")
print(f"wrote {templates_path}")
print(f"subflows mapped: {len(mapping)}")
print(f"lumo labels templated: {len(templates)}")
