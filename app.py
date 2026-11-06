"""
GiftGuard: Gift Card Fraud Detection and Risk Scoring System
Streamlit web application (final-year project prototype)

Files needed in the same folder:
    app.py, giftguard_xgb.pkl, requirements.txt
"""

import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_PATH = "giftguard_xgb.pkl"
DB_PATH = "giftguard.db"
ANALYST_PASSWORD = "giftguard2026"   # demo only: change before your defence

FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
            "newbalanceDest", "balance_drained", "amount_to_balance",
            "error_balance_orig", "is_transfer", "hour"]

# Plain-English names shown to analysts
FEATURE_LABELS = {
    "amount": "Transaction amount",
    "oldbalanceOrg": "Sender's balance before",
    "newbalanceOrig": "Sender's balance after",
    "oldbalanceDest": "Receiver's balance before",
    "newbalanceDest": "Receiver's balance after",
    "balance_drained": "Sender's account emptied",
    "amount_to_balance": "Amount compared with sender's balance",
    "error_balance_orig": "Sender's balances do not add up",
    "is_transfer": "Transaction type (transfer)",
    "hour": "Time of day",
}

# Rule layer thresholds (expert rules that sit on top of the ML model)
RULE_BALANCE_SHARE = 0.80      # moves 80% or more of the balance
RULE_NIGHT_START, RULE_NIGHT_END = 0, 4   # midnight to 4:59am
RULE_NIGHT_AMOUNT = 100_000    # large amount at night (NGN)

TIER_ORDER = {"Allow": 0, "Review": 1, "Block": 2}

st.set_page_config(page_title="GiftGuard", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------------------
# Model and database
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            amount REAL, old_bal REAL, new_bal REAL,
            old_dest REAL, new_dest REAL,
            txn_type TEXT, hour INTEGER,
            probability REAL, ml_score INTEGER, ml_tier TEXT,
            rules_fired TEXT, final_tier TEXT, reasons TEXT,
            status TEXT, analyst_note TEXT, reviewed_at TEXT
        )""")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------
def build_features(amount, old_bal, new_bal, old_dest, new_dest, txn_type, hour):
    row = {
        "amount": amount,
        "oldbalanceOrg": old_bal,
        "newbalanceOrig": new_bal,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "balance_drained": int(old_bal > 0 and new_bal == 0),
        "amount_to_balance": amount / (old_bal + 1),
        "error_balance_orig": new_bal + amount - old_bal,
        "is_transfer": int(txn_type == "Transfer"),
        "hour": hour,
    }
    return pd.DataFrame([row])[FEATURES]


def ml_tier(score):
    if score >= 8:
        return "Block"
    if score >= 5:
        return "Review"
    return "Allow"


def apply_rules(amount, old_bal, hour):
    """Expert rules: any rule that fires raises the tier to at least Review."""
    fired = []
    if old_bal > 0 and amount / old_bal >= RULE_BALANCE_SHARE:
        fired.append(f"Moves {RULE_BALANCE_SHARE:.0%} or more of the sender's balance")
    if RULE_NIGHT_START <= hour <= RULE_NIGHT_END and amount >= RULE_NIGHT_AMOUNT:
        fired.append(f"Large amount (₦{RULE_NIGHT_AMOUNT:,.0f}+) between midnight and 5am")
    return fired


def explain(model, X, top_n=3):
    """SHAP values computed by XGBoost itself (pred_contribs=True)."""
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    pairs = sorted(zip(FEATURES, contribs), key=lambda t: abs(t[1]), reverse=True)[:top_n]
    return [(FEATURE_LABELS[f], float(v)) for f, v in pairs]


def score_transaction(model, amount, old_bal, new_bal, old_dest, new_dest, txn_type, hour):
    X = build_features(amount, old_bal, new_bal, old_dest, new_dest, txn_type, hour)
    prob = float(model.predict_proba(X)[0, 1])
    score = round(prob * 10)
    tier_ml = ml_tier(score)
    rules = apply_rules(amount, old_bal, hour)
    final = tier_ml
    if rules and TIER_ORDER[final] < TIER_ORDER["Review"]:
        final = "Review"
    reasons = explain(model, X)
    return prob, score, tier_ml, rules, final, reasons


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
PRESETS = {
    "Blank": (0.0, 0.0, 0.0, 0.0, 0.0, "Transfer", 12),
    "Account emptied at 2am": (250000.0, 250000.0, 0.0, 0.0, 0.0, "Transfer", 2),
    "Normal small cash-out": (5000.0, 120000.0, 115000.0, 30000.0, 35000.0, "Cash-out", 14),
    "90% of balance at 11pm": (90000.0, 100000.0, 10000.0, 0.0, 90000.0, "Transfer", 23),
}

TIER_STYLE = {"Allow": ("✅", "success"), "Review": ("⚠️", "warning"), "Block": ("⛔", "error")}


def page_score(model, conn):
    st.title("Score a gift card transaction")
    st.caption("Enter the transaction details. GiftGuard returns a 0–10 risk score, "
               "a decision tier and the main reasons behind it.")

    preset = st.selectbox("Fill with an example (optional)", list(PRESETS.keys()))
    p = PRESETS[preset]

    with st.form("score_form"):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Amount (₦)", min_value=0.0, value=p[0], step=1000.0, key=f"a_{preset}")
            old_bal = st.number_input("Sender's balance before (₦)", min_value=0.0, value=p[1], step=1000.0, key=f"b_{preset}")
            new_bal = st.number_input("Sender's balance after (₦)", min_value=0.0, value=p[2], step=1000.0, key=f"c_{preset}")
            txn_type = st.selectbox("Transaction type", ["Transfer", "Cash-out"],
                                    index=0 if p[5] == "Transfer" else 1, key=f"t_{preset}")
        with c2:
            old_dest = st.number_input("Receiver's balance before (₦)", min_value=0.0, value=p[3], step=1000.0, key=f"d_{preset}")
            new_dest = st.number_input("Receiver's balance after (₦)", min_value=0.0, value=p[4], step=1000.0, key=f"e_{preset}")
            hour = st.slider("Hour of day (0 = midnight)", 0, 23, p[6], key=f"h_{preset}")
        submitted = st.form_submit_button("Score transaction", type="primary")

    if not submitted:
        return
    if amount <= 0:
        st.error("Enter an amount greater than zero.")
        return

    prob, score, tier_ml, rules, final, reasons = score_transaction(
        model, amount, old_bal, new_bal, old_dest, new_dest, txn_type, hour)

    status = "Approved" if final == "Allow" else "Pending review"
    conn.execute("""INSERT INTO transactions
        (created_at, amount, old_bal, new_bal, old_dest, new_dest, txn_type, hour,
         probability, ml_score, ml_tier, rules_fired, final_tier, reasons, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), amount, old_bal, new_bal,
         old_dest, new_dest, txn_type, hour, prob, score, tier_ml,
         "; ".join(rules), final, "; ".join(f"{n} ({v:+.2f})" for n, v in reasons), status))
    conn.commit()

    icon, kind = TIER_STYLE[final]
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", f"{score} / 10")
    m2.metric("Fraud probability", f"{prob:.4%}")
    m3.metric("Decision", f"{icon} {final}")

    getattr(st, kind)({
        "Allow": "Low risk. The transaction can proceed.",
        "Review": "Medium risk. The transaction is held for a fraud analyst to review.",
        "Block": "High risk. The transaction is blocked and sent to a fraud analyst.",
    }[final])

    st.subheader("Why this score")
    st.caption("Top factors from the model (SHAP values). Positive values push towards fraud; "
               "negative values push towards genuine.")
    st.table(pd.DataFrame(
        [{"Factor": n, "Effect": "Towards fraud" if v > 0 else "Towards genuine", "SHAP value": round(v, 3)}
         for n, v in reasons]))

    st.subheader("Rule checks")
    if rules:
        for r in rules:
            st.warning(f"Rule fired: {r}")
        if tier_ml == "Allow":
            st.info(f"The model alone said **Allow** (score {score}); the rules raised it to **Review**.")
    else:
        st.write("No expert rules fired.")

    if final != "Allow":
        st.subheader("Customer warning")
        st.error(
            "**Stop and check before you continue.** Scammers often ask people to pay with gift cards. "
            "If someone you have not met in person, or someone claiming to be your boss, a relative, "
            "a bank or a government office, asked you to buy gift cards or share codes, do not proceed. "
            "No genuine organisation asks for payment in gift cards.")


def require_login():
    if st.session_state.get("analyst"):
        return True
    st.title("Analyst login")
    pw = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        if pw == ANALYST_PASSWORD:
            st.session_state["analyst"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def page_review(conn):
    if not require_login():
        return
    st.title("Review queue")
    df = pd.read_sql_query(
        "SELECT * FROM transactions WHERE status = 'Pending review' ORDER BY id DESC", conn)
    if df.empty:
        st.success("No transactions waiting for review.")
        return
    st.caption(f"{len(df)} transaction(s) waiting for a decision.")
    for _, r in df.iterrows():
        icon, _ = TIER_STYLE[r["final_tier"]]
        with st.expander(f"{icon} #{r['id']} · ₦{r['amount']:,.0f} · score {r['ml_score']}/10 · "
                         f"{r['final_tier']} · {r['created_at']}"):
            st.write(f"**Type:** {r['txn_type']} · **Hour:** {r['hour']}:00")
            st.write(f"**Sender balance:** ₦{r['old_bal']:,.0f} → ₦{r['new_bal']:,.0f} · "
                     f"**Receiver balance:** ₦{r['old_dest']:,.0f} → ₦{r['new_dest']:,.0f}")
            st.write(f"**Model:** probability {r['probability']:.4%}, tier {r['ml_tier']}")
            st.write(f"**Top reasons:** {r['reasons']}")
            st.write(f"**Rules fired:** {r['rules_fired'] or 'None'}")
            note = st.text_input("Analyst note", key=f"note_{r['id']}")
            c1, c2 = st.columns(2)
            for col, label, new_status in ((c1, "Confirm fraud", "Confirmed fraud"),
                                           (c2, "Clear as genuine", "Cleared as genuine")):
                if col.button(label, key=f"{new_status}_{r['id']}"):
                    conn.execute("UPDATE transactions SET status=?, analyst_note=?, reviewed_at=? WHERE id=?",
                                 (new_status, note, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(r["id"])))
                    conn.commit()
                    st.rerun()


def page_dashboard(conn):
    if not require_login():
        return
    st.title("Fraud dashboard")
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY id DESC", conn)
    if df.empty:
        st.info("No transactions scored yet. Score some on the first page.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions scored", len(df))
    c2.metric("Blocked", int((df["final_tier"] == "Block").sum()))
    c3.metric("Pending review", int((df["status"] == "Pending review").sum()))
    c4.metric("Confirmed fraud", int((df["status"] == "Confirmed fraud").sum()))

    st.subheader("Decisions by tier")
    counts = df["final_tier"].value_counts().reindex(["Allow", "Review", "Block"], fill_value=0)
    st.bar_chart(counts)

    st.subheader("All transactions")
    show = df[["id", "created_at", "amount", "txn_type", "ml_score", "final_tier",
               "rules_fired", "status", "analyst_note"]].rename(columns={
        "id": "ID", "created_at": "Time", "amount": "Amount (₦)", "txn_type": "Type",
        "ml_score": "Risk score", "final_tier": "Tier", "rules_fired": "Rules fired",
        "status": "Status", "analyst_note": "Analyst note"})
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download as CSV", show.to_csv(index=False), "giftguard_transactions.csv", "text/csv")


def page_about():
    st.title("About the model")
    st.markdown("""
GiftGuard scores each transaction with an **XGBoost** classifier trained on the PaySim dataset
(TRANSFER and CASH_OUT transactions; all fraud cases plus 300,000 genuine ones; 80/20 stratified split).

**Test-set performance (fraud class)**

| Model | Precision | Recall | F1-score | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.4061 | 0.9702 | 0.5726 | 0.8993 |
| Random Forest | 0.9945 | 0.9988 | 0.9967 | 0.9999 |
| **XGBoost (used here)** | **0.9970** | **0.9988** | **0.9979** | 0.9991 |

**How a decision is made**

1. The ten model features are calculated from the transaction.
2. XGBoost returns a fraud probability; risk score = round(probability × 10).
3. Tier from the score: 0–4 Allow, 5–7 Review, 8–10 Block.
4. Expert rules can raise an Allow to Review (they never lower a tier).
5. SHAP values show the top three factors behind the score.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model = load_model()
    conn = get_db()

    st.sidebar.title("🛡️ GiftGuard")
    st.sidebar.caption("Gift card fraud detection and risk scoring")
    page = st.sidebar.radio("Go to", ["Score a transaction", "Review queue", "Dashboard", "About the model"])
    if st.session_state.get("analyst") and st.sidebar.button("Log out"):
        st.session_state["analyst"] = False
        st.rerun()

    if page == "Score a transaction":
        page_score(model, conn)
    elif page == "Review queue":
        page_review(conn)
    elif page == "Dashboard":
        page_dashboard(conn)
    else:
        page_about()


main()
