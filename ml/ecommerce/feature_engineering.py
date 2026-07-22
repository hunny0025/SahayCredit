"""
SahayCredit — E-Commerce Feature Engineering
=============================================
Computes customer-level features from raw Olist order/payment/review tables.
Used both during training (from Olist CSVs) and at inference time (from
real AA transaction data mapped to e-commerce order format).

All features are computed at the CUSTOMER level: one row per customer.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_customer_features(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    reference_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Build a customer-level feature matrix from raw Olist tables.

    Parameters
    ----------
    orders : DataFrame with columns [order_id, customer_id, order_status,
             order_purchase_timestamp, order_delivered_customer_date,
             order_estimated_delivery_date]
    items : DataFrame with [order_id, product_id, price, freight_value]
    payments : DataFrame with [order_id, payment_type, payment_installments,
               payment_value]
    reviews : DataFrame with [order_id, review_score]
    products : DataFrame with [product_id, product_category_name]
    reference_date : Timestamp for recency calculation (defaults to max date)

    Returns
    -------
    DataFrame with one row per customer_id and all engineered features.
    """
    logger.info("Building customer-level features ...")

    # ── Ensure datetime columns ──────────────────────────────────────────
    for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    if reference_date is None:
        reference_date = orders["order_purchase_timestamp"].max()

    # ── Only delivered orders for most features ──────────────────────────
    delivered = orders[orders["order_status"] == "delivered"].copy()

    # ── Order-level value (sum of item prices per order) ─────────────────
    order_values = (
        items.groupby("order_id")["price"]
        .sum()
        .reset_index()
        .rename(columns={"price": "order_value"})
    )

    delivered = delivered.merge(order_values, on="order_id", how="left")
    delivered["order_value"] = delivered["order_value"].fillna(0)

    # ── RFM Features ─────────────────────────────────────────────────────
    rfm = delivered.groupby("customer_id").agg(
        last_order=("order_purchase_timestamp", "max"),
        first_order=("order_purchase_timestamp", "min"),
        frequency=("order_id", "nunique"),
        monetary_total=("order_value", "sum"),
        monetary_avg=("order_value", "mean"),
    ).reset_index()

    rfm["recency_days"] = (reference_date - rfm["last_order"]).dt.days.clip(lower=0)
    rfm["order_span_days"] = (rfm["last_order"] - rfm["first_order"]).dt.days.clip(lower=0)

    # ── Order Value Statistics ───────────────────────────────────────────
    ov_stats = delivered.groupby("customer_id")["order_value"].agg(
        order_value_std="std",
        order_value_max="max",
        order_value_min="min",
    ).reset_index()

    ov_stats["order_value_std"] = ov_stats["order_value_std"].fillna(0)
    ov_stats["order_value_range"] = ov_stats["order_value_max"] - ov_stats["order_value_min"]

    features = rfm.merge(ov_stats, on="customer_id", how="left")

    # CV = std / mean (only meaningful for multi-order customers)
    features["order_value_cv"] = np.where(
        features["monetary_avg"] > 0,
        features["order_value_std"] / features["monetary_avg"],
        0
    )

    # ── Category Diversity ───────────────────────────────────────────────
    items_cats = items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id", how="left"
    )
    items_cats = items_cats.merge(
        delivered[["order_id", "customer_id"]],
        on="order_id", how="inner"
    )

    cat_counts = (
        items_cats.groupby(["customer_id", "product_category_name"])
        .size()
        .reset_index(name="cat_count")
    )

    n_categories = cat_counts.groupby("customer_id").size().reset_index(name="n_categories")

    # Herfindahl concentration index
    cat_totals = cat_counts.groupby("customer_id")["cat_count"].sum().reset_index(name="total_items")
    cat_shares = cat_counts.merge(cat_totals, on="customer_id")
    cat_shares["share_sq"] = (cat_shares["cat_count"] / cat_shares["total_items"]) ** 2
    herfindahl = cat_shares.groupby("customer_id")["share_sq"].sum().reset_index(name="category_concentration")

    features = features.merge(n_categories, on="customer_id", how="left")
    features = features.merge(herfindahl, on="customer_id", how="left")
    features["n_categories"] = features["n_categories"].fillna(1)
    features["category_concentration"] = features["category_concentration"].fillna(1.0)

    # ── Payment Features ─────────────────────────────────────────────────
    # Only for delivered orders
    delivered_payments = payments.merge(
        delivered[["order_id", "customer_id"]],
        on="order_id", how="inner"
    )

    pay_agg = delivered_payments.groupby("customer_id").agg(
        total_payments=("order_id", "count"),
        n_boleto=("payment_type", lambda x: (x == "boleto").sum()),
        n_credit_card=("payment_type", lambda x: (x == "credit_card").sum()),
        avg_installments=("payment_installments", "mean"),
        max_installments=("payment_installments", "max"),
    ).reset_index()

    pay_agg["boleto_ratio"] = pay_agg["n_boleto"] / pay_agg["total_payments"].clip(lower=1)
    pay_agg["credit_card_ratio"] = pay_agg["n_credit_card"] / pay_agg["total_payments"].clip(lower=1)

    features = features.merge(
        pay_agg[["customer_id", "boleto_ratio", "credit_card_ratio",
                 "avg_installments", "max_installments"]],
        on="customer_id", how="left"
    )

    # ── Review Features ──────────────────────────────────────────────────
    delivered_reviews = reviews.merge(
        delivered[["order_id", "customer_id"]],
        on="order_id", how="inner"
    )

    review_agg = delivered_reviews.groupby("customer_id").agg(
        n_reviews=("review_score", "count"),
        avg_review_score=("review_score", "mean"),
        n_low_reviews=("review_score", lambda x: (x <= 2).sum()),
    ).reset_index()

    # Merge with frequency to compute review_rate
    review_agg = review_agg.merge(
        features[["customer_id", "frequency"]],
        on="customer_id", how="left"
    )
    review_agg["review_rate"] = review_agg["n_reviews"] / review_agg["frequency"].clip(lower=1)
    review_agg["low_review_ratio"] = review_agg["n_low_reviews"] / review_agg["n_reviews"].clip(lower=1)

    features = features.merge(
        review_agg[["customer_id", "avg_review_score", "low_review_ratio", "review_rate"]],
        on="customer_id", how="left"
    )

    # ── Delivery Features ────────────────────────────────────────────────
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"] -
        delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered["is_late"] = (
        delivered["order_delivered_customer_date"] >
        delivered["order_estimated_delivery_date"]
    ).fillna(False)

    delivery_agg = delivered.groupby("customer_id").agg(
        avg_delivery_days=("delivery_days", "mean"),
        delivery_time_std=("delivery_days", "std"),
        n_late=("is_late", "sum"),
        n_delivered=("order_id", "nunique"),
    ).reset_index()

    delivery_agg["delivery_time_std"] = delivery_agg["delivery_time_std"].fillna(0)
    delivery_agg["late_delivery_ratio"] = delivery_agg["n_late"] / delivery_agg["n_delivered"].clip(lower=1)

    features = features.merge(
        delivery_agg[["customer_id", "avg_delivery_days", "delivery_time_std", "late_delivery_ratio"]],
        on="customer_id", how="left"
    )

    # ── Temporal Features ────────────────────────────────────────────────
    # Average days between consecutive orders
    order_dates = (
        delivered.sort_values("order_purchase_timestamp")
        .groupby("customer_id")["order_purchase_timestamp"]
        .apply(lambda x: x.diff().dt.days.mean() if len(x) > 1 else np.nan)
        .reset_index(name="avg_days_between_orders")
    )

    features = features.merge(order_dates, on="customer_id", how="left")

    # Weekend order ratio
    delivered["is_weekend"] = delivered["order_purchase_timestamp"].dt.dayofweek >= 5

    weekend_agg = delivered.groupby("customer_id").agg(
        n_weekend=("is_weekend", "sum"),
        n_total=("order_id", "nunique"),
    ).reset_index()
    weekend_agg["weekend_order_ratio"] = weekend_agg["n_weekend"] / weekend_agg["n_total"].clip(lower=1)

    features = features.merge(
        weekend_agg[["customer_id", "weekend_order_ratio"]],
        on="customer_id", how="left"
    )

    # ── Clean up and select final columns ────────────────────────────────
    from .model_config import ALL_FEATURES

    # Fill remaining NaNs with sensible defaults
    for col in ALL_FEATURES:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Select only the features we need + customer_id
    available = [c for c in ALL_FEATURES if c in features.columns]
    missing = [c for c in ALL_FEATURES if c not in features.columns]
    if missing:
        logger.warning(f"Missing features (filled with 0): {missing}")
        for col in missing:
            features[col] = 0

    result = features[["customer_id"] + ALL_FEATURES].copy()

    logger.info(f"Feature matrix: {result.shape[0]} customers × {len(ALL_FEATURES)} features")
    return result


def build_target_label(orders: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the proxy target label: is_risky_customer.

    A customer is labeled as "risky" (1) if they exhibit ANY of:
    - Had an order cancelled or marked unavailable
    - Gave a review score of 1 (extreme dissatisfaction) on 50%+ of reviewed orders
    - Had late delivery AND low review on the same order (dispute proxy)

    IMPORTANT: This is a PROXY label engineered from e-commerce behavioral
    signals. It is NOT a real credit default outcome. This is documented
    transparently and must be labeled as such in all outputs.

    Parameters
    ----------
    orders : DataFrame with [order_id, customer_id, order_status,
             order_delivered_customer_date, order_estimated_delivery_date]
    reviews : DataFrame with [order_id, review_score]

    Returns
    -------
    DataFrame with [customer_id, is_risky_customer, risk_reason]
    """
    logger.info("Building proxy target label (is_risky_customer) ...")

    # Ensure datetime columns
    for col in ["order_delivered_customer_date", "order_estimated_delivery_date"]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # ── Signal 1: Order cancellation or unavailability ───────────────────
    cancel_customers = (
        orders[orders["order_status"].isin(["canceled", "unavailable"])]
        .groupby("customer_id")["order_id"]
        .nunique()
        .reset_index(name="n_cancelled")
    )

    # ── Signal 2: Extreme dissatisfaction (review score = 1 on 50%+ orders)
    order_reviews = orders[["order_id", "customer_id"]].merge(
        reviews[["order_id", "review_score"]], on="order_id", how="inner"
    )

    review_stats = order_reviews.groupby("customer_id").agg(
        n_reviews=("review_score", "count"),
        n_score_1=("review_score", lambda x: (x == 1).sum()),
    ).reset_index()
    review_stats["extreme_dissatisfaction"] = (
        (review_stats["n_score_1"] / review_stats["n_reviews"].clip(lower=1)) >= 0.5
    ) & (review_stats["n_reviews"] >= 1)

    # ── Signal 3: Dispute proxy (late delivery + low review on same order)
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered["is_late"] = (
        delivered["order_delivered_customer_date"] >
        delivered["order_estimated_delivery_date"]
    ).fillna(False)

    dispute_orders = delivered[delivered["is_late"]].merge(
        reviews[reviews["review_score"] <= 2][["order_id"]],
        on="order_id", how="inner"
    )

    dispute_customers = (
        dispute_orders.groupby("customer_id")["order_id"]
        .nunique()
        .reset_index(name="n_disputes")
    )

    # ── Combine signals ─────────────────────────────────────────────────
    all_customers = orders["customer_id"].unique()
    target = pd.DataFrame({"customer_id": all_customers})

    target = target.merge(cancel_customers, on="customer_id", how="left")
    target = target.merge(
        review_stats[["customer_id", "extreme_dissatisfaction"]],
        on="customer_id", how="left"
    )
    target = target.merge(dispute_customers, on="customer_id", how="left")

    target["n_cancelled"] = target["n_cancelled"].fillna(0)
    target["extreme_dissatisfaction"] = target["extreme_dissatisfaction"].fillna(False)
    target["n_disputes"] = target["n_disputes"].fillna(0)

    # A customer is risky if ANY signal fires
    target["is_risky_customer"] = (
        (target["n_cancelled"] > 0) |
        (target["extreme_dissatisfaction"]) |
        (target["n_disputes"] > 0)
    ).astype(int)

    # Build reason string for auditability
    reasons = []
    for _, row in target.iterrows():
        r = []
        if row["n_cancelled"] > 0:
            r.append(f"cancelled:{int(row['n_cancelled'])}")
        if row["extreme_dissatisfaction"]:
            r.append("extreme_dissatisfaction")
        if row["n_disputes"] > 0:
            r.append(f"disputes:{int(row['n_disputes'])}")
        reasons.append("|".join(r) if r else "none")
    target["risk_reason"] = reasons

    n_risky = target["is_risky_customer"].sum()
    n_total = len(target)
    logger.info(f"Target label: {n_risky}/{n_total} risky ({n_risky/n_total*100:.1f}%)")

    return target[["customer_id", "is_risky_customer", "risk_reason"]]


def compute_inference_features(order_history: list) -> dict:
    """
    Compute the same feature set from a borrower's order history at
    inference time. This mirrors the training-time feature engineering
    but operates on a single borrower's data (list of order dicts).

    Parameters
    ----------
    order_history : list of dicts, each with keys:
        date, amount, category, reviewScore, wasLate, paymentType, installments

    Returns
    -------
    dict of feature_name -> value (matching ALL_FEATURES order)
    """
    from .model_config import ALL_FEATURES

    if not order_history or len(order_history) == 0:
        return {f: 0.0 for f in ALL_FEATURES}

    df = pd.DataFrame(order_history)

    # Parse dates
    df["date"] = pd.to_datetime(df.get("date", pd.NaT), errors="coerce")
    ref_date = df["date"].max()

    n_orders = len(df)
    amounts = df["amount"].fillna(0).astype(float)

    features = {}

    # RFM
    features["recency_days"] = max(0, (ref_date - df["date"].max()).days) if pd.notna(ref_date) else 0
    features["frequency"] = n_orders
    features["monetary_total"] = float(amounts.sum())
    features["monetary_avg"] = float(amounts.mean()) if n_orders > 0 else 0

    # Order value stats
    features["order_value_std"] = float(amounts.std()) if n_orders > 1 else 0
    features["order_value_cv"] = (
        features["order_value_std"] / features["monetary_avg"]
        if features["monetary_avg"] > 0 else 0
    )
    features["order_value_max"] = float(amounts.max())
    features["order_value_range"] = float(amounts.max() - amounts.min())

    # Category diversity
    categories = df["category"].dropna().unique() if "category" in df.columns else []
    features["n_categories"] = max(1, len(categories))
    if len(categories) > 0:
        cat_counts = df["category"].value_counts()
        shares = (cat_counts / cat_counts.sum()) ** 2
        features["category_concentration"] = float(shares.sum())
    else:
        features["category_concentration"] = 1.0

    # Payment features
    if "paymentType" in df.columns:
        pay_types = df["paymentType"].fillna("unknown")
        features["boleto_ratio"] = float((pay_types == "boleto").mean())
        features["credit_card_ratio"] = float((pay_types == "credit_card").mean())
    else:
        features["boleto_ratio"] = 0.0
        features["credit_card_ratio"] = 0.0

    if "installments" in df.columns:
        inst = df["installments"].fillna(1).astype(float)
        features["avg_installments"] = float(inst.mean())
        features["max_installments"] = float(inst.max())
    else:
        features["avg_installments"] = 1.0
        features["max_installments"] = 1.0

    # Review features
    if "reviewScore" in df.columns:
        scores = df["reviewScore"].dropna().astype(float)
        if len(scores) > 0:
            features["avg_review_score"] = float(scores.mean())
            features["low_review_ratio"] = float((scores <= 2).mean())
            features["review_rate"] = len(scores) / n_orders
        else:
            features["avg_review_score"] = 4.0
            features["low_review_ratio"] = 0.0
            features["review_rate"] = 0.0
    else:
        features["avg_review_score"] = 4.0
        features["low_review_ratio"] = 0.0
        features["review_rate"] = 0.0

    # Delivery features
    if "wasLate" in df.columns:
        late = df["wasLate"].fillna(False).astype(bool)
        features["late_delivery_ratio"] = float(late.mean())
    else:
        features["late_delivery_ratio"] = 0.0
    features["avg_delivery_days"] = 10.0  # Default when not available
    features["delivery_time_std"] = 3.0

    # Temporal features
    if n_orders > 1 and pd.notna(ref_date):
        sorted_dates = df["date"].dropna().sort_values()
        diffs = sorted_dates.diff().dt.days.dropna()
        features["avg_days_between_orders"] = float(diffs.mean()) if len(diffs) > 0 else 0
        features["order_span_days"] = float((sorted_dates.max() - sorted_dates.min()).days)
    else:
        features["avg_days_between_orders"] = 0.0
        features["order_span_days"] = 0.0

    if "date" in df.columns:
        weekend = df["date"].dt.dayofweek >= 5
        features["weekend_order_ratio"] = float(weekend.mean())
    else:
        features["weekend_order_ratio"] = 0.0

    # Ensure all features are present in correct order
    return {f: float(features.get(f, 0.0)) for f in ALL_FEATURES}
