"""
SahayCredit — E-Commerce ML Model Configuration
================================================
Central configuration for the e-commerce scoring model training pipeline.
All hyperparameters, feature lists, and constants are defined here for
reproducibility and auditability.
"""

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2       # 20% held-out test set (never seen during training)
VAL_SIZE = 0.25       # 25% of remaining = 20% of total for validation
N_CV_FOLDS = 5        # Stratified K-Fold cross-validation
OPTUNA_TRIALS = 50    # Hyperparameter search budget per model

# ── Target Label Definition ─────────────────────────────────────────────────
# Since Olist has NO credit/default label, we engineer a proxy target:
# is_risky_customer = 1 if the customer exhibits ANY of:
#   - Order cancelled or unavailable
#   - Payment via boleto that was never confirmed (proxy for payment failure)
#   - High dispute ratio (review_score <= 2 AND late delivery)
#   - Multiple order issues
#
# This is a PROXY label — transparently documented, not a real credit outcome.
TARGET_COL = "is_risky_customer"

# ── Feature Groups ──────────────────────────────────────────────────────────
# These are computed at the CUSTOMER level from raw Olist tables.

RFM_FEATURES = [
    "recency_days",           # Days since last order
    "frequency",              # Total number of orders
    "monetary_total",         # Total spend (sum of all order values)
    "monetary_avg",           # Average order value
]

ORDER_VALUE_FEATURES = [
    "order_value_std",        # Std dev of order values
    "order_value_cv",         # Coefficient of variation
    "order_value_max",        # Maximum single order value
    "order_value_range",      # Max - Min order value
]

CATEGORY_FEATURES = [
    "n_categories",           # Number of distinct product categories
    "category_concentration", # Herfindahl index of category distribution
]

PAYMENT_FEATURES = [
    "boleto_ratio",           # Fraction of orders paid via boleto
    "credit_card_ratio",      # Fraction paid via credit card
    "avg_installments",       # Average number of installments used
    "max_installments",       # Maximum installments used in any order
]

REVIEW_FEATURES = [
    "avg_review_score",       # Mean review score given (1-5)
    "low_review_ratio",       # Fraction of reviews with score 1-2
    "review_rate",            # Fraction of orders that were reviewed
]

DELIVERY_FEATURES = [
    "late_delivery_ratio",    # Fraction of orders delivered late
    "avg_delivery_days",      # Average delivery time in days
    "delivery_time_std",      # Std dev of delivery time
]

TEMPORAL_FEATURES = [
    "avg_days_between_orders",  # Average inter-order interval
    "weekend_order_ratio",      # Fraction of orders placed on weekends
    "order_span_days",          # Days between first and last order
]

ALL_FEATURES = (
    RFM_FEATURES +
    ORDER_VALUE_FEATURES +
    CATEGORY_FEATURES +
    PAYMENT_FEATURES +
    REVIEW_FEATURES +
    DELIVERY_FEATURES +
    TEMPORAL_FEATURES
)

# Total: 25 features

# ── Olist Dataset Files ─────────────────────────────────────────────────────
OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

# ── Hyperparameter Search Spaces ────────────────────────────────────────────
# Used by Optuna for each model type

XGBOOST_SPACE = {
    "n_estimators": (100, 1000),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "gamma": (0.0, 1.0),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.5, 5.0),
}

LIGHTGBM_SPACE = {
    "n_estimators": (100, 1000),
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_samples": (5, 50),
    "num_leaves": (15, 127),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.5, 5.0),
}

CATBOOST_SPACE = {
    "iterations": (100, 1000),
    "depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "l2_leaf_reg": (1.0, 10.0),
    "bagging_temperature": (0.0, 1.0),
    "random_strength": (0.0, 1.0),
}

RF_SPACE = {
    "n_estimators": (100, 500),
    "max_depth": (5, 30),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),
    "max_features": ["sqrt", "log2", None],
}
