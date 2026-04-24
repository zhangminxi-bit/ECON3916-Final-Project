# -*- coding: utf-8 -*-
# ECON 3916: ML Prediction Project — Final Project
# Predicting Stock Breakout Success Using Price and Volume Features

# ============================================================
# Part 0: Setup
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score
)
import yfinance as yf
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

print('Setup complete.')

# ============================================================
# 2.1 Load dataset
# ============================================================

TICKERS  = ['AAPL', 'MSFT', 'NVDA', 'XLF', 'XLV',
            'AMZN', 'GOOGL', 'META', 'JPM', 'UNH']
LOOKBACK = 20
COOLDOWN = 5
HORIZONS = [3, 5, 10, 20]

print('Downloading 5 years of daily data from Yahoo Finance...')
raw = yf.download(TICKERS, period='5y', interval='1d', auto_adjust=True)

close_wide  = raw['Close']
volume_wide = raw['Volume']
high_wide   = raw['High']
low_wide    = raw['Low']

frames = []
for t in TICKERS:
    tmp = pd.DataFrame({
        'Date':   close_wide.index,
        'Ticker': t,
        'Close':  close_wide[t].values,
        'High':   high_wide[t].values,
        'Low':    low_wide[t].values,
        'Volume': volume_wide[t].values
    })
    frames.append(tmp)

df = pd.concat(frames).dropna().sort_values(['Ticker','Date']).reset_index(drop=True)
df['Date'] = pd.to_datetime(df['Date'])

def compute_signals(group):
    group = group.copy().sort_values('Date').reset_index(drop=True)
    closes = group['Close'].values
    highs  = group['High'].values
    lows   = group['Low'].values
    vols   = group['Volume'].values
    n      = len(group)

    group['high_20d'] = (group['Close'].shift(1)
                         .rolling(LOOKBACK, min_periods=LOOKBACK).max())
    group['raw_breakout'] = (group['Close'] > group['high_20d']).astype(int)
    group['breakout'] = 0
    last_b = -COOLDOWN - 1
    for i in range(n):
        if group.loc[i, 'raw_breakout'] == 1 and (i - last_b) > COOLDOWN:
            group.loc[i, 'breakout'] = 1
            last_b = i

    for h in HORIZONS:
        fwd = np.full(n, np.nan)
        for i in range(n):
            if i + h < n:
                fwd[i] = (closes[i+h] - closes[i]) / closes[i]
        group[f'ret_{h}d'] = fwd

    failed = np.full(n, np.nan)
    for i in range(n):
        if group.loc[i, 'breakout'] == 1:
            window = closes[i+1 : i+6]
            if len(window) == 5:
                failed[i] = int(any(window < closes[i]))
    group['failed_5d'] = failed

    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows  - np.roll(closes, 1))
        )
    )
    tr[0] = highs[0] - lows[0]
    group['atr'] = pd.Series(tr).rolling(14).mean().values

    return group

df = (df.groupby('Ticker', group_keys=False)
        .apply(compute_signals)
        .reset_index(drop=True))

counts = df[df['breakout'] == 1].groupby('Ticker').size()
print('Breakout counts per ticker:')
print(counts.to_string())
print(f'\nTotal raw breakout events: {counts.sum()}')

def engineer_features(group):
    group = group.copy().sort_values('Date').reset_index(drop=True)
    closes = group['Close'].values
    vols   = group['Volume'].values
    n      = len(group)

    group['breakout_strength'] = (
        (group['Close'] - group['high_20d']) / group['high_20d']
    )
    group['vol_avg_20d']  = (group['Volume'].shift(1)
                              .rolling(20, min_periods=10).mean())
    group['volume_ratio'] = group['Volume'] / group['vol_avg_20d']

    ret_5  = np.full(n, np.nan)
    ret_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 5:
            ret_5[i]  = (closes[i] - closes[i-5])  / closes[i-5]
        if i >= 20:
            ret_20[i] = (closes[i] - closes[i-20]) / closes[i-20]
    group['ret_prior_5d']  = ret_5
    group['ret_prior_20d'] = ret_20

    group['daily_ret']      = group['Close'].pct_change()
    group['volatility_20d'] = (group['daily_ret']
                                .rolling(20, min_periods=10).std())
    group['atr_ratio'] = group['atr'] / group['Close']

    return group

df = (df.groupby('Ticker', group_keys=False)
        .apply(engineer_features)
        .reset_index(drop=True))

FEATURE_COLS = ['breakout_strength', 'volume_ratio', 'ret_prior_5d',
                'ret_prior_20d', 'volatility_20d', 'atr_ratio']

event_table = (
    df[df['breakout'] == 1]
    [['Date','Ticker','Close','high_20d','Volume','ret_5d','failed_5d']
     + FEATURE_COLS]
    .dropna()
    .copy()
    .reset_index(drop=True)
)

event_table['success'] = (
    event_table['ret_5d'] > 0
).astype(int)

ticker_dummies = pd.get_dummies(event_table['Ticker'], prefix='ticker',
                                 drop_first=True)
event_table = pd.concat([event_table, ticker_dummies], axis=1)

# ============================================================
# 2.2 Describe your data
# ============================================================

print(f'Final event table: {len(event_table)} observations')
print(f'\nFeature columns: {FEATURE_COLS}')
print(f'\nTarget distribution:')
print(event_table['success'].value_counts())
print(f'\nSuccess rate: {event_table["success"].mean():.1%}')
event_table[FEATURE_COLS + ['success']].describe().round(4)
event_table.info()

# ============================================================
# 2.3 Missing data heatmap (Ch 1: MCAR/MAR/MNAR)
# ============================================================

missing_pct = df.isnull().mean().sort_values(ascending=False)
print('Missing data (%) by column:')
print(missing_pct[missing_pct > 0].round(4))

plt.figure(figsize=(14, 5))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap — Full Dataset (before event filtering)')
plt.tight_layout()
plt.show()

print('\nMissing values in event table (after .dropna()):')
print(event_table[FEATURE_COLS + ['success']].isnull().sum())

# ============================================================
# 2.4 Distribution of key features (Ch 3)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

feat_labels = {
    'breakout_strength': 'Breakout Strength',
    'volume_ratio':      'Volume Ratio (vs 20d avg)',
    'ret_prior_5d':      'Prior 5-day Return',
    'ret_prior_20d':     'Prior 20-day Return',
    'volatility_20d':    '20-day Volatility',
    'atr_ratio':         'ATR Ratio'
}

success_ev = event_table[event_table['success'] == 1]
failure_ev = event_table[event_table['success'] == 0]

for ax, feat in zip(axes, FEATURE_COLS):
    ax.hist(success_ev[feat].dropna(), bins=20, alpha=0.6,
            color='#16a34a', label='Success', density=True)
    ax.hist(failure_ev[feat].dropna(), bins=20, alpha=0.6,
            color='#dc2626', label='Failure', density=True)
    ax.set_title(feat_labels[feat], fontsize=11, fontweight='bold')
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)

fig.suptitle('Feature Distributions: Successful vs. Failed Breakouts',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 2.5 Outlier detection (Ch 4: Tukey Fences / IQR)
# ============================================================

def tukey_fences(series, k=1.5):
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - k * IQR, Q3 + k * IQR

print('Outlier counts per feature (Tukey fences, k=1.5):')
print(f'{"Feature":<22} {"Outliers":>10} {"% of N":>10}')
print('-' * 45)
for col in FEATURE_COLS:
    lower, upper = tukey_fences(event_table[col].dropna())
    n_out = ((event_table[col] < lower) | (event_table[col] > upper)).sum()
    print(f'{col:<22} {n_out:>10} {n_out/len(event_table)*100:>9.1f}%')

# ============================================================
# 2.6 Correlation heatmap (Ch 3)
# ============================================================

corr_matrix = event_table[FEATURE_COLS].corr()

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', mask=mask,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 10})
plt.title('Correlation Matrix — Breakout Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print('Correlation of each feature with target (success):')
target_corr = (event_table[FEATURE_COLS + ['success']]
               .corr()['success']
               .drop('success')
               .sort_values(key=abs, ascending=False)
               .round(4))
print(target_corr.to_string())

# ============================================================
# 3.1 Train/test split (Ch 6)
# ============================================================

ticker_dummy_cols = [c for c in event_table.columns if c.startswith('ticker_')]
ALL_FEATURES      = FEATURE_COLS + ticker_dummy_cols

X = event_table[ALL_FEATURES]
y = event_table['success']

event_sorted = event_table.sort_values('Date').reset_index(drop=True)
split_idx    = int(len(event_sorted) * 0.70)

train_df = event_sorted.iloc[:split_idx]
test_df  = event_sorted.iloc[split_idx:]

X_train = train_df[ALL_FEATURES].values
X_test  = test_df[ALL_FEATURES].values
y_train = train_df['success'].values
y_test  = test_df['success'].values

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f'Train: {X_train.shape[0]} samples  '
      f'({train_df["Date"].min().date()} → {train_df["Date"].max().date()})')
print(f'Test:  {X_test.shape[0]} samples   '
      f'({test_df["Date"].min().date()} → {test_df["Date"].max().date()})')
print(f'Features: {X_train.shape[1]}')
print(f'Scaler fit on training data only ✓')

# ============================================================
# 3.2 Model 1 — Baseline: Logistic Regression
# ============================================================

model_1 = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                              class_weight='balanced')
model_1.fit(X_train_sc, y_train)

y_pred_1 = model_1.predict(X_test_sc)

print('Model 1: Logistic Regression')
print(classification_report(y_test, y_pred_1,
                             target_names=['Failure', 'Success']))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_1),
    display_labels=['Failure', 'Success']
).plot(ax=ax, cmap='Blues', values_format=',', colorbar=False)
ax.set_title('Model 1: Logistic Regression — Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.show()

coef_df = pd.DataFrame({
    'Feature':     ALL_FEATURES,
    'Coefficient': model_1.coef_[0].round(4),
    'Odds Ratio':  np.exp(model_1.coef_[0]).round(4)
}).sort_values('Coefficient', ascending=False)
print('\nCoefficients and Odds Ratios:')
print(coef_df.to_string(index=False))

# ============================================================
# 3.3 Model 2 — Random Forest Classifier
# ============================================================

model_2 = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                  class_weight='balanced')
model_2.fit(X_train_sc, y_train)

y_pred_2 = model_2.predict(X_test_sc)

print('Model 2: Random Forest')
print(classification_report(y_test, y_pred_2,
                             target_names=['Failure', 'Success']))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_2),
    display_labels=['Failure', 'Success']
).plot(ax=ax, cmap='Greens', values_format=',', colorbar=False)
ax.set_title('Model 2: Random Forest — Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 3.4 Cross-validation (Ch 15)
# ============================================================

scoring = 'f1'

cv_1 = cross_val_score(model_1, X_train_sc, y_train, cv=5, scoring=scoring)
cv_2 = cross_val_score(model_2, X_train_sc, y_train, cv=5, scoring=scoring)

print(f'Model 1 CV {scoring}: {cv_1.mean():.4f} +/- {cv_1.std():.4f}')
print(f'Model 2 CV {scoring}: {cv_2.mean():.4f} +/- {cv_2.std():.4f}')

comparison = pd.DataFrame({
    'Model': ['Model 1 (Logistic Regression)', 'Model 2 (Random Forest)'],
    f'CV {scoring} (mean)': [round(cv_1.mean(), 4), round(cv_2.mean(), 4)],
    f'CV {scoring} (std)':  [round(cv_1.std(),  4), round(cv_2.std(),  4)],
    'Test Accuracy': [
        round(accuracy_score(y_test, y_pred_1), 4),
        round(accuracy_score(y_test, y_pred_2), 4)
    ],
    'Test Precision': [
        round(precision_score(y_test, y_pred_1, zero_division=0), 4),
        round(precision_score(y_test, y_pred_2, zero_division=0), 4)
    ],
    'Test Recall': [
        round(recall_score(y_test, y_pred_1, zero_division=0), 4),
        round(recall_score(y_test, y_pred_2, zero_division=0), 4)
    ],
    'Test F1': [
        round(f1_score(y_test, y_pred_1, zero_division=0), 4),
        round(f1_score(y_test, y_pred_2, zero_division=0), 4)
    ]
})
print(comparison.to_string(index=False))

# ============================================================
# 4.1 Feature importance (Ch 19)
# ============================================================

importances = pd.Series(
    model_2.feature_importances_, index=ALL_FEATURES
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#16a34a' if f in FEATURE_COLS else '#9ca3af'
          for f in importances.index]
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.text(
    0.98, 0.02,
    'Predictive importance only.\nDoes not imply causal effect.',
    transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
    style='italic', color='#c0392b',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', edgecolor='#e74c3c')
)
plt.tight_layout()
plt.show()

# ============================================================
# 4.2 Key visualization — ROC curve
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

for model, name, color in [
    (model_1, 'Logistic Regression', '#2563eb'),
    (model_2, 'Random Forest',       '#16a34a')
]:
    proba = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{name}  (AUC = {auc:.3f})')

ax.plot([0,1],[0,1], color='gray', linestyle='--',
        linewidth=1.2, label='Random classifier (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Breakout Success Prediction\n'
             'Logistic Regression vs. Random Forest',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6.1 Save model and scaler for Streamlit app
# ============================================================

joblib.dump(model_1, 'model.pkl')
joblib.dump(scaler,  'scaler.pkl')
print('model.pkl and scaler.pkl saved ✓')
print('Use these in app.py with joblib.load()')
