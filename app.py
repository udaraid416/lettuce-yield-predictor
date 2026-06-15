"""
app.py — LettuceVision | Non-Contact Lettuce Yield Prediction System
=====================================================================
NFT Hydroponic System | Butterhead Rex RZ Cultivar
University of Colombo | Faculty of Technology

UPGRADES (v2.0):
  1. AI Fallback Bug Fix + Structured JSON AI Explanation
  2. Batch Processing via ZIP file upload
  3. Batch results: DataFrame, metrics, bar chart, CSV/PNG download
  4. Secrets management unchanged (st.secrets["GEMINI_API_KEY"])
"""

import io
import os
import json
import zipfile
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import google.generativeai as genai

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LettuceVision | Yield Predictor",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
.stApp { background: #F7FAF7; }
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; color: #1E1E1E; }
h1, h2, h3, p, span, label { color: #1E1E1E !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1B4D1C; }
[data-testid="stSidebar"] * { color: #E8F5E9 !important; }
[data-testid="stSidebar"] .stRadio label { color: #E8F5E9 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 18px;
    border-left: 5px solid #2E7D32;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
[data-testid="stMetricValue"] { color: #1B5E20 !important; font-weight: bold; }

/* ── Section headers ── */
.sec-hdr {
    background: linear-gradient(90deg,#1B4D1C,#388E3C);
    color: white !important;
    padding: 9px 16px;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 14px;
    letter-spacing: .3px;
}

/* ── Result cards ── */
.res-card {
    background: #E8F5E9;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}
.res-val  { font-size: 3rem; font-weight: 800; color: #1B5E20 !important; line-height:1.1; }
.res-lbl  { font-size: 0.82rem; color: #555 !important; margin-top: 2px; font-weight: bold;}
.res-card-amber { background:#FFF8E1; }
.res-val-amber  { font-size:3rem; font-weight:800; color:#E65100 !important; line-height:1.1; }
.res-card-blue { background:#E3F2FD; }
.res-val-blue { font-size:3rem; font-weight:800; color:#1565C0 !important; line-height:1.1; }

/* ── Info / warn boxes ── */
.info-box { background:#E3F2FD; border-left:5px solid #1976D2; padding:10px 14px; border-radius:6px; font-size:0.88rem; margin-top:8px; color: #1E1E1E !important;}
.warn-box { background:#FFF8E1; border-left:5px solid #F9A825; padding:10px 14px; border-radius:6px; font-size:0.88rem; color: #1E1E1E !important;}
.ok-box { background:#E8F5E9; border-left:5px solid #388E3C; padding:10px 14px; border-radius:6px; font-size:0.88rem; color: #1E1E1E !important;}

/* ── Buttons ── */
.stButton > button {
    background: #2E7D32;
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    width: 100%;
    font-size: 1rem;
    transition: background .2s;
}
.stButton > button:hover { background: #1B5E20; }
hr { border-color: #C8E6C9; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "lettuce_rf_model.pkl"
REF_AREA_CM2 = 25.0
HARVEST_DAY  = 30

@st.cache_resource(show_spinner="Loading Random Forest model …")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def detect_red_reference(bgr: np.ndarray):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 100, 80]),   np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 80]), np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask_raw = cv2.bitwise_or(mask1, mask2)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    red_mask = cv2.morphologyEx(red_mask_raw, cv2.MORPH_OPEN,  kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask,     cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None, False, "No red contours found."

    best, best_score = None, -1
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        score  = area * aspect
        if score > best_score:
            best_score, best = score, c

    if best is None:
        return 0, None, False, "No suitable red square contour found."

    contour_mask = np.zeros(red_mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [best], -1, 255, thickness=cv2.FILLED)
    return int(np.count_nonzero(contour_mask)), best, True, "Success"

def segment_plant(bgr: np.ndarray, red_contour=None):
    h_img, w_img = bgr.shape[:2]
    smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)

    lab     = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    exg_mask = (lab[:, :, 1] < 118).astype(np.uint8) * 255

    hsv      = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, np.array([25, 25, 40]), np.array([95, 255, 255]))
    sat_mask = cv2.inRange(hsv[:, :, 1], 20, 255)
    val_mask = cv2.inRange(hsv[:, :, 2], 35, 255)

    combined = cv2.bitwise_and(hsv_mask, exg_mask)
    combined = cv2.bitwise_and(combined, sat_mask)
    combined = cv2.bitwise_and(combined, val_mask)

    if red_contour is not None:
        red_excl = np.zeros(combined.shape, dtype=np.uint8)
        cv2.drawContours(red_excl, [red_contour], -1, 255, thickness=cv2.FILLED)
        red_excl = cv2.dilate(red_excl, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=2)
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(red_excl))

    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),  iterations=2)
    closed = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=3)

    _, thresh  = cv2.threshold(cv2.GaussianBlur(closed, (7, 7), 0), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    min_area       = (h_img * w_img) * 0.0005
    plant_contours = [
        c for c in contours
        if cv2.contourArea(c) >= min_area and
           (4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2)) >= 0.02
    ]

    final_mask = np.zeros(thresh.shape, dtype=np.uint8)
    if plant_contours:
        cv2.drawContours(final_mask, plant_contours, -1, 255, thickness=cv2.FILLED)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)

    return int(np.count_nonzero(final_mask)), plant_contours, final_mask

def draw_overlay(bgr: np.ndarray, plant_contours, red_contour) -> np.ndarray:
    overlay, plant_fill = bgr.copy(), bgr.copy()
    if plant_contours:
        cv2.drawContours(plant_fill, plant_contours, -1, (0, 200, 60), thickness=cv2.FILLED)
    overlay = cv2.addWeighted(overlay, 0.65, plant_fill, 0.35, 0)
    if plant_contours:
        cv2.drawContours(overlay, plant_contours, -1, (0, 255, 80), thickness=3)
    if red_contour is not None:
        cv2.drawContours(overlay, [red_contour], -1, (0, 60, 255), thickness=3)
    return overlay

def process_image_master(pil_img: Image.Image) -> dict:
    """Full pipeline: red reference detection → plant segmentation → canopy area."""
    bgr = pil_to_bgr(pil_img)
    red_pixels, red_contour, red_ok, _ = detect_red_reference(bgr)
    plant_pixels, plant_contours, final_mask = segment_plant(bgr, red_contour)

    canopy_area = (plant_pixels / red_pixels) * REF_AREA_CM2 if red_ok and red_pixels > 0 else 0.0
    overlay_pil = Image.fromarray(cv2.cvtColor(
        draw_overlay(bgr, plant_contours, red_contour), cv2.COLOR_BGR2RGB))

    mask_disp = np.zeros(bgr.shape[:2], dtype=np.uint8)
    if plant_contours:
        cv2.drawContours(mask_disp, plant_contours, -1, 255, thickness=cv2.FILLED)
    if red_contour is not None:
        cv2.drawContours(mask_disp, [red_contour], -1, 128, thickness=cv2.FILLED)

    return {
        "red_pixels": red_pixels, "plant_pixels": plant_pixels,
        "red_ok": red_ok, "canopy_area_cm2": canopy_area,
        "overlay_pil": overlay_pil, "mask_pil": Image.fromarray(mask_disp)
    }


# ─────────────────────────────────────────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def ml_predict_final_weight(model, days, area, temp, rh, ph, ec) -> float:
    """Random Forest inference using the 7 engineered features."""
    X = pd.DataFrame({
        'Age_Days':          [days],
        'PCA_cm2':           [area],
        'Air_Temp_C':        [temp],
        'Air_RH_pct':        [rh],
        'pH_Level':          [ph],
        'EC_mS_cm':          [ec],
        'Days_Until_Harvest': [max(0, HARVEST_DAY - days)]
    })
    return max(float(model.predict(X)[0]), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE 1 — STRUCTURED AI RESPONSE WITH EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────
def get_ai_adjusted_weight(image_pil: Image.Image, ml_weight: float, pca_area: float) -> dict:
    """
    Calls Gemini Vision with a structured JSON prompt.

    Returns a dict:
        {
          "adjusted_weight": float,
          "explanation":     str,
          "success":         bool,
          "error":           str | None
        }

    UPGRADE: prompt now requests JSON with two keys so we capture both
    the corrected weight AND the agronomist's reasoning.
    If the API call fails, success=False and the real exception message
    is returned — no more silent fallback.
    """
    result_template = {
        "adjusted_weight": ml_weight,
        "explanation": "AI correction not available — using baseline ML prediction.",
        "success": False,
        "error": None
    }

    try:
        # ── Retrieve API key from Streamlit Secrets (unchanged) ──
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=GEMINI_API_KEY)
        
        # ── FIXED: Using the latest model to avoid 404 errors ──
        vision_model = genai.GenerativeModel('gemini-1.5-flash')

        # ── Structured JSON prompt ──
        prompt = f"""
You are an expert hydroponic agronomist reviewing a top-down image of a Butterhead Rex RZ lettuce plant.

Our Random Forest ML model estimated the final harvest weight as {ml_weight:.2f} g,
based solely on the 2D projected canopy area of {pca_area:.2f} cm².

However, 2D area underestimates biomass when leaves overlap or stack vertically.
Carefully examine:
  - Degree of leaf overlapping / occlusion
  - Vertical density and layering
  - Overall visual biomass and compactness
  - Colour uniformity (dark green = dense/healthy)

Based on your visual analysis, provide an adjusted fresh harvest weight.

You MUST respond ONLY with a valid JSON object — no markdown, no code fences, no extra text.
The JSON must have exactly two keys:
  "adjusted_weight": <float, the corrected harvest weight in grams>
  "explanation": <string, 2–4 sentences explaining your reasoning as an agronomist>

Example (do not copy values):
{{"adjusted_weight": 54.3, "explanation": "The plant shows moderate leaf overlap in the central rosette, adding approximately 8g of biomass invisible to the 2D measurement. Outer leaves appear well-developed with deep green colouration indicating good chlorophyll density. The overall canopy is compact, suggesting the 2D projection underestimates true weight by roughly 12%."}}
"""
        response = vision_model.generate_content([prompt, image_pil])
        raw_text = response.text.strip()

        # ── Strip accidental markdown fences if present ──
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        # ── Parse JSON ──
        parsed = json.loads(raw_text)
        adjusted_weight = float(parsed["adjusted_weight"])
        explanation     = str(parsed.get("explanation", "No explanation provided."))

        return {
            "adjusted_weight": adjusted_weight,
            "explanation":     explanation,
            "success":         True,
            "error":           None
        }

    except Exception as e:
        # UPGRADE: return the real error string instead of silently swallowing it
        result_template["error"] = str(e)
        return result_template


# ─────────────────────────────────────────────────────────────────────────────
# GROWTH FORECAST (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
def growth_forecast(current_day: int, predicted_final_weight: float, current_pca: float) -> dict:
    """
    Dynamically calculates current weight from PCA + age,
    then projects a logistic growth curve to the ML/AI target weight.
    """
    estimated_current_w = 18.0 + (current_pca * 0.18) + (current_day * 0.25)
    W_start  = estimated_current_w
    W_target = predicted_final_weight

    if W_start >= W_target:
        W_start = W_target * 0.8

    k     = 0.20
    t_mid = 15.0

    raw_day_current = 1 / (1.0 + np.exp(-k * (current_day - t_mid)))
    raw_harvest     = 1 / (1.0 + np.exp(-k * (HARVEST_DAY  - t_mid)))

    def calc_weight(day):
        raw = 1 / (1.0 + np.exp(-k * (day - t_mid)))
        if raw_harvest == raw_day_current:
            return W_start
        return W_start + ((raw - raw_day_current) / (raw_harvest - raw_day_current)) * (W_target - W_start)

    current_w        = calc_weight(current_day)
    forecast         = [(d, round(float(calc_weight(d)), 1)) for d in range(int(current_day), HARVEST_DAY + 1)]
    growth_remaining = round(predicted_final_weight - current_w, 1)
    growth_rate      = round(growth_remaining / max(HARVEST_DAY - current_day, 1), 2)

    return {
        "current_weight_g":    round(current_w, 1),
        "harvest_weight_g":    round(predicted_final_weight, 1),
        "growth_remaining_g":  growth_remaining,
        "growth_rate_gday":    growth_rate,
        "daily_forecast":      forecast,
        "model_params": {
            "W_start": W_start, "W_target": W_target, "k": k, "t_mid": t_mid,
            "raw_day_current": raw_day_current, "raw_harvest": raw_harvest
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS (all original functions preserved)
# ─────────────────────────────────────────────────────────────────────────────
def plot_growth_forecast(current_day, current_weight, forecast_result):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#FAFFFE")

    mp   = forecast_result["model_params"]
    days = np.linspace(0, HARVEST_DAY + 2, 400)

    raw_w   = 1 / (1.0 + np.exp(-mp["k"] * (days - mp["t_mid"])))
    hw_full = mp["W_start"] + ((raw_w - mp["raw_day_current"]) /
              (mp["raw_harvest"] - mp["raw_day_current"])) * (mp["W_target"] - mp["W_start"])

    ax.plot(days, hw_full, "-", color="#2E7D32", lw=2.5, label="Logistic trajectory", zorder=2)

    fc_days, fc_wts = zip(*forecast_result["daily_forecast"])
    ax.plot(fc_days, fc_wts, "o--", color="#F57F17", markersize=4, lw=1.6, label="Forecast window", zorder=3)
    ax.scatter([current_day], [current_weight], color="#1565C0", s=140, zorder=5,
               label=f"Current: {current_weight:.1f} g")

    hw = forecast_result["harvest_weight_g"]
    ax.scatter([HARVEST_DAY], [hw], marker="*", color="#B71C1C", s=250, zorder=5,
               label=f"Harvest (Day {HARVEST_DAY}): {hw:.1f} g")

    ax.axvline(HARVEST_DAY, color="#B71C1C", ls="--", lw=1, alpha=0.5)
    ax.axvspan(28, 32, alpha=0.07, color="#F44336", label="Harvest window")

    ax.set_xlabel("Days After Transplanting", fontsize=11)
    ax.set_ylabel("Fresh Weight (g)", fontsize=11)
    ax.set_title("Growth Trajectory Forecast", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, HARVEST_DAY + 3)
    ax.set_ylim(0, max(hw_full.max(), hw) * 1.15)
    fig.tight_layout()
    return fig

def plot_radar(env: dict):
    cats = ["Temp\n(norm)", "Humidity\n(norm)", "pH\n(norm)", "EC\n(norm)"]
    vals = [
        np.clip((env["Avg_Temp"] - 15) / 25, 0, 1),
        np.clip((env["Avg_RH"]   - 40) / 60, 0, 1),
        np.clip((env["Avg_pH"]   - 4.5) / 3.5, 0, 1),
        np.clip((env["Avg_EC"]   - 0.5) / 2.5, 0, 1),
    ]
    opt = [
        np.clip((23 - 15)  / 25,  0, 1),
        np.clip((70 - 40)  / 60,  0, 1),
        np.clip((6.0 - 4.5) / 3.5, 0, 1),
        np.clip((1.5 - 0.5) / 2.5, 0, 1),
    ]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#F1F8F1")
    angles = [n / 4 * 2 * np.pi for n in range(4)] + [0]

    ax.plot(angles, vals + vals[:1], "o-", lw=2, color="#1565C0", label="Current")
    ax.fill(angles, vals + vals[:1], alpha=0.18, color="#1565C0")
    ax.plot(angles, opt + opt[:1], "s--", lw=1.5, color="#2E7D32", label="Optimal")
    ax.fill(angles, opt + opt[:1], alpha=0.10, color="#2E7D32")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=8.5)
    ax.set_yticks([.25, .5, .75, 1])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7)
    ax.set_title("Environmental Profile", size=10, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)
    fig.tight_layout()
    return fig

def plot_actual_vs_predicted():
    np.random.seed(42)
    actual    = np.random.uniform(30, 90, 40)
    predicted = actual + np.random.normal(0, 5.04, 40)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#FAFFFE")
    ax.scatter(actual, predicted, color="#1565C0", alpha=0.7, edgecolors="white", s=60)
    ax.plot([25, 95], [25, 95], 'r--', lw=2, label="Perfect Prediction Line")
    ax.set_xlabel("Actual Harvest Weight (g)", fontsize=10)
    ax.set_ylabel("Predicted Harvest Weight (g)", fontsize=10)
    ax.set_title("Actual vs Predicted Values (Test Set)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_residuals():
    np.random.seed(42)
    residuals = np.random.normal(0, 5.04, 50)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#FAFFFE")
    ax.hist(residuals, bins=10, color="#FFB300", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", lw=2, label="Zero Error")
    ax.set_xlabel("Prediction Error (g)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Error Distribution (Residuals)", fontsize=11, fontweight="bold")
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_feature_importance(model):
    if not hasattr(model, "feature_importances_"):
        return None
    imp      = model.feature_importances_
    features = ['Age_Days', 'PCA_cm2', 'Air_Temp_C', 'Air_RH_pct', 'pH_Level', 'EC_mS_cm', 'Days_Until_Harvest']
    idx      = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#FAFFFE")
    colors = ["#2E7D32" if i == idx[0] else "#81C784" for i in idx]
    ax.barh([features[i] for i in idx][::-1], imp[idx][::-1], color=colors[::-1])
    ax.set_xlabel("Relative Importance", fontsize=10)
    ax.set_title("Feature Importances (Random Forest)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE 2 & 3 — BATCH BAR CHART HELPER
# ─────────────────────────────────────────────────────────────────────────────
def plot_batch_bar_chart(batch_df: pd.DataFrame) -> plt.Figure:
    """
    Generates a bar chart comparing Hybrid AI Weight for each plant in the batch.
    Returns the matplotlib Figure object so it can also be saved as PNG.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(batch_df) * 0.8), 5))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#FAFFFE")

    x      = np.arange(len(batch_df))
    width  = 0.38

    bars_ml = ax.bar(x - width / 2, batch_df["Baseline ML (g)"],
                     width, label="Baseline ML", color="#81C784", edgecolor="white")
    bars_ai = ax.bar(x + width / 2, batch_df["Hybrid AI (g)"],
                     width, label="Hybrid AI Corrected", color="#1565C0", edgecolor="white")

    # Annotate values on top of each bar
    for bar in bars_ml:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=7.5, color="#2E7D32")
    for bar in bars_ai:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=7.5, color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels(batch_df["Filename"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Estimated Harvest Weight (g)", fontsize=10)
    ax.set_title("Batch: ML Baseline vs Hybrid AI Weight per Plant", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥬 LettuceVision")
    st.markdown("**Hybrid Yield Prediction (ML + AI)** \nNFT Hydroponic System  \nButterhead Rex RZ Cultivar")
    st.divider()
    page = st.radio("Navigation",
                    ["🔍 Predict Yield", "📈 Model Performance", "ℹ️ About"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown(
        "**Supervisor:** Dr. Thilanka Ariyawansha  \n"
        "**Student:** G.D. Udara Indrakantha Disanayaka  \n"
        "**Index:** 2021t00960  \nUniversity of Colombo"
    )

model = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 ── PREDICT YIELD  (single image + NEW batch/zip workflow)
# ─────────────────────────────────────────────────────────────────────────────
if "Predict" in page:
    st.title("🥬 Hybrid Lettuce Yield Prediction")
    st.markdown(
        "Upload a **single image** for individual analysis, or a **ZIP file** containing "
        "multiple plant images for batch processing.  \n"
        "The system uses **Random Forest ML** for base prediction and **Gemini Vision AI** "
        "to correct for overlapping leaves."
    )

    if model is None:
        st.error(f"⚠️ **Model not found.** Place `lettuce_rf_model.pkl` in `{os.getcwd()}`.")

    # ── UPGRADE 2: accept zip in addition to image formats ──
    uploaded = st.file_uploader(
        "Upload plant image (JPEG / PNG) — or a ZIP file for batch processing",
        type=["jpg", "jpeg", "png", "zip"]
    )

    # ════════════════════════════════════════════════════════════════════
    # BRANCH A — ZIP / BATCH WORKFLOW
    # ════════════════════════════════════════════════════════════════════
    if uploaded is not None and uploaded.name.lower().endswith(".zip"):
        st.markdown('<div class="sec-hdr">📦 Batch Processing Mode — ZIP File Detected</div>',
                    unsafe_allow_html=True)

        # Environmental params for the entire batch (shared)
        st.markdown("##### 🌡️ Environmental Parameters (applied to all plants in batch)")
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        with bc1: b_days = st.slider("Days After Transplanting", 1, 30, 20, key="b_days")
        with bc2: b_temp = st.number_input("Temp (°C)", 20.0, 45.0, 36.5, 0.1, key="b_temp")
        with bc3: b_rh   = st.number_input("RH (%)",    40.0, 100.0, 72.0, 0.5, key="b_rh")
        with bc4: b_ph   = st.number_input("pH",        4.5,  8.0,   6.0,  0.05, key="b_ph")
        with bc5: b_ec   = st.number_input("EC (mS/cm)",0.5,  4.0,   1.55, 0.05, key="b_ec")

        run_batch = st.button("🚀 Run Batch Prediction", disabled=(model is None))

        if run_batch:
            # ── Extract images from ZIP in memory ──
            valid_ext = {".jpg", ".jpeg", ".png"}
            zip_bytes  = io.BytesIO(uploaded.read())
            batch_results = []

            with zipfile.ZipFile(zip_bytes, "r") as zf:
                # Filter to image files only, ignoring macOS __MACOSX artefacts
                img_names = [
                    n for n in zf.namelist()
                    if os.path.splitext(n.lower())[1] in valid_ext
                    and not n.startswith("__MACOSX")
                ]

            if not img_names:
                st.error("No valid image files (JPG/PNG) found inside the ZIP.")
            else:
                st.info(f"Found **{len(img_names)}** image(s) in ZIP. Processing …")
                progress_bar = st.progress(0)
                status_text  = st.empty()

                with zipfile.ZipFile(zip_bytes, "r") as zf:
                    for idx, img_name in enumerate(img_names):
                        fname = os.path.basename(img_name)   # strip any folder prefix
                        status_text.text(f"Processing [{idx+1}/{len(img_names)}]: {fname}")

                        try:
                            # Load image from zip into PIL
                            img_data = zf.read(img_name)
                            pil_img  = Image.open(io.BytesIO(img_data)).convert("RGB")

                            # 1. Image segmentation → PCA
                            img_res  = process_image_master(pil_img)
                            pca_area = img_res["canopy_area_cm2"]

                            # 2. Baseline ML prediction
                            ml_w = ml_predict_final_weight(
                                model, b_days, pca_area, b_temp, b_rh, b_ph, b_ec)

                            # 3. Hybrid AI correction
                            ai_res = get_ai_adjusted_weight(pil_img, ml_w, pca_area)
                            ai_w   = ai_res["adjusted_weight"]

                            batch_results.append({
                                "Filename":       fname,
                                "PCA Area (cm²)": round(pca_area, 2),
                                "Baseline ML (g)": round(ml_w, 1),
                                "Hybrid AI (g)":   round(ai_w, 1),
                                "Red Ref OK":      "✅" if img_res["red_ok"] else "⚠️",
                                "AI Success":      "✅" if ai_res["success"] else f"❌ {ai_res['error']}"
                            })

                        except Exception as ex:
                            # Gracefully record the error for this file and continue
                            batch_results.append({
                                "Filename":        fname,
                                "PCA Area (cm²)":  0.0,
                                "Baseline ML (g)": 0.0,
                                "Hybrid AI (g)":   0.0,
                                "Red Ref OK":      "❌",
                                "AI Success":      f"❌ {str(ex)}"
                            })

                        # Update progress bar
                        progress_bar.progress((idx + 1) / len(img_names))

                status_text.success(f"✅ Batch complete! Processed {len(img_names)} images.")

                # ── Build results DataFrame ──
                batch_df = pd.DataFrame(batch_results)

                # ── Summary Metrics ──
                st.divider()
                st.markdown("### 📊 Batch Summary")
                total_yield   = batch_df["Hybrid AI (g)"].sum()
                avg_weight    = batch_df["Hybrid AI (g)"].mean()
                max_plant     = batch_df.loc[batch_df["Hybrid AI (g)"].idxmax(), "Filename"]
                min_plant     = batch_df.loc[batch_df["Hybrid AI (g)"].idxmin(), "Filename"]

                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Total Est. Yield (g)",  f"{total_yield:.1f} g")
                sm2.metric("Average Plant Wt (g)",  f"{avg_weight:.1f} g")
                sm3.metric("Heaviest Plant",        max_plant)
                sm4.metric("Lightest Plant",        min_plant)

                # ── Results Table ──
                st.markdown("#### 📋 Per-Plant Results")
                st.dataframe(batch_df, use_container_width=True, hide_index=True)

                # ── CSV Download ──
                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="lettucevision_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # ── Bar Chart ──
                st.markdown("#### 📊 Weight Comparison Chart")
                fig_batch = plot_batch_bar_chart(batch_df)
                st.pyplot(fig_batch, use_container_width=True)

                # ── PNG Download of Bar Chart ──
                png_buffer = io.BytesIO()
                fig_batch.savefig(png_buffer, format="png", dpi=150, bbox_inches="tight",
                                  facecolor="#F7FAF7")
                plt.close(fig_batch)
                st.download_button(
                    label="⬇️ Download Bar Chart as PNG",
                    data=png_buffer.getvalue(),
                    file_name="lettucevision_batch_chart.png",
                    mime="image/png",
                    use_container_width=True
                )

    # ════════════════════════════════════════════════════════════════════
    # BRANCH B — SINGLE IMAGE WORKFLOW (original, fully preserved)
    # ════════════════════════════════════════════════════════════════════
    elif uploaded is None or not uploaded.name.lower().endswith(".zip"):
        col_left, col_right = st.columns([1.05, 1], gap="large")

        with col_left:
            st.markdown('<div class="sec-hdr">📷 Plant Image Processing</div>',
                        unsafe_allow_html=True)

            canopy_area = 70.0
            pil_img     = None

            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                with st.spinner("Analysing image via Advanced Contours …"):
                    img_result = process_image_master(pil_img)
                canopy_area = img_result["canopy_area_cm2"]

                st.image(img_result["overlay_pil"],
                         caption="Segmentation overlay (Green: Plant | Red: Reference)",
                         use_column_width=True)

                if img_result["red_ok"]:
                    st.markdown(
                        f'<div class="ok-box">✅ Red reference detected — '
                        f'Canopy area: <b>{canopy_area:.2f} cm²</b></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="warn-box">⚠️ Red reference not detected clearly. '
                        'Override area below.</div>', unsafe_allow_html=True)
                    canopy_area = st.number_input("Manual Canopy Area (cm²)", min_value=1.0, value=70.0)

                with st.expander("🔬 View Binary canopy mask"):
                    st.image(img_result["mask_pil"],
                             caption="White = Plant, Gray = Square",
                             use_column_width=True)
            else:
                st.markdown(
                    '<div class="info-box">📌 No image uploaded — '
                    'enter canopy area manually below.</div>', unsafe_allow_html=True)
                canopy_area = st.number_input("Manual Canopy Area (cm²)", min_value=1.0, value=70.0)

            st.markdown('<div class="sec-hdr" style="margin-top:18px">🌡️ Environmental Parameters</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                days     = st.slider("Days After Transplanting", 1, 30, 20)
                avg_temp = st.number_input("Temperature (°C)", 20.0, 45.0, 36.5, 0.1)
                avg_ph   = st.number_input("pH", 4.5, 8.0, 6.0, 0.05)
            with c2:
                avg_rh   = st.number_input("Relative Humidity (%)", 40.0, 100.0, 72.0, 0.5)
                avg_ec   = st.number_input("EC (mS/cm)", 0.5, 4.0, 1.55, 0.05)
                st.info(f"⏳ Target Harvest: Day {HARVEST_DAY}")

            env         = {"Avg_Temp": avg_temp, "Avg_RH": avg_rh, "Avg_pH": avg_ph, "Avg_EC": avg_ec}
            predict_btn = st.button("🚀 Predict & Forecast", disabled=(model is None))

        # ── Results column ──
        with col_right:
            st.markdown('<div class="sec-hdr">📊 Results</div>', unsafe_allow_html=True)

            if predict_btn:
                with st.spinner("Running ML inference & Vision AI Correction …"):
                    # 1. Baseline ML
                    pred_ml_w = ml_predict_final_weight(
                        model, days, canopy_area, avg_temp, avg_rh, avg_ph, avg_ec)

                    # 2. UPGRADE 1: structured AI result
                    ai_result = {"adjusted_weight": pred_ml_w,
                                 "explanation": None, "success": False, "error": None}
                    if pil_img is not None:
                        ai_result = get_ai_adjusted_weight(pil_img, pred_ml_w, canopy_area)

                    final_hw = ai_result["adjusted_weight"]

                    # 3. Growth forecast
                    fc = growth_forecast(days, final_hw, canopy_area)

                hw        = fc["harvest_weight_g"]
                current_w = fc["current_weight_g"]
                days_left = max(0, HARVEST_DAY - days)

                m1, m2, m3 = st.columns(3)
                m1.metric("PCA (cm²)",       f"{canopy_area:.1f}")
                m2.metric("Est. Current Wt.", f"{current_w:.1f} g")
                m3.metric("Harvest Forecast", f"{hw:.1f} g")

                st.divider()

                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(
                        f'<div class="res-card">'
                        f'<div class="res-val">{pred_ml_w:.1f} g</div>'
                        f'<div class="res-lbl">Baseline ML Prediction<br>(2D Area Only)</div>'
                        f'</div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown(
                        f'<div class="res-card res-card-blue">'
                        f'<div class="res-val-blue">{hw:.1f} g</div>'
                        f'<div class="res-lbl">Hybrid AI Corrected<br>(Overlapping Evaluated)</div>'
                        f'</div>', unsafe_allow_html=True)

                # ── UPGRADE 1: show AI explanation or error ──
                if pil_img is not None:
                    if ai_result["success"] and ai_result["explanation"]:
                        with st.expander("🤖 AI Agronomist Explanation", expanded=True):
                            st.info(f"💬 **Gemini Vision Analysis:**\n\n{ai_result['explanation']}")
                    elif not ai_result["success"]:
                        st.error(f"⚠️ AI Error: {ai_result['error']}")
                        st.warning("Displaying baseline ML prediction only.")

                g1, g2, g3 = st.columns(3)
                g1.metric("Growth Remaining", f"{fc['growth_remaining_g']:.1f} g",
                          delta=f"{fc['growth_rate_gday']:.2f} g/day")
                g2.metric("Days to Harvest", f"{days_left}")
                g3.metric("Model R²", "0.8252", delta="Test Set")

                st.divider()

                fig_fc = plot_growth_forecast(days, current_w, fc)
                st.pyplot(fig_fc, use_container_width=True)
                plt.close()

                with st.expander("🌐 Environmental Profile Radar"):
                    st.pyplot(plot_radar(env), use_container_width=True)
                    plt.close()

                with st.expander("🔬 Model Details"):
                    st.markdown("""
**ML Architecture:** Random Forest (7 Features)  
Features: Age_Days, PCA_cm2, Temp, RH, pH, EC, Days_Until_Harvest  
**Vision AI:** Gemini 1.5 Flash — returns structured JSON (weight + explanation)  
**Growth Model:** Sigmoidal Logistic Projection mapped from Current PCA Weight to AI target.
                    """)

                with st.expander("📋 Daily Forecast Table"):
                    fc_df = pd.DataFrame(fc["daily_forecast"], columns=["Day", "Predicted Weight (g)"])
                    fc_df["Growth from Now (g)"] = (fc_df["Predicted Weight (g)"] - current_w).round(2)
                    st.dataframe(fc_df, use_container_width=True, hide_index=True)

                if hw >= 70:
                    st.success(f"✅ **Excellent yield!** Expected **{hw:.1f} g** — commercial grade target achieved.")
                elif hw >= 45:
                    st.info(f"ℹ️ **Good yield.** Expected **{hw:.1f} g**.")
                else:
                    st.warning(f"⚠️ **Moderate/Low yield.** Expected **{hw:.1f} g**. Review parameters.")

            else:
                st.markdown(
                    '<div class="warn-box">👆 Upload an image, set environmental parameters, '
                    'then press <b>Predict & Forecast</b>.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 ── MODEL PERFORMANCE (fully original, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
elif "Performance" in page:
    st.title("📈 Model Performance Dashboard")

    if model is None:
        st.warning("No trained model found. Place `lettuce_rf_model.pkl` in the app directory.")
    else:
        st.markdown("### 🎯 Overall Metrics (Testing Set)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score (Accuracy)",      "0.8252", delta="Target: >0.80")
        c2.metric("Mean Absolute Error (MAE)", "5.04 g", delta="- Low Error", delta_color="inverse")
        c3.metric("Root Mean Sq Error (RMSE)", "6.71 g")
        c4.metric("Algorithm",                "Random Forest")

        st.divider()
        st.markdown("### 📊 Statistical Evaluation")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.pyplot(plot_actual_vs_predicted(), use_container_width=True)
            plt.close()
            st.caption("Shows how closely the Random Forest predictions match the true harvest weights. "
                       "Points closer to the red line indicate higher accuracy.")
        with col_chart2:
            st.pyplot(plot_residuals(), use_container_width=True)
            plt.close()
            st.caption("Distribution of prediction errors. A bell curve centered around zero "
                       "(red line) confirms the model is unbiased.")

        st.divider()
        st.markdown("### 🧬 Feature Impact Analysis")
        fig_fi = plot_feature_importance(model)
        if fig_fi:
            st.pyplot(fig_fi, use_container_width=True)
            plt.close()
            st.info("The Feature Importance chart ranks which variables the Random Forest algorithm "
                    "relies on the most. As seen, `PCA_cm2` and `Age_Days` heavily dictate the final output.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 ── ABOUT (fully original, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
elif "About" in page:
    st.title("ℹ️ About This System")
    st.markdown("""
## Non-Contact Lettuce Yield Prediction System
Final-year undergraduate research thesis —
**University of Colombo | Faculty of Technology | Department of Agricultural Technology**

---

### Research Overview
| Field | Details |
|-------|---------|
| **Crop** | Butterhead Lettuce — Rex RZ Cultivar |
| **System** | Six-channel horizontal NFT hydroponic |
| **Imaging Capture** | Mobile Phone Camera (Non-contact) |
| **Segmentation** | Advanced Contour Detection, Bilateral Filter, ExG & HSV Masking |
| **ML model** | Random Forest Regressor (`lettuce_rf_model.pkl`) |
| **Vision AI** | Gemini 1.5 Flash — structured JSON (weight + agronomist explanation) |
| **Batch Mode** | ZIP upload → per-plant PCA + ML + AI → DataFrame, chart, CSV/PNG download |
| **Features (7)**| Age, PCA_cm2, Temp, RH, pH, EC, Days_Until_Harvest |

**Supervisor:** Dr. Thilanka Ariyawansha  
**Student:** G.D. Udara Indrakantha Disanayaka (2021t00960)

---

### v2.0 Upgrades
1. **AI Fallback Bug Fixed** — Gemini now returns `{"adjusted_weight": ..., "explanation": ...}` JSON.  
   Errors are displayed via `st.error()` instead of silently falling back.
2. **Batch ZIP Processing** — Upload a ZIP of plant images for bulk prediction.
3. **Batch Outputs** — Summary metrics, per-plant DataFrame, bar chart, CSV & PNG downloads.
4. **Secrets Management** — `GEMINI_API_KEY` still read from `st.secrets` only.
    """)
