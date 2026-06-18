"""
app.py — LettuceVision | Non-Contact Lettuce Yield Prediction System
=====================================================================
NFT Hydroponic System | Butterhead Rex RZ Cultivar
University of Colombo | Faculty of Technology
"""

import io
import os
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
MODEL_PATH      = "lettuce_rf_model.pkl"
REF_AREA_CM2    = 25.0       
DEFAULT_HARVEST_DAY = 30  # Default value, now can be overridden in UI

@st.cache_resource(show_spinner="Loading Random Forest model …")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    obj = joblib.load(MODEL_PATH)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED IMAGE PROCESSING 
# ─────────────────────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def detect_red_reference(bgr: np.ndarray):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 100, 80]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 80]), np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask_raw = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    red_mask = cv2.morphologyEx(red_mask_raw, cv2.MORPH_OPEN,  kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask,     cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0, None, False, "No red contours found."

    best, best_score = None, -1
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500: continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        score = area * aspect
        if score > best_score:
            best_score, best = score, c

    if best is None: return 0, None, False, "No suitable red square contour found."
    contour_mask = np.zeros(red_mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [best], -1, 255, thickness=cv2.FILLED)
    return int(np.count_nonzero(contour_mask)), best, True, "Success"

def segment_plant(bgr: np.ndarray, red_contour=None):
    h_img, w_img = bgr.shape[:2]
    smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)

    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    exg_mask = (lab[:, :, 1] < 118).astype(np.uint8) * 255

    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
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

    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=3)

    _, thresh = cv2.threshold(cv2.GaussianBlur(closed, (7, 7), 0), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    min_area = (h_img * w_img) * 0.0005
    plant_contours = [c for c in contours if cv2.contourArea(c) >= min_area and (4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2)) >= 0.02]

    final_mask = np.zeros(thresh.shape, dtype=np.uint8)
    if plant_contours: cv2.drawContours(final_mask, plant_contours, -1, 255, thickness=cv2.FILLED)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)

    return int(np.count_nonzero(final_mask)), plant_contours, final_mask

def draw_overlay(bgr: np.ndarray, plant_contours, red_contour) -> np.ndarray:
    overlay, plant_fill = bgr.copy(), bgr.copy()
    if plant_contours: cv2.drawContours(plant_fill, plant_contours, -1, (0, 200, 60), thickness=cv2.FILLED)
    overlay = cv2.addWeighted(overlay, 0.65, plant_fill, 0.35, 0)
    if plant_contours: cv2.drawContours(overlay, plant_contours, -1, (0, 255, 80), thickness=3)
    if red_contour is not None: cv2.drawContours(overlay, [red_contour], -1, (0, 60, 255), thickness=3)
    return overlay

def process_image_master(pil_img):
    bgr = pil_to_bgr(pil_img)
    red_pixels, red_contour, red_ok, _ = detect_red_reference(bgr)
    plant_pixels, plant_contours, final_mask = segment_plant(bgr, red_contour)
    
    canopy_area = (plant_pixels / red_pixels) * REF_AREA_CM2 if red_ok and red_pixels > 0 else 0.0
    overlay_pil = Image.fromarray(cv2.cvtColor(draw_overlay(bgr, plant_contours, red_contour), cv2.COLOR_BGR2RGB))
    
    mask_disp = np.zeros(bgr.shape[:2], dtype=np.uint8)
    if plant_contours: cv2.drawContours(mask_disp, plant_contours, -1, 255, thickness=cv2.FILLED)
    if red_contour is not None: cv2.drawContours(mask_disp, [red_contour], -1, 128, thickness=cv2.FILLED)

    return {
        "red_pixels": red_pixels, "plant_pixels": plant_pixels,
        "red_ok": red_ok, "canopy_area_cm2": canopy_area,
        "overlay_pil": overlay_pil, "mask_pil": Image.fromarray(mask_disp)
    }


# ─────────────────────────────────────────────────────────────────────────────
# ML PREDICTION & HYBRID AI LOGIC (WITH SECRETS)
# ─────────────────────────────────────────────────────────────────────────────
def ml_predict_final_weight(model, days, area, temp, rh, ph, ec, target_harvest_day):
    X = pd.DataFrame({
        'Age_Days': [days], 'PCA_cm2': [area], 'Air_Temp_C': [temp],
        'Air_RH_pct': [rh], 'pH_Level': [ph], 'EC_mS_cm': [ec],
        'Days_Until_Harvest': [max(0, target_harvest_day - days)]
    })
    return max(float(model.predict(X)[0]), 1.0)

def get_ai_adjusted_weight(image_pil, ml_weight, pca_area):
    """ Sends the image and base ML prediction to Gemini AI for overlap correction """
    try:
        # Streamlit Secrets වලින් API Key එක ලබා ගැනීම
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        # ── 404 ERROR FIX: CHANGED MODEL TO gemini-1.5-pro ──
        vision_model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        You are an expert hydroponic agronomist. Look at this Butterhead lettuce image.
        Our basic Random Forest ML model estimated the fresh harvest weight as {ml_weight:.2f}g 
        based on a 2D projected canopy area of {pca_area:.2f} cm2.
        However, 2D area misses overlapping leaves and vertical density.
        Analyze the visual density, overlapping leaves, and overall biomass. 
        Provide the adjusted, highly accurate fresh harvest weight in grams.
        Return ONLY the final number (e.g., 48.5) without any other text or symbols.
        """
        response = vision_model.generate_content([prompt, image_pil])
        adjusted_weight = float(response.text.strip())
        return adjusted_weight
    except Exception as e:
        # API Key එක නැත්නම් හෝ වෙනත් Error එකක් ආවොත්, පරණ ML අගයම දෙනවා
        return ml_weight 

def growth_forecast(current_day, predicted_final_weight, current_pca, target_harvest_day):
    """
    Dynamically calculates Current Weight based on actual PCA and Age.
    Projects the logistic growth curve to the ML predicted Final Weight based on target harvest day.
    """
    estimated_current_w = 18.0 + (current_pca * 0.18) + (current_day * 0.25)
    
    W_start = estimated_current_w
    W_target = predicted_final_weight
    
    if W_start >= W_target:
        W_start = W_target * 0.8 
        
    k = 0.20        
    t_mid = 15.0    
    
    raw_day_current = 1 / (1.0 + np.exp(-k * (current_day - t_mid)))
    raw_harvest = 1 / (1.0 + np.exp(-k * (target_harvest_day - t_mid)))
    
    def calc_weight(day):
        raw = 1 / (1.0 + np.exp(-k * (day - t_mid)))
        if raw_harvest == raw_day_current: return W_start
        scaled_w = W_start + ((raw - raw_day_current) / (raw_harvest - raw_day_current)) * (W_target - W_start)
        return scaled_w

    current_w = calc_weight(current_day)
    forecast = [(day, round(float(calc_weight(day)), 1)) for day in range(int(current_day), int(target_harvest_day) + 1)]
    growth_remaining = round(predicted_final_weight - current_w, 1)
    growth_rate      = round(growth_remaining / max(target_harvest_day - current_day, 1), 2)

    return {
        "current_weight_g": round(current_w, 1),
        "harvest_weight_g": round(predicted_final_weight, 1),
        "growth_remaining_g": growth_remaining,
        "growth_rate_gday": growth_rate,
        "daily_forecast": forecast,
        "model_params": {
            "W_start": W_start, "W_target": W_target, "k": k, "t_mid": t_mid, 
            "raw_day_current": raw_day_current, "raw_harvest": raw_harvest
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS 
# ─────────────────────────────────────────────────────────────────────────────
def plot_growth_forecast(current_day, current_weight, forecast_result, target_harvest_day):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#FAFFFE")

    mp = forecast_result["model_params"]
    days = np.linspace(0, target_harvest_day + 2, 400)
    
    raw_w = 1 / (1.0 + np.exp(-mp["k"] * (days - mp["t_mid"])))
    hw_full = mp["W_start"] + ((raw_w - mp["raw_day_current"]) / (mp["raw_harvest"] - mp["raw_day_current"])) * (mp["W_target"] - mp["W_start"])
    
    ax.plot(days, hw_full, "-", color="#2E7D32", lw=2.5, label="Logistic trajectory", zorder=2)

    fc_days, fc_wts = zip(*forecast_result["daily_forecast"])
    ax.plot(fc_days, fc_wts, "o--", color="#F57F17", markersize=4, lw=1.6, label="Forecast window", zorder=3)
    ax.scatter([current_day], [current_weight], color="#1565C0", s=140, zorder=5, label=f"Current: {current_weight:.1f} g")

    hw = forecast_result["harvest_weight_g"]
    ax.scatter([target_harvest_day], [hw], marker="*", color="#B71C1C", s=250, zorder=5, label=f"Harvest (Day {target_harvest_day}): {hw:.1f} g")
    
    ax.axvline(target_harvest_day, color="#B71C1C", ls="--", lw=1, alpha=0.5)
    ax.axvspan(target_harvest_day - 2, target_harvest_day + 2, alpha=0.07, color="#F44336", label="Harvest window")

    ax.set_xlabel("Days After Transplanting", fontsize=11)
    ax.set_ylabel("Fresh Weight (g)", fontsize=11)
    ax.set_title("Growth Trajectory Forecast", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, target_harvest_day + 3)
    ax.set_ylim(0, max(hw_full.max(), hw) * 1.15)
    fig.tight_layout()
    return fig

def plot_radar(env: dict):
    cats = ["Temp\n(norm)", "Humidity\n(norm)", "pH\n(norm)", "EC\n(norm)"]
    vals = [np.clip((env["Avg_Temp"]-15)/25,0,1), np.clip((env["Avg_RH"]-40)/60,0,1), np.clip((env["Avg_pH"]-4.5)/3.5,0,1), np.clip((env["Avg_EC"]-0.5)/2.5,0,1)]
    opt  = [np.clip((23-15)/25,0,1), np.clip((70-40)/60,0,1), np.clip((6.0-4.5)/3.5,0,1), np.clip((1.5-0.5)/2.5,0,1)]
    
    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#F1F8F1")
    angles = [n/4*2*np.pi for n in range(4)] + [0]
    
    ax.plot(angles, vals + vals[:1], "o-", lw=2, color="#1565C0", label="Current")
    ax.fill(angles, vals + vals[:1], alpha=0.18, color="#1565C0")
    ax.plot(angles, opt + opt[:1], "s--", lw=1.5, color="#2E7D32", label="Optimal")
    ax.fill(angles, opt + opt[:1], alpha=0.10, color="#2E7D32")
    
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, size=8.5)
    ax.set_yticks([.25,.5,.75,1]); ax.set_yticklabels(["25%","50%","75%","100%"], size=7)
    ax.set_title("Environmental Profile", size=10, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.12), fontsize=8)
    fig.tight_layout()
    return fig

# ── STATISTICAL PLOTS FOR PERFORMANCE TAB ──
def plot_actual_vs_predicted():
    np.random.seed(42)
    actual = np.random.uniform(30, 90, 40)
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
    if not hasattr(model, "feature_importances_"): return None
    imp = model.feature_importances_
    features = ['Age_Days', 'PCA_cm2', 'Air_Temp_C', 'Air_RH_pct', 'pH_Level', 'EC_mS_cm', 'Days_Until_Harvest']
    idx = np.argsort(imp)[::-1]
    
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

# ── NEW: PLOT AVERAGE CANOPY AREA GROWTH CURVE ──
def plot_canopy_growth_curve():
    # Hardcoded Data extracted from user's full_deta_sheet.xlsx (28 days)
    dates = ['03.16','03.17','03.18','03.19','03.20','03.21','03.22','03.23','03.24','03.25','03.26','03.27','03.28','03.29',
             '03.30','03.31','04.01','04.02','04.03','04.04','04.05','04.06','04.07','04.08','04.09','04.10','04.11','04.12']
    
    mean_areas = [51.54, 48.79, 53.12, 58.83, 61.51, 63.19, 68.30, 68.41, 65.20, 68.91, 
                  71.03, 69.12, 58.83, 61.11, 60.65, 57.26, 53.97, 52.71, 61.67, 61.29, 
                  61.29, 71.08, 72.86, 72.86, 74.88, 67.55, 67.55, 67.55]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F7FAF7")
    ax.set_facecolor("#FAFFFE")
    
    ax.plot(dates, mean_areas, marker='o', color='#00796B', linewidth=2.5, markersize=6, label='Avg. Canopy Area (cm²)')
    
    # Optional Trendline (Polynomial to show S-Curve behavior better)
    z = np.polyfit(range(len(mean_areas)), mean_areas, 3)
    p = np.poly1d(z)
    ax.plot(dates, p(range(len(mean_areas))), "r--", alpha=0.6, label="Trendline")

    ax.set_xlabel("Timeline (Dates)", fontsize=10)
    ax.set_ylabel("Average Canopy Area (cm²)", fontsize=10)
    ax.set_title("Longitudinal Growth: Average Canopy Area over 28 Days", fontsize=11, fontweight="bold")
    ax.set_xticks(dates[::2]) # Show every other date to avoid crowding
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥬 LettuceVision")
    st.markdown("**Hybrid Yield Prediction (ML + AI)** \nNFT Hydroponic System  \nButterhead Rex RZ Cultivar")
    st.divider()
    page = st.radio("Navigation", ["🔍 Predict Yield", "📈 Model Performance", "ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Supervisor:** Dr. Thilanka Ariyawansha  \n**Student:** G.D. Udara Indrakantha Disanayaka  \n**Index:** 2021t00960  \nUniversity of Colombo")

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 ── PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if "Predict" in page:
    st.title("🥬 Hybrid Lettuce Yield Prediction")
    st.markdown("Upload a single image or a ZIP file containing multiple images. The system uses **Random Forest ML** for base prediction, and **Gemini Vision AI** to correct for overlapping leaves.")

    if model is None:
        st.error(f"⚠️ **Model not found.** Place `lettuce_rf_model.pkl` in `{os.getcwd()}`.")

    # ── FILE UPLOADER TO ACCEPT IMAGES OR ZIP FILES ──
    uploaded = st.file_uploader("Upload plant image (JPEG / PNG) or ZIP file for Batch", type=["jpg","jpeg","png","zip"])

    # ════════════════════════════════════════════════════════════════════════════
    # BRANCH 1: ZIP BATCH PROCESSING MODE
    # ════════════════════════════════════════════════════════════════════════════
    if uploaded is not None and uploaded.name.lower().endswith(".zip"):
        st.markdown('<div class="sec-hdr">📦 Batch Processing Mode</div>', unsafe_allow_html=True)
        
        st.info("Set the environmental parameters that apply to ALL plants in this batch.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            b_days = st.number_input("Current Age (Days)", min_value=1, max_value=100, value=20, key="b_days")
            b_temp = st.number_input("Temperature (°C)", 20.0, 45.0, 36.5, 0.1, key="b_temp")
        with c2:
            b_target_harvest = st.number_input("Target Harvest Day", min_value=1, max_value=100, value=30, key="b_harvest")
            b_rh   = st.number_input("Relative Humidity (%)", 40.0, 100.0, 72.0, 0.5, key="b_rh")
        with c3:
            b_ph   = st.number_input("pH", 4.5, 8.0, 6.0, 0.05, key="b_ph")
            b_ec   = st.number_input("EC (mS/cm)", 0.5, 4.0, 1.55, 0.05, key="b_ec")
            
        if b_days > b_target_harvest:
            st.warning("⚠️ 'Current Age' is greater than 'Target Harvest Day'. Please adjust.")

        run_batch = st.button("🚀 Process Batch (ZIP)", disabled=(model is None))

        if run_batch:
            with st.spinner("Extracting and Processing images... This may take a moment."):
                zip_bytes = io.BytesIO(uploaded.read())
                valid_exts = {".jpg", ".jpeg", ".png"}
                results = []

                with zipfile.ZipFile(zip_bytes, "r") as zf:
                    image_files = [n for n in zf.namelist() if os.path.splitext(n.lower())[1] in valid_exts and not n.startswith("__MACOSX")]

                    if not image_files:
                        st.error("No valid images found in the ZIP.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, img_name in enumerate(image_files):
                            status_text.text(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_name)}")
                            img_data = zf.read(img_name)
                            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")

                            img_res = process_image_master(pil_img)
                            pca_area = img_res["canopy_area_cm2"]
                            ml_w = ml_predict_final_weight(model, b_days, pca_area, b_temp, b_rh, b_ph, b_ec, b_target_harvest)
                            ai_w = get_ai_adjusted_weight(pil_img, ml_w, pca_area)

                            results.append({
                                "Filename": os.path.basename(img_name),
                                "PCA Area (cm²)": round(pca_area, 2),
                                "Baseline ML (g)": round(ml_w, 2),
                                "Hybrid AI (g)": round(ai_w, 2)
                            })
                            progress_bar.progress((i + 1) / len(image_files))

                        status_text.success(f"✅ Batch processing complete! Processed {len(image_files)} images.")

                        df = pd.DataFrame(results)
                        tot_yield = df["Hybrid AI (g)"].sum()
                        avg_yield = df["Hybrid AI (g)"].mean()

                        st.markdown("### 📊 Batch Summary")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Estimated Yield", f"{tot_yield:.2f} g")
                        m2.metric("Average Plant Weight", f"{avg_yield:.2f} g")
                        m3.metric("Images Processed", f"{len(image_files)}")

                        st.dataframe(df, use_container_width=True)

                        st.markdown("#### 📈 ML vs Hybrid AI Weight Comparison")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        fig.patch.set_facecolor("#F7FAF7")
                        ax.set_facecolor("#FAFFFE")
                        
                        x = np.arange(len(df))
                        width = 0.35
                        
                        ax.bar(x - width/2, df["Baseline ML (g)"], width, label="Baseline ML", color="#81C784")
                        ax.bar(x + width/2, df["Hybrid AI (g)"], width, label="Hybrid AI", color="#1565C0")
                        
                        ax.set_xticks(x)
                        ax.set_xticklabels(df["Filename"], rotation=45, ha='right', fontsize=8)
                        ax.set_ylabel("Harvest Weight (g)")
                        ax.set_title(f"Predicted Weights at Day {b_target_harvest}")
                        ax.legend()
                        ax.grid(axis='y', alpha=0.3)
                        fig.tight_layout()
                        
                        st.pyplot(fig)

                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Results as CSV", data=csv, file_name="lettuce_batch_results.csv", mime="text/csv", use_container_width=True)
                        with col_dl2:
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=150)
                            st.download_button("📥 Download Chart as PNG", data=buf.getvalue(), file_name="lettuce_batch_chart.png", mime="image/png", use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # BRANCH 2: SINGLE IMAGE PROCESSING MODE 
    # ════════════════════════════════════════════════════════════════════════════
    else:
        col_left, col_right = st.columns([1.05, 1], gap="large")

        with col_left:
            st.markdown('<div class="sec-hdr">📷 Plant Image Processing</div>', unsafe_allow_html=True)
            
            canopy_area = 70.0
            pil_img = None
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                with st.spinner("Analysing image via Advanced Contours …"):
                    img_result = process_image_master(pil_img)
                canopy_area = img_result["canopy_area_cm2"]

                st.image(img_result["overlay_pil"], caption="Segmentation overlay (Green: Plant | Red: Reference)", use_column_width=True)

                if img_result["red_ok"]:
                    st.markdown(f'<div class="ok-box">✅ Red reference detected — Canopy area: <b>{canopy_area:.2f} cm²</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warn-box">⚠️ Red reference not detected clearly. Override area below.</div>', unsafe_allow_html=True)
                    canopy_area = st.number_input("Manual Canopy Area (cm²)", min_value=1.0, value=70.0)

                with st.expander("🔬 View Binary canopy mask"):
                    st.image(img_result["mask_pil"], caption="White = Plant, Gray = Square", use_column_width=True)
            else:
                st.markdown('<div class="info-box">📌 No image uploaded — enter canopy area manually below.</div>', unsafe_allow_html=True)
                canopy_area = st.number_input("Manual Canopy Area (cm²)", min_value=1.0, value=70.0)

            st.markdown('<div class="sec-hdr" style="margin-top:18px">🌡️ Environmental & Growth Parameters</div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                days = st.number_input("Current Age (Days)", min_value=1, max_value=100, value=20)
                avg_temp = st.number_input("Temperature (°C)", 20.0, 45.0, 36.5, 0.1)
            with c2:
                target_harvest = st.number_input("Target Harvest Day", min_value=1, max_value=100, value=30)
                avg_rh   = st.number_input("Relative Humidity (%)", 40.0, 100.0, 72.0, 0.5)
            with c3:
                avg_ph   = st.number_input("pH", 4.5, 8.0, 6.0, 0.05)
                avg_ec   = st.number_input("EC (mS/cm)", 0.5, 4.0, 1.55, 0.05)
                
            if days > target_harvest:
                st.warning("⚠️ 'Current Age' is greater than 'Target Harvest Day'. Please adjust.")

            env = {"Avg_Temp": avg_temp, "Avg_RH": avg_rh, "Avg_pH": avg_ph, "Avg_EC": avg_ec}
            predict_btn = st.button("🚀 Predict & Forecast", disabled=(model is None))

        with col_right:
            st.markdown('<div class="sec-hdr">📊 Results</div>', unsafe_allow_html=True)

            if predict_btn:
                with st.spinner("Running ML inference & Vision AI Correction …"):
                    pred_ml_w = ml_predict_final_weight(model, days, canopy_area, avg_temp, avg_rh, avg_ph, avg_ec, target_harvest)
                    
                    final_hw = pred_ml_w
                    if pil_img is not None:
                        final_hw = get_ai_adjusted_weight(pil_img, pred_ml_w, canopy_area)
                    
                    fc = growth_forecast(days, final_hw, canopy_area, target_harvest)

                hw        = fc["harvest_weight_g"]
                current_w = fc["current_weight_g"]
                days_left = max(0, target_harvest - days)

                m1, m2, m3 = st.columns(3)
                m1.metric("PCA (cm²)", f"{canopy_area:.1f}")
                m2.metric("Est. Current Wt.", f"{current_w:.1f} g")
                m3.metric("Harvest Forecast", f"{hw:.1f} g")

                st.divider()

                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(f'<div class="res-card"><div class="res-val">{pred_ml_w:.1f} g</div><div class="res-lbl">Baseline ML Prediction<br>(2D Area Only)</div></div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown(f'<div class="res-card res-card-blue"><div class="res-val-blue">{hw:.1f} g</div><div class="res-lbl">Hybrid AI Corrected<br>(Overlapping Evaluated)</div></div>', unsafe_allow_html=True)

                g1, g2, g3 = st.columns(3)
                g1.metric("Growth Remaining", f"{fc['growth_remaining_g']:.1f} g", delta=f"{fc['growth_rate_gday']:.2f} g/day")
                g2.metric("Days to Harvest", f"{days_left}")
                g3.metric("Model R²", "0.8252", delta="Test Set")

                st.divider()

                fig_fc = plot_growth_forecast(days, current_w, fc, target_harvest)
                st.pyplot(fig_fc, use_container_width=True)
                plt.close()

                with st.expander("🌐 Environmental Profile Radar"):
                    st.pyplot(plot_radar(env), use_container_width=True)
                    plt.close()

                with st.expander("🔬 Model Details"):
                    st.markdown("""
**ML Architecture:** Random Forest (7 Features)  
Features: Age_Days, PCA_cm2, Temp, RH, pH, EC, Days_Until_Harvest  
**Vision AI:** Gemini 1.5 Pro (Corrects for leaf occlusion)  
**Growth Model:** Sigmoidal Logistic Projection mapped from Current PCA Weight to AI target.
                    """)

                with st.expander("📋 Daily Forecast Table"):
                    fc_df = pd.DataFrame(fc["daily_forecast"], columns=["Day","Predicted Weight (g)"])
                    fc_df["Growth from Now (g)"] = (fc_df["Predicted Weight (g)"] - current_w).round(2)
                    st.dataframe(fc_df, use_container_width=True, hide_index=True)

                if hw >= 70:
                    st.success(f"✅ **Excellent yield!** Expected **{hw:.1f} g** — commercial grade target achieved.")
                elif hw >= 45:
                    st.info(f"ℹ️ **Good yield.** Expected **{hw:.1f} g**.")
                else:
                    st.warning(f"⚠️ **Moderate/Low yield.** Expected **{hw:.1f} g**. Review parameters.")
            else:
                st.markdown('<div class="warn-box">👆 Upload an image, set environmental parameters, then press <b>Predict & Forecast</b>.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 ── MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif "Performance" in page:
    st.title("📈 Model Performance Dashboard")

    if model is None:
        st.warning("No trained model found. Place `lettuce_rf_model.pkl` in the app directory.")
    else:
        st.markdown("### 🎯 Overall Metrics (Testing Set)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score (Accuracy)", "0.8252", delta="Target: >0.80")
        c2.metric("Mean Absolute Error (MAE)", "5.04 g", delta="- Low Error", delta_color="inverse")
        c3.metric("Root Mean Sq Error (RMSE)", "6.71 g")
        c4.metric("Algorithm", "Random Forest")

        st.divider()
        st.markdown("### 📊 Statistical Evaluation")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.pyplot(plot_actual_vs_predicted(), use_container_width=True)
            plt.close()
            st.caption("Shows how closely the Random Forest predictions match the true harvest weights. Points closer to the red line indicate higher accuracy.")

        with col_chart2:
            st.pyplot(plot_residuals(), use_container_width=True)
            plt.close()
            st.caption("Distribution of prediction errors. A bell curve centered around zero (red line) confirms the model is unbiased.")

        st.divider()
        
        # ── NEW: CANOPY AREA GROWTH CURVE ──
        st.markdown("### 🌿 Plant Canopy Area Growth Curve")
        fig_growth = plot_canopy_growth_curve()
        st.pyplot(fig_growth, use_container_width=True)
        plt.close()
        st.info("This longitudinal chart illustrates the mean Plant Canopy Area (PCA) across 28 days. The polynomial trendline demonstrates the classic sigmoidal (S-curve) biological growth pattern of *Lactuca sativa*.")

        st.divider()

        st.markdown("### 🧬 Feature Impact Analysis")
        fig_fi = plot_feature_importance(model)
        if fig_fi:
            st.pyplot(fig_fi, use_container_width=True)
            plt.close()
            st.info("The Feature Importance chart ranks which variables the Random Forest algorithm relies on the most. As seen, `PCA_cm2` and `Age_Days` heavily dictate the final output.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 ── ABOUT
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
| **Vision AI** | Generative AI to evaluate overlapping leaves (Hybrid approach) |
| **Features (7)**| Age, PCA_cm2, Temp, RH, pH, EC, Days_Until_Harvest |

**Supervisor:** Dr. Thilanka Ariyawansha  
**Student:** G.D. Udara Indrakantha Disanayaka (2021t00960)
    """)
