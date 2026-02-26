# portfolio/goals.py — Goal-Based Investing Engine
#
# Goals: Retirement, FIRE, Education, Marriage, Savings (Emergency/General)
# Horizon: 1–40 years
#
# Each goal profile defines:
#   - Required return (based on goal corpus needed)
#   - Risk guardrails (override or constrain user risk)
#   - Asset allocation guidance (equity %, debt %)
#   - Rebalancing strategy per phase
#   - SIP recommendation
#   - Portfolio constraints (sector tilts, max drawdown tolerance)
#   - Projection: corpus at end of horizon

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

INFLATION_RATE   = 0.06   # India avg ~6%
EQUITY_LONG_TERM = 0.12   # Expected long-term NIFTY CAGR
DEBT_RETURN      = 0.07   # Debt fund / FD return
TRADING_DAYS_PER_YEAR = 252

# ─────────────────────────────────────────────────────────────────────────────
# GOAL PROFILES
# ─────────────────────────────────────────────────────────────────────────────

GOAL_PROFILES = {

    "retirement": {
        "display_name": "Retirement",
        "emoji": "🏖️",
        "description": "Build a corpus to sustain post-retirement life",
        "typical_horizon": (15, 40),
        "inflation_adjustment": True,
        # Equity glidepath: start aggressive, de-risk as you near goal
        # Format: {years_to_goal: equity_pct}
        "equity_glidepath": {
            30: 0.90, 20: 0.80, 15: 0.75, 10: 0.65, 7: 0.55, 5: 0.45, 3: 0.35, 1: 0.25
        },
        # Sector preferences (tilt weights)
        "sector_tilt": {
            "FMCG": 1.3, "Pharma": 1.2, "Banking": 1.1, "IT": 1.1,
            "Energy": 0.9, "Metals": 0.7,
        },
        "max_drawdown_tolerance": -0.30,   # can stomach -30%
        "required_return_buffer": 0.02,    # target 2% above inflation
        "notes": [
            "Focus on quality large-caps with consistent dividends",
            "Gradually shift to debt as retirement approaches (glidepath)",
            "Avoid high-beta cyclical stocks near retirement",
            "Consider ELSS for tax saving under 80C",
        ],
    },

    "fire": {
        "display_name": "FIRE (Financial Independence, Retire Early)",
        "emoji": "🔥",
        "description": "Aggressive wealth accumulation to retire early",
        "typical_horizon": (7, 20),
        "inflation_adjustment": True,
        # Stay high equity throughout — FIRE needs aggressive compounding
        "equity_glidepath": {
            20: 0.95, 15: 0.90, 10: 0.85, 7: 0.80, 5: 0.75, 3: 0.65, 1: 0.55
        },
        "sector_tilt": {
            "IT": 1.4, "Finance": 1.3, "Consumer": 1.2,
            "FMCG": 0.9, "Energy": 0.8,
        },
        "max_drawdown_tolerance": -0.40,   # high tolerance for FIRE investors
        "required_return_buffer": 0.04,    # needs to beat inflation by 4%+
        "notes": [
            "FIRE requires 25x annual expenses corpus (4% withdrawal rule)",
            "Maximise Sharpe ratio — need returns with manageable volatility",
            "Prioritise compounders: HDFC Bank, TCS, Asian Paints type stocks",
            "Review portfolio annually and rebalance aggressively",
            "Consider adding index funds alongside this equity portfolio",
        ],
    },

    "education": {
        "display_name": "Child's Education",
        "emoji": "🎓",
        "description": "Save for higher education (college/university fees)",
        "typical_horizon": (5, 18),
        "inflation_adjustment": True,
        # Education inflation in India ~10% — need aggressive returns
        "equity_glidepath": {
            15: 0.85, 10: 0.75, 7: 0.65, 5: 0.55, 3: 0.40, 2: 0.30, 1: 0.20
        },
        "sector_tilt": {
            "IT": 1.2, "Banking": 1.2, "FMCG": 1.1, "Pharma": 1.0,
            "Metals": 0.6, "Auto": 0.8,
        },
        "max_drawdown_tolerance": -0.25,   # moderate — money needed on fixed date
        "required_return_buffer": 0.04,    # education inflation ~10%, need 10%+ returns
        "notes": [
            "Education inflation in India runs at ~10% — plan accordingly",
            "Switch to debt 2-3 years before the target year",
            "Consider Sukanya Samriddhi (for girl child) alongside this portfolio",
            "Do NOT take undue risk in last 3 years — capital preservation matters",
            "Review every 6 months and rebalance",
        ],
    },

    "marriage": {
        "display_name": "Marriage / Wedding",
        "emoji": "💍",
        "description": "Save for a wedding (self or child)",
        "typical_horizon": (2, 10),
        "inflation_adjustment": True,
        "equity_glidepath": {
            10: 0.80, 7: 0.70, 5: 0.60, 3: 0.45, 2: 0.30, 1: 0.15
        },
        "sector_tilt": {
            "FMCG": 1.2, "Consumer": 1.2, "Banking": 1.1,
            "Metals": 0.7, "IT": 0.9,
        },
        "max_drawdown_tolerance": -0.20,   # lower — fixed date, can't wait for recovery
        "required_return_buffer": 0.02,
        "notes": [
            "This is a fixed-date goal — capital protection in final 2 years is critical",
            "Shift to liquid funds / FD for last 12-18 months",
            "Avoid small/mid cap exposure near target date",
            "Wedding inflation in India is high — add 8-10% inflation buffer to corpus",
        ],
    },

    "savings": {
        "display_name": "Savings / Wealth Building",
        "emoji": "💰",
        "description": "General wealth creation or emergency fund growth",
        "typical_horizon": (1, 10),
        "inflation_adjustment": False,
        "equity_glidepath": {
            10: 0.75, 7: 0.70, 5: 0.65, 3: 0.55, 2: 0.45, 1: 0.30
        },
        "sector_tilt": {
            "Banking": 1.1, "IT": 1.1, "FMCG": 1.1,
            "Metals": 0.8, "Energy": 0.9,
        },
        "max_drawdown_tolerance": -0.20,
        "required_return_buffer": 0.01,
        "notes": [
            "Keep 6 months expenses in liquid fund/FD before investing in equity",
            "Balanced approach — capital preservation + growth",
            "SIP is preferred over lump sum for savings goals",
            "Review quarterly and take profits if goal is near",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GOAL CONFIGURATION DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalConfig:
    """User-defined goal configuration."""
    goal_type: str                         # retirement | fire | education | marriage | savings
    investment_horizon_years: int          # how many years to invest
    lump_sum_amount: float                 # initial ₹ investment
    monthly_sip: float = 0.0              # optional monthly SIP (₹)
    target_corpus: Optional[float] = None  # optional: desired final ₹ value
    current_age: Optional[int] = None      # for retirement/FIRE planning
    target_age: Optional[int] = None       # for retirement/FIRE
    user_risk: str = "Medium"              # Low | Medium | High (can be overridden)

    def __post_init__(self):
        if self.goal_type not in GOAL_PROFILES:
            raise ValueError(
                f"Unknown goal type: '{self.goal_type}'. "
                f"Choose from: {list(GOAL_PROFILES.keys())}"
            )
        if self.investment_horizon_years < 1:
            raise ValueError("Horizon must be at least 1 year.")
        if self.investment_horizon_years > 40:
            raise ValueError("Horizon cannot exceed 40 years.")


# ─────────────────────────────────────────────────────────────────────────────
# HORIZON → RISK RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def resolve_effective_risk(goal_config: GoalConfig) -> str:
    """
    Determine effective risk level by combining:
      1. User's stated risk appetite
      2. Goal type requirements
      3. Investment horizon (short horizon = lower risk regardless)

    Returns final effective risk: "Low" | "Medium" | "High"
    """
    user_risk = goal_config.user_risk
    horizon   = goal_config.investment_horizon_years
    goal      = goal_config.goal_type
    profile   = GOAL_PROFILES[goal]

    # ── Horizon-based risk cap ─────────────────────────────────────────────
    # Short horizon = must protect capital regardless of what user says
    if horizon <= 2:
        horizon_risk = "Low"
    elif horizon <= 5:
        horizon_risk = "Medium"
    else:
        horizon_risk = "High"   # Long horizon can handle volatility

    # ── Goal-specific adjustment ───────────────────────────────────────────
    # Fixed-date goals (marriage, education near end) → cap risk
    if goal in ("marriage", "education") and horizon <= 3:
        goal_risk_cap = "Low"
    elif goal == "savings" and horizon <= 2:
        goal_risk_cap = "Low"
    elif goal == "fire":
        goal_risk_cap = "High"   # FIRE always benefits from more equity
    else:
        goal_risk_cap = None     # no override

    # ── Risk ordering ──────────────────────────────────────────────────────
    RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}

    levels = [RISK_ORDER[user_risk], RISK_ORDER[horizon_risk]]
    if goal_risk_cap:
        levels.append(RISK_ORDER[goal_risk_cap])

    # Take conservative: min of all constraints
    effective_idx = min(levels)
    effective_risk = {0: "Low", 1: "Medium", 2: "High"}[effective_idx]

    return effective_risk


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY ALLOCATION (from glidepath)
# ─────────────────────────────────────────────────────────────────────────────

def get_equity_allocation(goal_config: GoalConfig) -> float:
    """
    Get the recommended equity % for current horizon using glidepath.
    The rest goes to debt (not modelled in this equity pipeline,
    but communicated to user as guidance).
    """
    profile   = GOAL_PROFILES[goal_config.goal_type]
    glidepath = profile["equity_glidepath"]
    horizon   = goal_config.investment_horizon_years

    # Find nearest glidepath entry
    sorted_years = sorted(glidepath.keys(), reverse=True)
    for y in sorted_years:
        if horizon >= y:
            return glidepath[y]
    return glidepath[min(glidepath.keys())]


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

def project_corpus(
    goal_config: GoalConfig,
    portfolio_expected_return: float,    # annualised, from ML pipeline
    portfolio_volatility: float,         # annualised
) -> dict:
    """
    Project portfolio value at end of horizon.

    Models:
    - Lump sum compounding
    - SIP compounding (monthly)
    - Inflation-adjusted real value
    - Confidence bands (using volatility)

    Returns dict with projection data.
    """
    lump     = goal_config.lump_sum_amount
    sip      = goal_config.monthly_sip
    n_years  = goal_config.investment_horizon_years
    profile  = GOAL_PROFILES[goal_config.goal_type]
    r        = portfolio_expected_return       # equity return
    vol      = portfolio_volatility
    infl     = INFLATION_RATE if profile["inflation_adjustment"] else 0.0

    # Split lump sum into equity and debt portions
    equity_pct  = get_equity_allocation(goal_config)   # e.g. 0.65
    debt_pct    = 1.0 - equity_pct                     # e.g. 0.35
    lump_equity = lump * equity_pct
    lump_debt   = lump * debt_pct

    # Compound each separately
    lump_equity_final = lump_equity * ((1 + r) ** n_years)
    lump_debt_final   = lump_debt   * ((1 + DEBT_RETURN) ** n_years)
    lump_final        = lump_equity_final + lump_debt_final

    # Blended return for SIP projection
    r_blended = r * equity_pct + DEBT_RETURN * debt_pct

    # ── SIP projection (monthly compounding) ──────────────────────────────
    r_monthly = (1 + r_blended) ** (1/12) - 1
    if r_monthly > 0 and sip > 0:
        sip_final = sip * (((1 + r_monthly) ** (n_years * 12) - 1) / r_monthly) * (1 + r_monthly)
    else:
        sip_final = sip * n_years * 12

    total_corpus = lump_final + sip_final

    # ── Inflation-adjusted real value ─────────────────────────────────────
    real_corpus = total_corpus / ((1 + infl) ** n_years)

    # ── Total invested ─────────────────────────────────────────────────────
    total_invested = lump + sip * 12 * n_years

    # ── Confidence bands (±1σ and ±2σ) ────────────────────────────────────
    # Using lognormal approximation for long horizon
    mu_log  = (r - 0.5 * vol**2) * n_years
    sig_log = vol * math.sqrt(n_years)

    p10  = lump_equity * math.exp(mu_log - 1.28 * sig_log)  + lump_debt_final + sip_final
    p25  = lump_equity * math.exp(mu_log - 0.674 * sig_log) + lump_debt_final + sip_final
    p50  = lump_equity * math.exp(mu_log)                   + lump_debt_final + sip_final
    p75  = lump_equity * math.exp(mu_log + 0.674 * sig_log) + lump_debt_final + sip_final
    p90  = lump_equity * math.exp(mu_log + 1.28 * sig_log)  + lump_debt_final + sip_final

    # ── CAGR ──────────────────────────────────────────────────────────────
    cagr = (total_corpus / max(total_invested, 1)) ** (1 / n_years) - 1

    # ── Year-by-year projection ───────────────────────────────────────────
    yearly = []
    current_val = lump
    for yr in range(1, n_years + 1):
        # current_val = current_val * (1 + r) + sip * 12 * (1 + r_monthly * 6)
        current_val = current_val * (1 + r_blended) + sip * 12 * (1 + r_monthly * 6)
        real_val    = current_val / ((1 + infl) ** yr)
        invested_so_far = lump + sip * 12 * yr
        yearly.append({
            "year":       yr,
            "corpus":     round(current_val, 0),
            "real_value": round(real_val, 0),
            "invested":   round(invested_so_far, 0),
            "gain":       round(current_val - invested_so_far, 0),
        })

    return {
        "total_corpus":    round(total_corpus, 0),
        "lump_final":      round(lump_final, 0),
        "sip_final":       round(sip_final, 0),
        "real_corpus":     round(real_corpus, 0),
        "total_invested":  round(total_invested, 0),
        "total_gain":      round(total_corpus - total_invested, 0),
        "cagr":            round(cagr * 100, 2),
        "wealth_multiple": round(total_corpus / max(total_invested, 1), 2),
        "p10":  round(p10, 0),
        "p25":  round(p25, 0),
        "p50":  round(p50, 0),
        "p75":  round(p75, 0),
        "p90":  round(p90, 0),
        "yearly_projection": yearly,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIP RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

def recommend_sip(
    goal_config: GoalConfig,
    target_corpus: float,
    portfolio_expected_return: float,
) -> float:
    """
    Given a target corpus and lump sum already invested,
    compute the monthly SIP needed to reach the target.
    """
    lump   = goal_config.lump_sum_amount
    n      = goal_config.investment_horizon_years
    r      = portfolio_expected_return
    r_m    = (1 + r) ** (1/12) - 1

    lump_final = lump * (1 + r) ** n
    remaining  = target_corpus - lump_final

    if remaining <= 0:
        return 0.0  # lump sum alone is enough

    if r_m <= 0:
        return remaining / (n * 12)

    # SIP formula rearranged: P = FV * r_m / (((1+r_m)^n - 1) * (1+r_m))
    months = n * 12
    sip_needed = remaining * r_m / (((1 + r_m) ** months - 1) * (1 + r_m))
    return round(max(sip_needed, 0), 0)


# ─────────────────────────────────────────────────────────────────────────────
# GOAL-BASED PORTFOLIO CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────

def get_goal_constraints(goal_config: GoalConfig) -> dict:
    """
    Returns additional constraints to pass to the portfolio optimizer.
    These override or augment the base config constraints.
    """
    profile  = GOAL_PROFILES[goal_config.goal_type]
    horizon  = goal_config.investment_horizon_years
    equity_pct = get_equity_allocation(goal_config)

    constraints = {
        "max_drawdown_tolerance":   profile["max_drawdown_tolerance"],
        "sector_tilt":              profile["sector_tilt"],
        "equity_allocation":        equity_pct,
        "debt_allocation":          round(1 - equity_pct, 2),
        "effective_risk":           resolve_effective_risk(goal_config),
        "required_return_min":      INFLATION_RATE + profile["required_return_buffer"],
    }

    # Near-goal penalty: if <3 years, be extra conservative
    if horizon <= 3:
        constraints["max_weight"]         = 0.15   # lower concentration
        constraints["min_stocks"]         = 10     # more diversification
        constraints["vol_penalty_factor"] = 2.0    # extra penalise volatility in objective

    return constraints


# ─────────────────────────────────────────────────────────────────────────────
# PHASE DETECTION (for multi-phase investing)
# ─────────────────────────────────────────────────────────────────────────────

def get_investment_phase(horizon: int, goal_type: str) -> dict:
    """
    Returns the current investment phase:
      - Accumulation: building corpus (equity heavy)
      - Transition: shifting from equity to debt
      - Preservation: protecting corpus (near goal)
    """
    if goal_type == "fire":
        if horizon > 5:
            return {"phase": "Accumulation", "action": "Stay fully invested in equity. Maximise compounding."}
        elif horizon > 2:
            return {"phase": "Transition", "action": "Begin shifting 10-15% to debt each year."}
        else:
            return {"phase": "Preservation", "action": "Move 50%+ to debt/liquid. Protect corpus."}

    elif goal_type == "retirement":
        if horizon > 10:
            return {"phase": "Accumulation", "action": "Equity-heavy portfolio. Focus on wealth creation."}
        elif horizon > 5:
            return {"phase": "Transition", "action": "Glidepath activated. Reduce equity 5% per year."}
        else:
            return {"phase": "Preservation", "action": "De-risk significantly. Shift to balanced/debt funds."}

    else:  # education, marriage, savings
        if horizon > 5:
            return {"phase": "Accumulation", "action": "Build corpus aggressively."}
        elif horizon > 2:
            return {"phase": "Transition", "action": "Start moving to lower-volatility stocks and debt."}
        else:
            return {"phase": "Preservation", "action": "Capital protection mode. Move majority to debt/FD."}


# ─────────────────────────────────────────────────────────────────────────────
# GOAL REPORT FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def format_goal_report(
    goal_config: GoalConfig,
    projection: dict,
    portfolio: dict,
    constraints: dict,
) -> str:
    """Generate a comprehensive goal-based planning report."""
    profile      = GOAL_PROFILES[goal_config.goal_type]
    phase_info   = get_investment_phase(goal_config.investment_horizon_years, goal_config.goal_type)
    equity_pct   = constraints["equity_allocation"] * 100
    debt_pct     = constraints["debt_allocation"] * 100
    effective_risk = constraints["effective_risk"]

    lines = []
    lines.append("╔" + "═" * 62 + "╗")
    lines.append(f"║  {profile['emoji']}  GOAL-BASED INVESTMENT PLAN" + " " * 30 + "║")
    lines.append("╠" + "═" * 62 + "╣")
    lines.append(f"║  Goal        : {profile['display_name']:<47}║")
    lines.append(f"║  Horizon     : {goal_config.investment_horizon_years} years" + " " * (46 - len(str(goal_config.investment_horizon_years))) + "║")
    lines.append(f"║  Lump Sum    : ₹{goal_config.lump_sum_amount:>12,.0f}" + " " * 32 + "║")
    if goal_config.monthly_sip > 0:
        lines.append(f"║  Monthly SIP : ₹{goal_config.monthly_sip:>12,.0f}" + " " * 32 + "║")
    lines.append(f"║  Risk (user) : {goal_config.user_risk:<47}║")
    lines.append(f"║  Risk (eff.) : {effective_risk:<47}║")
    lines.append("╚" + "═" * 62 + "╝")

    # ── Phase ──────────────────────────────────────────────────────────────
    lines.append(f"\n  📍 INVESTMENT PHASE: {phase_info['phase'].upper()}")
    lines.append(f"  → {phase_info['action']}")

    # ── Asset Allocation ───────────────────────────────────────────────────
    lines.append(f"\n  📊 RECOMMENDED ASSET ALLOCATION")
    lines.append(f"  {'Equity (this portfolio)':<30} {equity_pct:.0f}%  {'█' * int(equity_pct/5)}")
    lines.append(f"  {'Debt (FD/bonds/debt funds)':<30} {debt_pct:.0f}%  {'█' * int(debt_pct/5)}")

    # ── Portfolio Metrics ──────────────────────────────────────────────────
    lines.append(f"\n  📈 EQUITY PORTFOLIO METRICS")
    lines.append(f"  Expected Return (annualised) : {portfolio['expected_return']*100:.1f}%")
    lines.append(f"  Volatility (annualised)      : {portfolio['volatility']*100:.1f}%")
    lines.append(f"  Sharpe Ratio                 : {portfolio['sharpe']:.2f}")
    lines.append(f"  Estimated Max Drawdown       : {portfolio['max_drawdown_est']*100:.1f}%")
    lines.append(f"  Drawdown Tolerance           : {profile['max_drawdown_tolerance']*100:.0f}%")

    # ── Corpus Projection ─────────────────────────────────────────────────
    proj = projection
    lines.append(f"\n  🎯 CORPUS PROJECTION ({goal_config.investment_horizon_years} years)")
    lines.append(f"  {'─'*55}")
    lines.append(f"  Total Invested               : ₹{proj['total_invested']:>14,.0f}")
    lines.append(f"  Expected Corpus (median)     : ₹{proj['total_corpus']:>14,.0f}")
    if profile["inflation_adjustment"]:
        lines.append(f"  Real Value (inflation adj.)  : ₹{proj['real_corpus']:>14,.0f}")
    lines.append(f"  Total Gain                   : ₹{proj['total_gain']:>14,.0f}")
    lines.append(f"  Wealth Multiple              : {proj['wealth_multiple']:.2f}x")
    lines.append(f"  CAGR                         : {proj['cagr']:.1f}%")
    lines.append(f"  {'─'*55}")
    lines.append(f"  CONFIDENCE RANGE (equity portion):")
    lines.append(f"  Pessimistic (10th pct)       : ₹{proj['p10']:>14,.0f}")
    lines.append(f"  Conservative (25th pct)      : ₹{proj['p25']:>14,.0f}")
    lines.append(f"  Base case (50th pct)         : ₹{proj['p50']:>14,.0f}")
    lines.append(f"  Optimistic (75th pct)        : ₹{proj['p75']:>14,.0f}")
    lines.append(f"  Bull case (90th pct)         : ₹{proj['p90']:>14,.0f}")

    # ── Target corpus gap analysis ─────────────────────────────────────────
    if goal_config.target_corpus:
        gap = goal_config.target_corpus - proj["total_corpus"]
        lines.append(f"\n  🎯 TARGET CORPUS ANALYSIS")
        lines.append(f"  Your Target                  : ₹{goal_config.target_corpus:>14,.0f}")
        lines.append(f"  Projected Corpus             : ₹{proj['total_corpus']:>14,.0f}")
        if gap > 0:
            lines.append(f"  Shortfall                    : ₹{gap:>14,.0f}  ⚠")
            lines.append(f"  → Increase SIP or lump sum to bridge gap")
        else:
            lines.append(f"  Surplus                      : ₹{abs(gap):>14,.0f}  ✓")
            lines.append(f"  → You are on track to exceed your target!")

    # ── Year-by-year milestones ───────────────────────────────────────────
    lines.append(f"\n  📅 YEAR-BY-YEAR MILESTONES")
    lines.append(f"  {'Year':<6} {'Corpus':>14} {'Real Value':>14} {'Gain':>14}")
    lines.append(f"  {'─'*52}")
    yearly = proj["yearly_projection"]
    # Show key milestones: yr 1, 3, 5, then every 5 years, then final
    milestone_years = {1, 3, 5}
    milestone_years.update(range(5, goal_config.investment_horizon_years + 1, 5))
    milestone_years.add(goal_config.investment_horizon_years)
    for row in yearly:
        if row["year"] in milestone_years:
            lines.append(
                f"  {row['year']:<6} ₹{row['corpus']:>12,.0f} "
                f"₹{row['real_value']:>12,.0f} "
                f"₹{row['gain']:>12,.0f}"
            )

    # ── Goal-specific notes ────────────────────────────────────────────────
    lines.append(f"\n  💡 GOAL-SPECIFIC GUIDANCE")
    for note in profile["notes"]:
        lines.append(f"  • {note}")

    lines.append(f"\n  ⚠  This is not financial advice. Consult a SEBI-registered advisor.")
    lines.append("─" * 64)

    report = "\n".join(lines)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_goal_inputs(goal_type: str, horizon: int, amount: float, risk: str) -> list[str]:
    """Returns list of warnings for the user (not errors — just guidance)."""
    warnings_list = []
    profile = GOAL_PROFILES.get(goal_type, {})

    if profile:
        min_h, max_h = profile.get("typical_horizon", (1, 40))
        if horizon < min_h:
            warnings_list.append(
                f"Typical horizon for {goal_type} is {min_h}-{max_h} years. "
                f"Your {horizon} year horizon is short — returns may be insufficient."
            )
        if horizon > max_h:
            warnings_list.append(
                f"Horizon of {horizon} years is unusually long for {goal_type}. "
                f"Consider reassessing your goal timeline."
            )

    if goal_type in ("retirement", "fire") and horizon < 7:
        warnings_list.append(
            "Retirement/FIRE goals need at least 7-10 years to compound meaningfully."
        )

    if goal_type in ("marriage", "education") and risk == "High":
        warnings_list.append(
            f"High risk with fixed-date goal ({goal_type}) is dangerous. "
            f"Risk will be automatically reduced to protect your corpus near target date."
        )

    if amount < 10_000:
        warnings_list.append(
            "Investment amount below ₹10,000 may not achieve meaningful diversification."
        )

    return warnings_list
