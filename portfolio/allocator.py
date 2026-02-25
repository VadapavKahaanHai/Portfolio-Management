# portfolio/allocator.py — Step 10: Convert weights to buyable quantities (₹)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def allocate_to_shares(
    weights: pd.Series,
    close: pd.DataFrame,
    amount: float,
    tol: float = 0.02,
) -> pd.DataFrame:
    """
    Convert portfolio weights to integer share quantities given ₹amount.

    Algorithm:
    1. Compute ideal rupee allocation per stock
    2. Greedy floor rounding
    3. Repair pass: add 1 share to stocks with highest rounding loss
       until budget is exhausted or further addition would exceed budget.

    Returns DataFrame with columns:
        stock | weight_target | shares | price | invested | weight_actual | deviation
    """
    # Use latest available prices
    prices = close.iloc[-1][weights.index]

    # Ideal allocation
    ideal_rupees = weights * amount
    ideal_shares = ideal_rupees / prices

    # Floor rounding
    shares = ideal_shares.apply(np.floor).astype(int)
    shares = shares.clip(lower=0)  # no negative shares

    # Residuals (fractional shares lost)
    residuals = ideal_shares - shares
    spent     = (shares * prices).sum()

    # Repair: greedy fill with remaining budget
    remaining = amount - spent
    order = residuals.sort_values(ascending=False).index

    for stock in order:
        if remaining < prices[stock]:
            break
        shares[stock] += 1
        remaining -= prices[stock]

    # Remove stocks with 0 shares
    shares = shares[shares > 0]

    # Build output table
    invested     = shares * prices[shares.index]
    total_spent  = invested.sum()
    weight_actual = invested / total_spent

    result = pd.DataFrame({
        "weight_target": weights.reindex(shares.index),
        "shares":        shares,
        "price_inr":     prices[shares.index].round(2),
        "invested_inr":  invested.round(2),
        "weight_actual": weight_actual.round(4),
    })
    result["deviation"] = (result["weight_actual"] - result["weight_target"]).abs().round(4)
    result = result.sort_values("invested_inr", ascending=False)

    tracking_error = result["deviation"].mean()
    print(f"\n  ─── Allocation Summary ───────────────────────────────")
    print(f"  Budget          : ₹{amount:>12,.0f}")
    print(f"  Invested        : ₹{total_spent:>12,.0f}")
    print(f"  Cash remaining  : ₹{amount - total_spent:>12,.0f}  ({(amount-total_spent)/amount*100:.1f}%)")
    print(f"  Stocks          : {len(result)}")
    print(f"  Avg weight error: {tracking_error*100:.2f}%")
    print(f"  ─────────────────────────────────────────────────────")
    print(result[["shares", "price_inr", "invested_inr", "weight_target", "weight_actual"]].to_string())

    return result


def format_allocation_report(
    allocation: pd.DataFrame,
    portfolio: dict,
    user_risk: str,
    amount: float,
) -> str:
    """Generate a human-readable allocation report."""
    lines = []
    lines.append("=" * 62)
    lines.append(f"  PORTFOLIO ALLOCATION REPORT")
    lines.append(f"  Risk Profile  : {user_risk}")
    lines.append(f"  Investment    : ₹{amount:,.0f}")
    lines.append("=" * 62)
    lines.append(f"  Expected Return  (1Y) : {portfolio['expected_return']*100:.1f}%")
    lines.append(f"  Volatility      (1Y) : {portfolio['volatility']*100:.1f}%")
    lines.append(f"  Sharpe Ratio         : {portfolio['sharpe']:.2f}")
    lines.append(f"  Est. Max Drawdown    : {portfolio['max_drawdown_est']*100:.1f}%")
    lines.append("─" * 62)
    lines.append(f"  {'Stock':<15} {'Qty':>5} {'Price':>10} {'₹ Invested':>12} {'Wt%':>6}")
    lines.append("─" * 62)

    total_invested = allocation["invested_inr"].sum()
    for stock, row in allocation.iterrows():
        ticker = stock.replace(".NS", "")
        lines.append(
            f"  {ticker:<15} {int(row['shares']):>5} "
            f"₹{row['price_inr']:>9,.1f} "
            f"₹{row['invested_inr']:>11,.0f} "
            f"{row['weight_actual']*100:>5.1f}%"
        )

    lines.append("─" * 62)
    lines.append(f"  {'TOTAL':<15} {' ':>5} {' ':>10} ₹{total_invested:>11,.0f} {'100.0%':>6}")
    lines.append("=" * 62)

    # Risk contribution
    rc = portfolio.get("risk_contrib")
    if rc is not None:
        lines.append("\n  TOP RISK CONTRIBUTORS:")
        for stock, contrib in rc.nlargest(5).items():
            lines.append(f"    {stock.replace('.NS',''):<15} {contrib*100:.1f}% of portfolio risk")

    report = "\n".join(lines)
    print(report)
    return report
