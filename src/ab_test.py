import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
import os


# Sample Size Calculator

def required_sample_size(baseline_rate: float,
                         expected_lift: float,
                         alpha: float = 0.05,
                         power: float = 0.80) -> int:
   
    p1 = baseline_rate                  # Control group churn rate
    p2 = baseline_rate - expected_lift  # Treatment group expected churn rate

    z_alpha = stats.norm.ppf(1 - alpha)        # One-tailed z for alpha
    z_beta = stats.norm.ppf(power)             # z for desired power

    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p1 - p2) ** 2

    n = int(np.ceil(numerator / denominator))

    print(f"\n[SampleSize] Baseline churn rate:     {p1:.2%}")
    print(f"[SampleSize] Expected treatment rate: {p2:.2%} (lift = {expected_lift:.2%})")
    print(f"[SampleSize] Required n per group:    {n:,}")
    print(f"[SampleSize] Total customers needed:  {n * 2:,}")
    print(f"[SampleSize] Power: {power:.0%} | Alpha: {alpha:.2f}")

    return n


# ─────────────────────────────────────────────────────────────────────────────
# A/B Test Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ab_test(df: pd.DataFrame,
                     model_predictions: np.ndarray,
                     churn_threshold: float = 0.5,
                     discount_effect: float = 0.08,
                     sample_fraction: float = 0.3,
                     alpha: float = 0.05,
                     random_state: int = 42) -> dict:
   
    rng = np.random.RandomState(random_state)

    # Identify at-risk customers
    at_risk_mask = model_predictions >= churn_threshold
    at_risk_indices = np.where(at_risk_mask)[0]
    n_at_risk = len(at_risk_indices)

    print(f"\n[ABTest] Total customers: {len(df):,}")
    print(f"[ABTest] At-risk customers (prob >= {churn_threshold}): {n_at_risk:,} "
          f"({n_at_risk/len(df):.1%})")

    # Sample subset for the A/B test
    n_sample = int(n_at_risk * sample_fraction)
    sampled_indices = rng.choice(at_risk_indices, size=n_sample, replace=False)

    # Split into control (50%) and treatment (50%)
    n_per_group = n_sample // 2
    control_indices = sampled_indices[:n_per_group]
    treatment_indices = sampled_indices[n_per_group:n_per_group * 2]

    # Get baseline churn probabilities
    control_probs = model_predictions[control_indices].copy()
    treatment_probs = model_predictions[treatment_indices].copy()

    # Apply discount effect to treatment group
    # Discount reduces churn probability by discount_effect (clipped at 0)
    treatment_probs_with_discount = np.clip(
        treatment_probs - discount_effect, 0, 1
    )

    # Simulate actual churn outcomes (Bernoulli trial)
    control_churn = rng.binomial(1, control_probs)
    treatment_churn = rng.binomial(1, treatment_probs_with_discount)

    # Compute observed churn rates
    control_rate = control_churn.mean()
    treatment_rate = treatment_churn.mean()
    absolute_lift = control_rate - treatment_rate
    relative_lift = absolute_lift / control_rate if control_rate > 0 else 0

    print(f"\n[ABTest] Group sizes: Control={len(control_churn):,} | "
          f"Treatment={len(treatment_churn):,}")
    print(f"[ABTest] Control churn rate:   {control_rate:.4f} ({control_rate:.2%})")
    print(f"[ABTest] Treatment churn rate: {treatment_rate:.4f} ({treatment_rate:.2%})")
    print(f"[ABTest] Absolute reduction:   {absolute_lift:.4f} ({absolute_lift:.2%})")
    print(f"[ABTest] Relative reduction:   {relative_lift:.2%}")

    # ── Z-Test for Two Proportions ──────────────────────────────────────────
    # H0: p_control == p_treatment (no effect)
    # H1: p_control > p_treatment (discount reduces churn) — one-tailed

    n_control = len(control_churn)
    n_treatment = len(treatment_churn)

    p_control = control_rate
    p_treatment = treatment_rate

    # Pooled proportion under H0
    p_pooled = (control_churn.sum() + treatment_churn.sum()) / (n_control + n_treatment)

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_control + 1 / n_treatment))

    # Z statistic (positive z means control rate > treatment rate)
    if se == 0:
        z_stat = 0.0
    else:
        z_stat = (p_control - p_treatment) / se

    # One-tailed p-value (testing if treatment < control)
    p_value = 1 - stats.norm.cdf(z_stat)

    # Confidence interval for the difference in proportions (95%)
    z_95 = stats.norm.ppf(0.975)
    se_diff = np.sqrt(
        p_control * (1 - p_control) / n_control +
        p_treatment * (1 - p_treatment) / n_treatment
    )
    ci_lower = absolute_lift - z_95 * se_diff
    ci_upper = absolute_lift + z_95 * se_diff

    # Decision
    reject_h0 = p_value < alpha
    conclusion = (
        "REJECT H0 — The discount significantly reduces churn rate."
        if reject_h0
        else "FAIL TO REJECT H0 — No statistically significant effect detected."
    )

    print(f"\n[ABTest] ── Z-Test Results ──────────────────────────────")
    print(f"[ABTest] Z-statistic:          {z_stat:.4f}")
    print(f"[ABTest] P-value (one-tailed): {p_value:.6f}")
    print(f"[ABTest] Significance level:   {alpha}")
    print(f"[ABTest] 95% CI for diff:      [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"[ABTest] Decision:             {conclusion}")

    results = {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "control_churn_rate": control_rate,
        "treatment_churn_rate": treatment_rate,
        "absolute_reduction": absolute_lift,
        "relative_reduction": relative_lift,
        "z_statistic": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
        "reject_h0": reject_h0,
        "conclusion": conclusion,
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_ab_results(results: dict, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Plot 1: Churn Rate Comparison ───────────────────────────────────────
    ax = axes[0]
    groups = ["Control\n(No Discount)", "Treatment\n(With Discount)"]
    rates = [results["control_churn_rate"], results["treatment_churn_rate"]]
    colors = ["#D85A30", "#2E75B6"]
    bars = ax.bar(groups, rates, color=colors, width=0.4, alpha=0.85)
    ax.set_ylim(0, max(rates) * 1.4)
    ax.set_ylabel("Churn Rate", fontsize=12)
    ax.set_title("A/B Test: Churn Rate Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{rate:.2%}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    # Annotate reduction
    reduction = results["absolute_reduction"]
    p_val = results["p_value"]
    sig_label = "p < 0.05 (Significant)" if results["reject_h0"] else "p >= 0.05 (Not Significant)"
    ax.text(0.5, max(rates) * 1.25,
            f"Reduction: {reduction:.2%}\n{sig_label}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="#1F4E79",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#D6E4F0", alpha=0.8))

    # ── Plot 2: Z Distribution ───────────────────────────────────────────────
    ax2 = axes[1]
    x = np.linspace(-4, 4, 400)
    y = stats.norm.pdf(x)
    ax2.plot(x, y, color="#1F4E79", lw=2)

    # Shade rejection region (one-tailed, right tail for z > z_critical)
    z_critical = stats.norm.ppf(1 - results["alpha"])
    x_reject = np.linspace(z_critical, 4, 200)
    ax2.fill_between(x_reject, stats.norm.pdf(x_reject),
                     alpha=0.3, color="#D85A30", label=f"Rejection region (α={results['alpha']})")

    # Plot observed Z
    z_obs = results["z_statistic"]
    ax2.axvline(z_obs, color="#2E75B6", lw=2, linestyle="--",
                label=f"Observed Z = {z_obs:.3f}")
    ax2.axvline(z_critical, color="#D85A30", lw=1.5, linestyle=":",
                label=f"Z critical = {z_critical:.3f}")

    ax2.set_xlabel("Z statistic", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_title("Z-Test Distribution\n(H0: No difference in churn rates)",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "ab_test_results.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] A/B test results saved: {fname}")


def plot_churn_distribution(model_predictions: np.ndarray,
                            output_dir: str = "outputs"):
    """Plot the distribution of predicted churn probabilities."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(model_predictions, bins=40, color="#2E75B6", alpha=0.75, edgecolor="white")
    ax.axvline(0.5, color="#D85A30", lw=2, linestyle="--", label="Threshold = 0.5")
    ax.set_xlabel("Predicted Churn Probability", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Distribution of Predicted Churn Probabilities",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = os.path.join(output_dir, "churn_probability_distribution.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Churn distribution saved: {fname}")
