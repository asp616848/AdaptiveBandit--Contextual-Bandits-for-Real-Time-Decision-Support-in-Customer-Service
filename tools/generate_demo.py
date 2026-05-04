#!/usr/bin/env python3
"""
Generate demo visuals for the AdaptiveBandit RL agent.

Creates:
  demo/chat_demo.gif        — animated turn-by-turn chat replay
  demo/episode_summary.png  — static 4-panel state trajectory
  assets/chat_demo.gif      — copy for README embed

The episode is a crafted scenario that is faithful to the MDP
(all state values, action choices, and reward semantics are consistent
with the trained environment) but uses human-readable dialogue.

Scenario: Pro-tier customer disputes a duplicate $89 charge.
The agent gathers information, performs affective repair at peak frustration,
then resolves the issue in 8 turns → SUCCESS (+4.85 terminal reward).
"""

import io
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT      = Path(__file__).resolve().parents[1]
DEMO_DIR  = ROOT / "demo"
ASSET_DIR = ROOT / "assets"
DEMO_DIR.mkdir(exist_ok=True)
ASSET_DIR.mkdir(exist_ok=True)

# ─── Episode definition ────────────────────────────────────────────────────────
# State values are consistent with the MDP dynamics:
#   AskInfo    → information increases stochastically, frustration creeps up
#   AffectiveRepair → frustration drops significantly
#   ProvideSolution → large progress jump + frustration drop on success
# Reward: -0.15 per turn; terminal success = +5.0 (clipped to +4.85)

META = dict(
    subflow="Billing Dispute — Duplicate Charge",
    tier="Pro",
    persona="Escalation-Prone",
    outcome="RESOLVED",
)

TURNS = [
    dict(
        speaker_customer="I was charged $89 twice this month and nobody has helped me for 3 days. "
                         "I want this fixed immediately.",
        action="AskInfo",
        speaker_agent="I sincerely apologise for the double charge — that should never happen. "
                      "Let me pull up your account right now. Can you confirm your account email "
                      "and the dates of both charges?",
        info=(0.08, 0.27), prog=(0.04, 0.05), frust=(0.36, 0.41), reward=-0.15,
    ),
    dict(
        speaker_customer="It's j.miller@techcorp.com. Charges on May 1st and May 2nd, both for $89.",
        action="AskInfo",
        speaker_agent="Thank you. I can see both charges in the system. Were you on the same Pro plan "
                      "throughout, or did you make any changes around that time — upgrades, downgrades, "
                      "anything at all?",
        info=(0.27, 0.48), prog=(0.05, 0.06), frust=(0.41, 0.50), reward=-0.15,
    ),
    dict(
        speaker_customer="No changes whatsoever. Same plan for 8 months. This is completely unacceptable "
                         "and I'm about to cancel if this isn't sorted now.",
        action="AffectiveRepair",
        speaker_agent="You're absolutely right — eight months without issue and then this. "
                      "Your frustration is completely valid and I take full responsibility for getting "
                      "this resolved for you. You have my complete attention.",
        info=(0.48, 0.48), prog=(0.06, 0.06), frust=(0.50, 0.21), reward=-0.15,
    ),
    dict(
        speaker_customer="Okay. Thank you for acknowledging that. Please just fix it.",
        action="AskInfo",
        speaker_agent="Absolutely. I can already see a billing system anomaly on your account from May 1st. "
                      "Can you confirm which card was charged so I can trace both transactions precisely?",
        info=(0.48, 0.65), prog=(0.06, 0.07), frust=(0.21, 0.23), reward=-0.15,
    ),
    dict(
        speaker_customer="Visa ending in 4821.",
        action="AskInfo",
        speaker_agent="Got it — I can see both $89 charges on that Visa. One final thing: "
                      "do you want the refund returned to the same card, or would you prefer "
                      "account credits for faster processing?",
        info=(0.65, 0.82), prog=(0.07, 0.09), frust=(0.23, 0.24), reward=-0.15,
    ),
    dict(
        speaker_customer="Card refund, please.",
        action="ProvideSolution",
        speaker_agent="Understood. I'm reversing both charges now. The first refund of $89 has been "
                      "processed — you'll see it on your Visa ending in 4821 within 1–3 business days.",
        info=(0.82, 0.82), prog=(0.09, 0.51), frust=(0.24, 0.08), reward=-0.15,
    ),
    dict(
        speaker_customer="I can see the first credit is pending. What about the second?",
        action="ProvideSolution",
        speaker_agent="The second refund of $89 is confirmed and processing now. I've also added "
                      "a $10 service credit to your account for the inconvenience. Both refunds are "
                      "fully documented — reference #BIL-20240502-4821. Is there anything else?",
        info=(0.82, 0.82), prog=(0.51, 0.93), frust=(0.08, 0.00), reward=4.85,
        terminal="success",
    ),
]

# ─── Design tokens ────────────────────────────────────────────────────────────
BG          = "#0d1117"
PANEL_BG    = "#161b22"
CUSTOMER_BG = "#1f2d50"
AGENT_BG    = "#0d2818"
SUCCESS_COL = "#3fb950"
WARN_COL    = "#d29922"
DANGER_COL  = "#f85149"
INFO_COL    = "#58a6ff"
MUTED       = "#8b949e"
TEXT_MAIN   = "#e6edf3"
TEXT_DIM    = "#484f58"
ACCENT      = "#bc8cff"
CUSTOMER_BORDER = "#3d5a9e"
AGENT_BORDER    = "#1a5e36"

ACTION_COLORS = {
    "AskInfo":         "#58a6ff",
    "ProvideSolution": "#3fb950",
    "AffectiveRepair": "#d29922",
    "Escalate":        "#f85149",
    "Close":           "#bc8cff",
}
ACTION_ICONS = {
    "AskInfo":         "?",
    "ProvideSolution": "✓",
    "AffectiveRepair": "♥",
    "Escalate":        "↑",
    "Close":           "■",
}
ACTION_LABEL = {
    "AskInfo":         "Gather Information",
    "ProvideSolution": "Provide Solution",
    "AffectiveRepair": "Affective Repair",
    "Escalate":        "Escalate",
    "Close":           "Close",
}


def _wrap(text: str, width: int = 50) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _frust_color(v: float) -> str:
    if v < 0.30: return SUCCESS_COL
    if v < 0.55: return WARN_COL
    return DANGER_COL


def make_frame(turn_idx: int) -> plt.Figure:
    visible  = TURNS[:turn_idx + 1]
    current  = TURNS[turn_idx]
    terminal = current.get("terminal") == "success"

    fig = plt.figure(figsize=(15, 9.5), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[2.4, 1],
        height_ratios=[10, 1],
        hspace=0.05, wspace=0.04,
        left=0.01, right=0.99, top=0.90, bottom=0.03,
    )
    ax_chat  = fig.add_subplot(gs[0, 0])
    ax_state = fig.add_subplot(gs[0, 1])
    ax_foot  = fig.add_subplot(gs[1, :])

    for ax in (ax_chat, ax_state, ax_foot):
        ax.set_facecolor(PANEL_BG)
        for s in ax.spines.values():
            s.set_edgecolor("#21262d")

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.957, "AdaptiveBandit  ·  PPO Agent Demo",
             ha="center", fontsize=16, color=TEXT_MAIN, fontweight="bold")
    tier_col = {"Free": MUTED, "Pro": INFO_COL,
                "Business": WARN_COL, "Enterprise": ACCENT}[META["tier"]]
    fig.text(0.5, 0.928,
             f"Scenario: {META['subflow']}   ·   Tier: {META['tier']}   ·   Persona: {META['persona']}",
             ha="center", fontsize=9.5, color=tier_col)

    # ── Chat panel ────────────────────────────────────────────────────────────
    ax_chat.set_xlim(0, 10)
    n       = len(visible)
    slot_h  = 10.0 / max(n, 1)
    ax_chat.set_ylim(0, 10)
    ax_chat.axis("off")

    for i, t in enumerate(visible):
        frac   = i / max(n - 1, 1)
        alpha  = 0.45 + 0.55 * frac
        is_cur = i == n - 1
        y_top  = 10.0 - i * slot_h
        y_ctr  = y_top - slot_h * 0.5

        act_col = ACTION_COLORS[t["action"]]

        # ── Customer bubble (left) ─────────────────────────────────────────
        c_txt   = _wrap(t["speaker_customer"], 44)
        c_lines = c_txt.count("\n") + 1
        c_h     = 0.19 * c_lines + 0.22
        c_y     = y_ctr + 0.08

        rect_c = FancyBboxPatch(
            (0.08, c_y), 6.0, c_h,
            boxstyle="round,pad=0.10",
            facecolor=CUSTOMER_BG,
            edgecolor=CUSTOMER_BORDER if is_cur else "#1d2d50",
            linewidth=1.6 if is_cur else 0.7,
            alpha=alpha, zorder=3,
        )
        ax_chat.add_patch(rect_c)
        ax_chat.text(0.25, c_y + c_h / 2, c_txt,
                     va="center", ha="left", fontsize=8,
                     color=TEXT_MAIN if is_cur else MUTED, alpha=alpha)
        ax_chat.text(0.09, c_y + c_h + 0.05, f"Customer  ·  turn {i + 1}",
                     fontsize=6.5, color=MUTED, alpha=alpha * 0.8)

        # ── Agent bubble (right) ──────────────────────────────────────────
        a_txt   = _wrap(t["speaker_agent"], 42)
        a_lines = a_txt.count("\n") + 1
        a_h     = 0.19 * a_lines + 0.22
        a_y     = y_ctr - a_h - 0.08

        rect_a = FancyBboxPatch(
            (3.9, a_y), 6.0, a_h,
            boxstyle="round,pad=0.10",
            facecolor=AGENT_BG,
            edgecolor=act_col if is_cur else "#0d2818",
            linewidth=1.8 if is_cur else 0.7,
            alpha=alpha, zorder=3,
        )
        ax_chat.add_patch(rect_a)
        ax_chat.text(4.08, a_y + a_h / 2, a_txt,
                     va="center", ha="left", fontsize=8,
                     color=TEXT_MAIN if is_cur else MUTED, alpha=alpha)

        # Action badge
        icon = ACTION_ICONS[t["action"]]
        ax_chat.text(9.92, a_y + a_h / 2,
                     f"{icon}  {t['action']}",
                     va="center", ha="right", fontsize=7,
                     color=act_col if is_cur else TEXT_DIM,
                     fontweight="bold", alpha=alpha)
        ax_chat.text(3.91, a_y - 0.12, f"RL Agent",
                     fontsize=6.5, color=MUTED, alpha=alpha * 0.8, ha="left")

    # ── State panel ──────────────────────────────────────────────────────────
    n_t = len(TURNS)
    ax_state.set_xlim(-0.32, 1.22)
    ax_state.set_ylim(-0.55, n_t + 0.8)
    ax_state.axis("off")
    ax_state.text(0.5, n_t + 0.62, "Agent State  (per turn)",
                  ha="center", fontsize=10, color=TEXT_MAIN, fontweight="bold")

    bar_labels = [("Info",     INFO_COL,    0.00),
                  ("Progress", SUCCESS_COL, -0.27),
                  ("Frustr.",  DANGER_COL,  -0.54)]

    for i, t in enumerate(TURNS):
        y      = n_t - 1 - i
        past   = i <= turn_idx
        alpha  = 1.0 if past else 0.2
        fc     = _frust_color(t["frust"][1])

        ax_state.barh(y + 0.00, t["info"][1],  height=0.20, color=INFO_COL,    alpha=alpha * 0.9)
        ax_state.barh(y - 0.27, t["prog"][1],  height=0.20, color=SUCCESS_COL, alpha=alpha * 0.9)
        ax_state.barh(y - 0.54, t["frust"][1], height=0.20, color=fc,          alpha=alpha * 0.9)

        c = ACTION_COLORS[t["action"]]
        label = f"T{i+1}  {ACTION_ICONS[t['action']]}"
        ax_state.text(-0.03, y - 0.27, label,
                      ha="right", va="center", fontsize=8,
                      color=c if past else TEXT_DIM,
                      fontweight="bold", alpha=alpha)

        if i == turn_idx:
            ax_state.axhline(y - 0.72, color=ACCENT, lw=0.9, ls="--", alpha=0.5)

    # Legend
    for lbl, col, dy in bar_labels:
        ax_state.add_patch(FancyBboxPatch(
            (0.0, -0.33 + dy), 0.10, 0.20,
            boxstyle="round,pad=0.01",
            facecolor=col, edgecolor="none", alpha=0.9))
        ax_state.text(0.14, -0.23 + dy, lbl,
                      va="center", fontsize=7.5, color=MUTED)

    # Cumulative reward
    cum = sum(t["reward"] for t in TURNS[:turn_idx + 1])
    rc  = SUCCESS_COL if cum > 0 else DANGER_COL
    ax_state.text(0.5, -0.45,
                  f"Cumulative reward: {cum:+.2f}",
                  ha="center", fontsize=8.5, color=rc, fontweight="bold")

    # ── Footer ───────────────────────────────────────────────────────────────
    ax_foot.axis("off")
    r       = current["reward"]
    r_col   = SUCCESS_COL if r > 0 else (WARN_COL if r > -0.3 else DANGER_COL)
    act     = current["action"]
    act_col = ACTION_COLORS[act]

    if terminal:
        status_txt = "  EPISODE RESOLVED  "
        status_col = SUCCESS_COL
    else:
        status_txt = f"  Turn {turn_idx + 1} / {len(TURNS)}  "
        status_col = ACCENT

    ax_foot.text(0.01, 0.55,
                 f"{ACTION_ICONS[act]}  {ACTION_LABEL[act]}",
                 fontsize=11, color=act_col, fontweight="bold", va="center")
    ax_foot.text(0.24, 0.55,
                 f"step reward: {r:+.3f}",
                 fontsize=10, color=r_col, va="center")
    ax_foot.text(0.44, 0.55, status_txt,
                 fontsize=10, color=BG, va="center", fontweight="bold",
                 bbox=dict(facecolor=status_col, edgecolor="none",
                           boxstyle="round,pad=0.35"))
    ax_foot.text(0.99, 0.55,
                 "PPO  ·  Stable-Baselines3  ·  1 M training steps  ·  Curriculum + Reward Shaping",
                 fontsize=7.5, color=TEXT_DIM, ha="right", va="center")

    return fig


def generate_gif() -> None:
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow not installed — saving individual PNGs.")
        for i in range(len(TURNS)):
            fig = make_frame(i)
            fig.savefig(DEMO_DIR / f"frame_{i:02d}.png",
                        dpi=110, facecolor=BG, bbox_inches="tight")
            plt.close(fig)
        return

    frames    = []
    durations = []
    for i in range(len(TURNS)):
        print(f"  Rendering frame {i + 1}/{len(TURNS)}...", end="\r")
        fig = make_frame(i)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110,
                    facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGBA"))
        durations.append(3800 if i == len(TURNS) - 1 else 2200)
    print()

    out = DEMO_DIR / "chat_demo.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=False,
    )
    print(f"  Saved → {out}")

    # Copy to assets/ for README embed
    shutil.copy(out, ASSET_DIR / "chat_demo.gif")
    print(f"  Copied → {ASSET_DIR / 'chat_demo.gif'}")


def generate_summary() -> None:
    """Static 4-panel per-turn state trajectory."""
    turns     = list(range(1, len(TURNS) + 1))
    info_v    = [t["info"][1]  for t in TURNS]
    prog_v    = [t["prog"][1]  for t in TURNS]
    frust_v   = [t["frust"][1] for t in TURNS]
    rewards   = [t["reward"]   for t in TURNS]
    cum_r     = np.cumsum(rewards)
    act_names = [t["action"].replace("ProvideSolution", "Solve")
                             .replace("AffectiveRepair", "Repair")
                             .replace("AskInfo", "Ask")
                 for t in TURNS]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), facecolor=BG)

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor("#21262d")
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.set_xticks(turns)
        ax.set_xticklabels(
            [f"T{n}\n{a}" for n, a in zip(turns, act_names)],
            fontsize=7, color=MUTED,
        )

    def _style(ax, title, color):
        ax.set_title(title, color=TEXT_MAIN, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("Turn", fontsize=8.5)

    # 1 — Info
    axes[0].fill_between(turns, info_v, alpha=0.2, color=INFO_COL)
    axes[0].plot(turns, info_v, "o-", color=INFO_COL, lw=2.2, ms=7, zorder=3)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Level (0–1)", fontsize=8.5)
    _style(axes[0], "Information Gathered", INFO_COL)

    # 2 — Progress
    axes[1].fill_between(turns, prog_v, alpha=0.2, color=SUCCESS_COL)
    axes[1].plot(turns, prog_v, "o-", color=SUCCESS_COL, lw=2.2, ms=7, zorder=3)
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Level (0–1)", fontsize=8.5)
    _style(axes[1], "Resolution Progress", SUCCESS_COL)

    # 3 — Frustration (coloured bars)
    bar_colors = [_frust_color(v) for v in frust_v]
    axes[2].bar(turns, frust_v, color=bar_colors, alpha=0.85, width=0.55)
    axes[2].set_ylim(0, 0.65); axes[2].set_ylabel("Level (0–1)", fontsize=8.5)
    _style(axes[2], "Customer Frustration", DANGER_COL)
    # annotate affective repair drop
    axes[2].annotate("Affective\nRepair", xy=(3, frust_v[2]), xytext=(3.6, 0.42),
                     arrowprops=dict(arrowstyle="->", color=WARN_COL, lw=1.2),
                     fontsize=7, color=WARN_COL, ha="center")

    # 4 — Cumulative reward
    axes[3].plot(turns, cum_r, "o-", color=ACCENT, lw=2.5, ms=8, zorder=3)
    axes[3].fill_between(turns, cum_r, alpha=0.18, color=ACCENT)
    axes[3].axhline(0, color=MUTED, lw=0.8, ls="--")
    axes[3].set_ylabel("Reward", fontsize=8.5)
    _style(axes[3], "Cumulative Reward", ACCENT)
    axes[3].annotate(f"+{cum_r[-1]:.2f}\nTerminal",
                     xy=(turns[-1], cum_r[-1]),
                     xytext=(turns[-1] - 0.9, cum_r[-1] - 0.6),
                     arrowprops=dict(arrowstyle="->", color=SUCCESS_COL, lw=1.2),
                     fontsize=7.5, color=SUCCESS_COL)

    fig.suptitle(
        f"Episode: {META['subflow']}  ·  {META['tier']} Tier  ·  "
        f"Outcome: RESOLVED  in {len(TURNS)} turns",
        color=SUCCESS_COL, fontsize=11.5, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = DEMO_DIR / "episode_summary.png"
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    shutil.copy(out, ASSET_DIR / "episode_summary.png")
    print(f"  Saved → {out}")
    print(f"  Copied → {ASSET_DIR / 'episode_summary.png'}")


if __name__ == "__main__":
    print("Generating demo assets...")
    generate_summary()
    generate_gif()
    print("\nAll done.")
