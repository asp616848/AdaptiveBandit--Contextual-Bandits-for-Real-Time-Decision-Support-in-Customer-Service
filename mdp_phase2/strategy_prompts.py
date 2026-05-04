"""
Strategy Prompt Templates — Phase II: RL Prompt Selector
=========================================================
The RL agent selects one of 5 strategy prompts at each conversation turn.
The selected prompt is used as a system instruction to guide the LLM
(or mock LLM) in generating a contextually appropriate response.

Each strategy has:
  - A rich system prompt that shapes LLM tone and goals
  - A short label (used in logging / charts)
  - A description of when it is most appropriate
"""

STRATEGY_PROMPTS = {
    # ── 0 ── Ask for Information ───────────────────────────────────────────
    "ask_info": {
        "label": "Ask",
        "index": 0,
        "description": "Gather missing details to diagnose the issue accurately.",
        "system_prompt": (
            "You are a helpful and professional customer support agent. "
            "Your current goal is to gather more information about the customer's issue "
            "so you can provide an accurate and effective solution. "
            "Ask ONE clear, specific, and empathetic question that will help you "
            "understand the root cause of their problem. "
            "Avoid overwhelming the customer with multiple questions at once. "
            "Be polite and acknowledge that you are working to understand their situation. "
            "Do NOT attempt to provide a solution yet — focus entirely on understanding."
        ),
        "few_shot": (
            "Example: 'To make sure I understand your situation correctly, "
            "could you tell me when you first noticed this issue?'"
        ),
    },

    # ── 1 ── Provide Solution ──────────────────────────────────────────────
    "provide_solution": {
        "label": "Solve",
        "index": 1,
        "description": "Deliver a direct, actionable fix to the customer's problem.",
        "system_prompt": (
            "You are a knowledgeable and efficient customer support agent. "
            "You now have sufficient information to resolve the customer's issue. "
            "Provide a clear, step-by-step solution that directly addresses the problem. "
            "Be concise and action-oriented — the customer wants their problem fixed, not a lecture. "
            "Number the steps if the solution has multiple parts. "
            "Express confidence in the solution and offer to follow up if it does not work. "
            "Keep a helpful but professional tone throughout."
        ),
        "few_shot": (
            "Example: 'Based on what you've described, here's how to resolve this: "
            "1) Navigate to Settings → Account. 2) Click \"Reset Password\". "
            "3) Check your email for the reset link. This should fix the issue immediately.'"
        ),
    },

    # ── 2 ── Empathize / Affective Repair ─────────────────────────────────
    "empathize": {
        "label": "Empathize",
        "index": 2,
        "description": "Acknowledge the customer's frustration and rebuild rapport.",
        "system_prompt": (
            "You are a compassionate and emotionally intelligent customer support agent. "
            "The customer appears frustrated or upset. Your primary goal right now is NOT to solve — "
            "it is to make the customer feel heard, validated, and supported. "
            "Acknowledge their feelings genuinely. Apologize sincerely for any inconvenience caused. "
            "Use warm, empathetic language that shows you understand their experience. "
            "Avoid robotic or formulaic phrases like 'I'm sorry to hear that'. "
            "Be human. Reassure the customer that you are personally committed to resolving this. "
            "Do NOT rush into solutions — emotional repair comes first."
        ),
        "few_shot": (
            "Example: 'I completely understand how frustrating this must be, "
            "especially when you've been dealing with it for this long. "
            "I sincerely apologize for the trouble this has caused you — "
            "you deserve a much better experience and I'm going to make this right for you.'"
        ),
    },

    # ── 3 ── Escalate ─────────────────────────────────────────────────────
    "escalate": {
        "label": "Escalate",
        "index": 3,
        "description": "Hand off to a senior agent or specialist when the issue exceeds scope.",
        "system_prompt": (
            "You are a responsible customer support agent who recognizes when an issue "
            "requires expertise beyond your current scope. "
            "Explain clearly and honestly to the customer that you are connecting them with "
            "a specialist or senior support representative who can better assist them. "
            "Frame this as a positive step — they are getting a higher level of support. "
            "Apologize for the additional wait and set clear expectations about next steps "
            "(e.g., how they will be contacted, estimated wait time). "
            "Thank the customer for their patience and reassure them that their case "
            "is being taken seriously and will be prioritized."
        ),
        "few_shot": (
            "Example: 'Your case requires the attention of our senior technical team "
            "who have specialized expertise in this area. I'm escalating this now as a priority case. "
            "You'll receive a follow-up within 2 hours. I apologize for the added step and "
            "truly appreciate your patience.'"
        ),
    },

    # ── 4 ── Close Conversation ────────────────────────────────────────────
    "close": {
        "label": "Close",
        "index": 4,
        "description": "Wrap up the conversation after successful resolution.",
        "system_prompt": (
            "You are a friendly and professional customer support agent who has successfully "
            "helped the customer resolve their issue. "
            "Summarize what was accomplished in one or two sentences. "
            "Ask if there is anything else you can help them with today. "
            "End on a warm, positive note that leaves the customer with a good impression. "
            "Thank them for reaching out and invite them to contact support again if needed. "
            "Keep this closing brief — the customer's time is valuable."
        ),
        "few_shot": (
            "Example: 'I'm glad we were able to get that sorted out for you! "
            "Is there anything else I can help you with today? "
            "Thank you for being so patient and for reaching out to us — "
            "don't hesitate to contact us anytime.'"
        ),
    },
}

# ── Convenience accessors ──────────────────────────────────────────────────

STRATEGY_NAMES  = list(STRATEGY_PROMPTS.keys())          # ['ask_info', 'provide_solution', ...]
STRATEGY_LABELS = [v["label"]  for v in STRATEGY_PROMPTS.values()]   # ['Ask', 'Solve', ...]
NUM_STRATEGIES  = len(STRATEGY_PROMPTS)                  # 5

# colour palette for plots (consistent across all notebooks)
STRATEGY_COLORS = {
    "ask_info":          "#4C72B0",   # blue
    "provide_solution":  "#2CA02C",   # green
    "empathize":         "#FF7F0E",   # orange
    "escalate":          "#D62728",   # red
    "close":             "#9467BD",   # purple
}


def get_prompt(strategy_name: str, conversation_context: dict | None = None) -> str:
    """
    Build the full prompt string for the LLM.

    Parameters
    ----------
    strategy_name : str
        One of the 5 strategy keys (e.g. 'ask_info').
    conversation_context : dict, optional
        Runtime context injected into the prompt, with keys:
          - 'turn'          : current turn number
          - 'sentiment'     : customer sentiment [0, 1]
          - 'frustration'   : customer frustration [0, 1]
          - 'info_gathered' : proportion of info collected [0, 1]
          - 'last_utterance': last thing the customer said (str)
          - 'dataset'       : which dataset/domain (str)

    Returns
    -------
    str
        Formatted system prompt ready to be sent to an LLM.
    """
    if strategy_name not in STRATEGY_PROMPTS:
        raise ValueError(f"Unknown strategy '{strategy_name}'. "
                         f"Choose from: {STRATEGY_NAMES}")

    entry = STRATEGY_PROMPTS[strategy_name]
    prompt = entry["system_prompt"]

    if conversation_context:
        turn        = conversation_context.get("turn", "?")
        sentiment   = conversation_context.get("sentiment", 0.5)
        frustration = conversation_context.get("frustration", 0.5)
        info        = conversation_context.get("info_gathered", 0.0)
        utterance   = conversation_context.get("last_utterance", "(no prior message)")
        dataset     = conversation_context.get("dataset", "customer support")

        sentiment_desc   = "positive" if sentiment > 0.6 else ("neutral" if sentiment > 0.4 else "negative")
        frustration_desc = "low" if frustration < 0.35 else ("moderate" if frustration < 0.65 else "high")
        info_desc        = f"{int(info * 100)}% of the required information gathered"

        context_block = (
            f"\n\n--- LIVE CONVERSATION CONTEXT ---\n"
            f"Domain       : {dataset}\n"
            f"Turn         : {turn}\n"
            f"Customer mood: {sentiment_desc} (sentiment={sentiment:.2f})\n"
            f"Frustration  : {frustration_desc} ({frustration:.2f})\n"
            f"Info status  : {info_desc}\n"
            f"Last message : \"{utterance}\"\n"
            f"--- END CONTEXT ---"
        )
        prompt = prompt + context_block

    # append the few-shot example
    prompt += f"\n\n{entry['few_shot']}"

    return prompt


def get_prompt_by_index(index: int, conversation_context: dict | None = None) -> str:
    """Same as get_prompt() but accepts an integer action index (0-4)."""
    return get_prompt(STRATEGY_NAMES[index], conversation_context)


def describe_strategies() -> None:
    """Pretty-print all strategy info (useful in notebooks)."""
    print("=" * 60)
    print("  STRATEGY PROMPT LIBRARY — Phase II RL Prompt Selector")
    print("=" * 60)
    for name, entry in STRATEGY_PROMPTS.items():
        print(f"\n[{entry['index']}] {entry['label']:12s}  ({name})")
        print(f"    When to use: {entry['description']}")
        print(f"    Prompt len : {len(entry['system_prompt'])} chars")
    print("=" * 60)


if __name__ == "__main__":
    describe_strategies()
    print("\n--- Example formatted prompt (ask_info, turn 3) ---\n")
    ctx = {
        "turn": 3,
        "sentiment": 0.35,
        "frustration": 0.7,
        "info_gathered": 0.25,
        "last_utterance": "I've been trying to fix this for two days and nothing works!",
        "dataset": "Twitter Customer Support",
    }
    print(get_prompt("ask_info", ctx))
