"""
Mock LLM — Phase II RL Prompt Selector
=======================================
During *training* (thousands of episodes) we can't call a real LLM every step —
it would be prohibitively slow and expensive.

This module provides a structured mock that:
  1. Receives the formatted strategy prompt + conversation context.
  2. Generates a plausible, strategy-consistent response.
  3. Returns the response as a string (same interface a real LLM would use).

The mock is also used to demonstrate the full pipeline in notebooks so reviewers
can see: State → Agent → Prompt → LLM Response → New State without needing an API key.

For production / evaluation, swap MockLLM with OpenAILLM (stub provided at bottom).
"""

from __future__ import annotations
import random
import re
from strategy_prompts import get_prompt_by_index, STRATEGY_NAMES


# ── Response banks (strategy × dataset × sentiment) ──────────────────────────

_RESPONSES: dict[str, dict[str, list[str]]] = {
    # strategy → { dataset or 'default' → [response templates] }
    "ask_info": {
        "twitter": [
            "To help you as quickly as possible, could you let me know what device "
            "and OS version you're using when this happens?",
            "I want to get this resolved fast for you — can you tell me the exact "
            "error message you're seeing?",
            "One quick question: has this issue started recently or has it been "
            "happening for a while?",
        ],
        "reddit": [
            "Thanks for reaching out. To make sure I fully understand the issue, "
            "could you walk me through what you were doing when the problem first appeared?",
            "I'd like to dig into this properly — could you share which plan you're on "
            "and when you first noticed the discrepancy?",
            "Before I suggest a fix, could you clarify whether this happens consistently "
            "or only under certain conditions?",
        ],
        "openassistant": [
            "To point you in the right direction, could you tell me which section of "
            "the settings you've already checked?",
            "Just to make sure I give you the most accurate help — are you accessing "
            "this from the web app or the mobile app?",
            "Could you describe exactly what happens when you try to log in? "
            "Any error messages would be very helpful.",
        ],
        "default": [
            "To help you effectively, could you provide a bit more detail about "
            "when this issue first started?",
            "I want to make sure I fully understand your situation. "
            "Could you clarify one thing for me?",
        ],
    },

    "provide_solution": {
        "twitter": [
            "Got it — here's the quickest fix:\n"
            "1. Force-close the app completely\n"
            "2. Go to Settings → Apps → [App Name] → Clear Cache\n"
            "3. Reopen the app and try again\n"
            "Let me know in 5 mins if that works!",
            "Here's what to do right now:\n"
            "1. Log out of your account\n"
            "2. Wait 60 seconds\n"
            "3. Log back in\n"
            "This refreshes your session token and usually resolves it instantly.",
        ],
        "reddit": [
            "Based on what you've described, here's the step-by-step resolution:\n\n"
            "**Step 1:** Navigate to Account Settings → Billing History\n"
            "**Step 2:** Look for any payments showing 'Pending'\n"
            "**Step 3:** If you see one, click 'Verify Payment' — this triggers a "
            "manual reconciliation within 24 hours\n\n"
            "This resolves about 90% of billing discrepancy cases. "
            "Feel free to reply here if it doesn't work and I'll escalate it.",
            "The intermittent connectivity issue you're describing is typically caused "
            "by a stale DNS cache. Here's how to fix it:\n\n"
            "1. Open Terminal / Command Prompt\n"
            "2. Run: `ipconfig /flushdns` (Windows) or `sudo dscacheutil -flushcache` (Mac)\n"
            "3. Restart your router\n\n"
            "This should stabilise your connection completely.",
        ],
        "openassistant": [
            "Great news — I can walk you through this right now. "
            "To update your notification preferences:\n"
            "1. Click your profile icon (top-right corner)\n"
            "2. Select 'Settings & Privacy'\n"
            "3. Choose 'Notifications'\n"
            "4. Toggle the options to your preference and hit 'Save'\n\n"
            "The changes take effect immediately. Does that help?",
            "The login issue you're experiencing is usually resolved by a simple account unlock. "
            "I've gone ahead and removed the temporary lock on your account — "
            "please try logging in again now using your usual credentials. "
            "You should be in without any issues.",
        ],
        "default": [
            "Based on what you've described, here's how to resolve this issue:\n"
            "1. [Step one specific to your issue]\n"
            "2. [Step two]\n"
            "3. [Verification step]\n"
            "Please try this and let me know if it resolves the problem.",
        ],
    },

    "empathize": {
        "twitter": [
            "I'm really sorry you're dealing with this — having your account locked "
            "for two days is genuinely unacceptable and I completely understand "
            "your frustration. You deserve better than this and I'm personally "
            "going to make sure we get this sorted for you right now.",
            "I hear you and I'm so sorry for the trouble this is causing you. "
            "Losing an order because of an app crash is incredibly frustrating, "
            "and you have every right to be upset. Let me make this right.",
        ],
        "reddit": [
            "I want to acknowledge that dealing with a billing discrepancy for an "
            "extended period is genuinely stressful, especially when you've already "
            "tried to troubleshoot it yourself. Your patience means a lot and I'm "
            "committed to seeing this through with you until it's fully resolved.",
            "I can imagine how frustrating it must be to send in a troubleshooting "
            "guide that didn't actually help your situation. I apologise for that — "
            "generic solutions don't always fit specific cases, and you clearly "
            "need a more targeted approach. I'm here and I'm not going anywhere.",
        ],
        "openassistant": [
            "I'm sorry you've been locked out — that's a stressful situation, "
            "especially when you're relying on the service. I want you to know "
            "you've reached the right place and we'll have this sorted very quickly.",
            "Thank you for being so patient and clear in describing the issue. "
            "I understand how important it is to have access to your "
            "notification settings, and I'll make sure you can configure them "
            "exactly the way you want.",
        ],
        "default": [
            "I completely understand how frustrating this must be. "
            "I sincerely apologise for the inconvenience, and I want you to know "
            "that I'm fully committed to resolving this for you.",
            "Your frustration is completely valid, and I'm sorry we've put you "
            "in this situation. Thank you for your patience — let's fix this together.",
        ],
    },

    "escalate": {
        "twitter": [
            "Your case needs immediate attention from our senior technical team — "
            "I'm escalating this as Priority 1 right now. You'll receive a DM "
            "from a specialist within 1 hour. I'm really sorry for the added step "
            "and genuinely appreciate your patience with us.",
        ],
        "reddit": [
            "After reviewing your case carefully, I believe the best path forward "
            "is to involve our billing specialist team who have direct access to "
            "reconciliation tools that I don't have at this level. "
            "I'm escalating this now with a full summary of our conversation — "
            "you won't need to repeat yourself. Expect contact within 2–3 hours. "
            "I'm very sorry for the extra step.",
        ],
        "openassistant": [
            "Your account issue requires access to our backend systems, which "
            "our specialised account recovery team handles. I'm escalating with "
            "a detailed note so they have full context. They'll reach out within "
            "30 minutes during business hours. I appreciate your patience.",
        ],
        "default": [
            "I'm connecting you with a senior specialist who can resolve this "
            "with the level of expertise your case deserves. They'll contact you "
            "shortly. Thank you for your patience.",
        ],
    },

    "close": {
        "twitter": [
            "Glad we got that sorted so quickly! Is there anything else I can help "
            "you with today? Thanks for reaching out — don't hesitate to tweet us "
            "anytime! 😊",
            "Awesome — all fixed! Thanks for your patience and for bearing with us. "
            "Anything else before I let you go?",
        ],
        "reddit": [
            "Great, I'm glad we were able to work through this together! "
            "To summarise: we identified the root cause and applied the fix — "
            "your service should be fully stable now. Feel free to reply to this "
            "thread if anything comes up. Have a great day!",
            "Perfect — I'm glad we got to the bottom of it. Your account is fully "
            "reconciled now. Is there anything else I can help you with today?",
        ],
        "openassistant": [
            "Wonderful — I'm really happy we could get that sorted out for you! "
            "Your account is fully accessible now. Is there anything else you'd "
            "like help with? Don't hesitate to reach out anytime.",
            "It was a pleasure helping you today! Your settings are configured "
            "just the way you wanted. Feel free to come back if you ever need "
            "anything else. Take care!",
        ],
        "default": [
            "I'm glad we were able to resolve this for you today! "
            "Is there anything else I can help you with? "
            "Thank you for reaching out — have a great day!",
        ],
    },
}


def _pick_response(strategy: str, dataset: str) -> str:
    """Select the best response from the bank, falling back to 'default'."""
    bank = _RESPONSES.get(strategy, {})
    options = bank.get(dataset, bank.get("default", ["I'm here to help you."]))
    return random.choice(options)


# ── Mock LLM class ────────────────────────────────────────────────────────────

class MockLLM:
    """
    A fast, deterministic-ish mock of an LLM.

    In the training loop it is called with:
        response = llm.generate(action_index, llm_context)
    and returns a string response that reflects the chosen strategy.

    During notebook demos, call llm.generate_with_prompt() to also see
    the full formatted prompt (so it looks like a real LLM pipeline).
    """

    def __init__(self, dataset: str = 'twitter', verbose: bool = False):
        self.dataset = dataset
        self.verbose = verbose
        self._call_count = 0

    def generate(self, action_index: int, context: dict | None = None) -> str:
        """Fast path: pick a response without building the full prompt."""
        self._call_count += 1
        strategy = STRATEGY_NAMES[action_index]
        response = _pick_response(strategy, self.dataset)
        if self.verbose:
            print(f"  [MockLLM] strategy={strategy}  | response (truncated): "
                  f"{response[:80]}...")
        return response

    def generate_with_prompt(self, action_index: int,
                             context: dict | None = None) -> tuple[str, str]:
        """
        Returns (formatted_prompt, response) — used in demo notebooks
        so the full pipeline is visible.
        """
        prompt   = get_prompt_by_index(action_index, context)
        response = self.generate(action_index, context)
        return prompt, response

    @property
    def total_calls(self) -> int:
        return self._call_count


# ── Optional real LLM stub ────────────────────────────────────────────────────

class OpenAILLM:
    """
    Drop-in replacement for MockLLM that calls the OpenAI Chat Completions API.
    Requires: pip install openai
              OPENAI_API_KEY environment variable set.

    Usage: replace  MockLLM(dataset)  with  OpenAILLM(dataset, model='gpt-4o-mini')
    in evaluation notebooks only (not training — too slow/expensive).
    """

    def __init__(self, dataset: str = 'twitter', model: str = 'gpt-4o-mini'):
        self.dataset = dataset
        self.model   = model
        self._call_count = 0

    def generate(self, action_index: int, context: dict | None = None) -> str:
        try:
            import openai, os
        except ImportError:
            raise ImportError("Run: pip install openai")

        prompt = get_prompt_by_index(action_index, context)
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",  "content": prompt},
                {"role": "user",    "content": context.get("last_utterance", "Hello") if context else "Hello"},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        self._call_count += 1
        return resp.choices[0].message.content

    def generate_with_prompt(self, action_index: int,
                             context: dict | None = None) -> tuple[str, str]:
        prompt   = get_prompt_by_index(action_index, context)
        response = self.generate(action_index, context)
        return prompt, response

    @property
    def total_calls(self) -> int:
        return self._call_count


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = MockLLM(dataset='twitter', verbose=True)
    ctx = {
        "turn": 2, "sentiment": 0.3, "frustration": 0.7,
        "info_gathered": 0.2,
        "last_utterance": "This app keeps crashing and I need it fixed NOW.",
        "dataset": "Twitter Customer Support",
    }

    print("=" * 65)
    print("  MOCK LLM DEMO — full pipeline walkthrough")
    print("=" * 65)
    for i, strategy in enumerate(STRATEGY_NAMES):
        prompt, response = llm.generate_with_prompt(i, ctx)
        print(f"\n[Strategy {i}: {strategy}]")
        print(f"  Prompt (first 120 chars): {prompt[:120]}...")
        print(f"  Response: {response}")
    print(f"\nTotal LLM calls: {llm.total_calls}")
