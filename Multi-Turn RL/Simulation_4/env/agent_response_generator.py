from __future__ import annotations

"""
Agent Response Generator — Stage 3.

Generates natural-language agent responses for each RL action using Ollama.
This is the agent-side counterpart to NLGLayer (which generates customer
utterances).  Together they build a realistic two-sided conversation that is
fed back into the intent classifier and the customer NLG as context.

The generator is called inside NLPObservationWrapper.step() BEFORE env.step(),
so the generated agent text is passed to SupportEnv as agent_text and stored
in conversation_history.

Model selection:
  - Uses the same Ollama endpoint as NLGLayer.
  - Defaults to the same model (SUPPORT_SIM_LLM_MODEL, typically llama3).
  - A separate env var SUPPORT_SIM_AGENT_MODEL can override for the agent side
    if you want a different model for agent vs. customer (e.g., phi3 for agent,
    llama3 for customer).
"""

import os
from typing import Any

from Simulation_4.llm.backends import ChatBackend, make_backend


_SYSTEM_PROMPT = """\
You are a professional customer support agent working in a live chat.
Your goal is to resolve the customer's issue efficiently and empathetically.

Rules:
1. Write 1-3 sentences maximum. Be clear and direct.
2. Match your tone to the action you are taking.
3. Do not ask for information you already have.
4. Do not promise what you cannot deliver.
5. Never say "I cannot help you" — always offer the next step.
6. Respond as the AGENT only. Never speak as the customer.
"""


class AgentResponseGenerator:
    """Generate natural-language agent responses via Ollama."""

    # Canonical fallback phrases for when Ollama is unavailable.
    FALLBACK_TEXTS = {
        "AskInfo": "To help you better, could you provide a few more details about the issue?",
        "ProvideSolution": "I have applied a fix on our end. Please try again and let me know if the issue is resolved.",
        "AffectiveRepair": "I completely understand how frustrating this situation is, and I sincerely apologise for the inconvenience.",
        "Escalate": "I am going to connect you with a specialist who has the expertise to resolve this for you right away.",
        "Close": "I am glad we could resolve your issue today. Do not hesitate to reach out if you need anything else.",
    }

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        backend: ChatBackend | None = None,
    ):
        self.model = (
            model
            or os.getenv("SUPPORT_SIM_AGENT_MODEL")
            or os.getenv("SUPPORT_SIM_LLM_MODEL", "llama3")
        )
        self.endpoint = endpoint or os.getenv("SUPPORT_SIM_LLM_ENDPOINT", "http://localhost:11434/v1")

        self.backend: ChatBackend | None = backend
        if self.backend is None:
            self.backend = make_backend()

    def is_available(self) -> bool:
        return bool(self.backend is not None and self.backend.is_available())

    def generate(
        self,
        action_name: str,
        conversation_history: list[dict[str, str]],
        policy_context: str = "",
        classification: dict[str, Any] | None = None,
    ) -> str:
        """Generate a natural language agent response for the given action.

        Parameters
        ----------
        action_name : str
            The action the RL policy chose (AskInfo / ProvideSolution / etc.)
        conversation_history : list[dict]
            Current conversation history (role/content dicts).
        policy_context : str
            Retrieved policy/playbook excerpts from LumoRAG.
        classification : dict
            Latest intent classification from IntentClassifier (optional).
        """
        if not self.is_available():
            return self.FALLBACK_TEXTS.get(action_name, f"Agent action: {action_name}")

        action_instruction = self._action_instruction(action_name, classification)

        # Build system prompt with policy context and action instruction.
        system = _SYSTEM_PROMPT
        if policy_context:
            system += f"\n\nRelevant policy context:\n{policy_context[:600]}"
        system += f"\n\nYour current action: {action_name}\n{action_instruction}"

        # Include last 4 turns for context (keep token count low).
        messages = [{"role": "system", "content": system}]
        messages += (conversation_history[-4:] if conversation_history else [])
        messages.append({
            "role": "user",
            "content": f"Now write your response as the agent taking the action: {action_name}. "
                       "1-2 sentences maximum. Do not start with 'Agent:' or any label.",
        })

        try:
            assert self.backend is not None
            text = self.backend.chat(
                model=self.model,
                messages=messages,
                max_tokens=100,
                temperature=0.6,
            )
            return text if text else self.FALLBACK_TEXTS.get(action_name, f"Agent: {action_name}")
        except Exception:
            return self.FALLBACK_TEXTS.get(action_name, f"Agent: {action_name}")

    def _action_instruction(self, action_name: str, classification: dict[str, Any] | None) -> str:
        sentiment = (classification or {}).get("sentiment", "neutral")
        info_complete = float((classification or {}).get("info_completeness", 0.5))

        if action_name == "AskInfo":
            if info_complete < 0.3:
                return (
                    "You need the customer's basic information. "
                    "Ask for one specific piece of information (e.g., order number, email, or account ID)."
                )
            return "Ask a focused follow-up question to gather the remaining information needed."

        if action_name == "ProvideSolution":
            return (
                "Offer a concrete solution or next step. "
                "Be specific — name the action you are taking (e.g., 'I have reset your password', "
                "'I have issued a refund', 'Please clear your browser cache and try again')."
            )

        if action_name == "AffectiveRepair":
            if sentiment == "frustrated":
                return (
                    "The customer is frustrated. Lead with a sincere apology, "
                    "acknowledge the inconvenience, and reassure them you will resolve this."
                )
            return "Acknowledge the customer's experience and reassure them you are working to help."

        if action_name == "Escalate":
            return (
                "Explain that you are connecting them with a specialist. "
                "Be positive — frame escalation as getting them expert help, not as a failure."
            )

        if action_name == "Close":
            return (
                "Confirm the issue has been resolved (or summarise next steps if not fully resolved). "
                "Thank the customer and close warmly."
            )

        return "Respond professionally and helpfully."
