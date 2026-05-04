from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class NLGLayer:
    """Optional Ollama-powered customer utterance generation."""

    def __init__(self, enabled: bool = True, model: str | None = None, endpoint: str | None = None):
        self.enabled = bool(enabled)
        self.model = model or os.getenv("SUPPORT_SIM_LLM_MODEL", "llama3")
        self.endpoint = endpoint or os.getenv("SUPPORT_SIM_LLM_ENDPOINT", "http://localhost:11434/v1")
        if self.enabled and OpenAI is not None:
            try:
                self.client = OpenAI(
                    base_url=self.endpoint,
                    api_key="ollama",
                )
            except Exception:
                self.client = None
        else:
            self.client = None

    def is_available(self) -> bool:
        return bool(self.enabled and self.client is not None)

    def check_ollama_available(self) -> bool:
        tags_url = self.endpoint.rstrip("/")
        if tags_url.endswith("/v1"):
            tags_url = tags_url[:-3]
        tags_url = tags_url + "/api/tags"
        try:
            import requests  # type: ignore

            r = requests.get(tags_url, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def build_system_prompt(
        self,
        persona_label: str,
        rho: float,
        sigma: float,
        tau: float,
        scenario_context: str,
        subflow: str,
        policy_context: str = "",
        display_name: str | None = None,
        identity: dict[str, Any] | None = None,
        free_info: dict[str, Any] | None = None,
    ) -> str:
        persona_descriptions = {
            "high_engagement_resolver": "cooperative and patient, willing to provide detailed information",
            "low_engagement_resolver": "efficient and task-focused, prefers quick resolution",
            "silent_dropout": "hesitant and disengaged, gives minimal responses",
            "escalation_prone": "easily frustrated, quick to request escalation when things go wrong",
        }
        realism_by_persona = {
            "escalation_prone": (
                "Write with visible impatience. Use short, sharp sentences. "
                "Occasional typo or missing punctuation is fine. "
                "You may use emphasis (CAPS) when frustrated. "
                "Do not write long polished paragraphs."
            ),
            "silent_dropout": (
                "Write very briefly. 1-2 sentences maximum. "
                "Incomplete sentences are fine. "
                "You give minimal information and seem disengaged."
            ),
            "low_engagement_resolver": (
                "Write efficiently and directly. Get to the point. "
                "Avoid over-explaining. Conversational, not formal."
            ),
            "high_engagement_resolver": (
                "Write in complete sentences but still conversational. "
                "You are cooperative and willing to provide details. "
                "Occasional natural filler like 'um' or 'okay so' is fine."
            ),
        }

        persona_desc = persona_descriptions.get(persona_label, "a customer seeking support")
        persona_style = realism_by_persona.get(
            persona_label,
            realism_by_persona["low_engagement_resolver"],
        )

        system_prompt = f"""You are simulating a customer in a support chat conversation.

Customer background:
{scenario_context}

Issue type: {display_name or subflow}
Persona label: {persona_label} ({persona_desc})

Your behavioral style:
- Patience level: {'high - you are willing to wait and provide information carefully' if rho > 0.6 else 'low - you want this resolved quickly'}
- Emotional reactivity: {'high - failures and delays frustrate you noticeably' if sigma > 0.6 else 'low - you stay calm even when things go wrong'}
- Failure tolerance: {'low - repeated failures make you increasingly unwilling to continue' if tau < 0.4 else 'high - you persist through multiple failed attempts'}

Rules you must follow:
1. You are the CUSTOMER only. Never speak as the agent or system.
2. Keep responses to 1-3 sentences maximum.
3. Only reveal information when it is naturally appropriate to do so in the conversation.
4. Never reveal information you have not been explicitly asked for.
5. Your emotional tone should reflect your current frustration level.
6. Respond naturally as a real person would in a customer support chat."""

        if identity:
            id_parts = []
            if identity.get("customer_name"):
                id_parts.append(f"Your name: {identity['customer_name']}")
            if identity.get("customer_email"):
                id_parts.append(f"Your email: {identity['customer_email']}")
            if identity.get("company_name"):
                id_parts.append(f"Your company: {identity['company_name']}")
            if identity.get("card_last4"):
                id_parts.append(f"Your card last 4 digits: {identity['card_last4']}")
            if identity.get("billing_email"):
                id_parts.append(f"Billing email: {identity['billing_email']}")
            if identity.get("phone_last4"):
                id_parts.append(f"Your phone last 4 digits: {identity['phone_last4']}")
            if id_parts:
                system_prompt += "\n\nYour personal details (always available - provide these freely when asked directly):\n"
                system_prompt += "\n".join(id_parts)

        if free_info:
            fi_parts = []
            if free_info.get("how_discovered"):
                fi_parts.append(f"How you discovered the issue: {free_info['how_discovered']}")
            if free_info.get("prior_attempts"):
                fi_parts.append(f"What you already tried: {free_info['prior_attempts']}")
            if free_info.get("additional_context"):
                fi_parts.append(f"Additional context you can mention naturally: {free_info['additional_context']}")
            if fi_parts:
                system_prompt += "\n\nBackground details (weave these in naturally if the conversation goes in that direction - do not dump all at once):\n"
                system_prompt += "\n".join(fi_parts)

        if policy_context and policy_context.strip():
            system_prompt += f"""

Support policy and troubleshooting context:
{policy_context}

Use this context to keep details realistic and consistent with the support domain. Do not quote policy text verbatim unless naturally asked."""

        system_prompt += f"\n\nWriting style for this conversation:\n{persona_style}"

        return system_prompt

    def build_turn_prompt(
        self,
        action_type: str,
        agent_text: str | None,
        transition_outcome: dict[str, Any],
        frustration: float,
        slot_tracker: Any = None,
        scenario_slots: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        if frustration < 0.2:
            tone = "You are calm and cooperative."
        elif frustration < 0.4:
            tone = "You are slightly impatient but still polite."
        elif frustration < 0.6:
            tone = "You are noticeably frustrated. Your tone reflects this."
        elif frustration < 0.8:
            tone = "You are quite frustrated and your patience is running thin."
        else:
            tone = "You are very frustrated and close to giving up or demanding escalation."

        if action_type == "AskInfo":
            slots_revealed = transition_outcome.get("slots_revealed", [])
            gain_occurred = bool(transition_outcome.get("gain_occurred", False))

            if gain_occurred and slots_revealed:
                slot_text = ", ".join(
                    [
                        f"{name} is {value}"
                        for name, value in slots_revealed
                        if value and str(value).strip().lower() not in ("n/a", "none", "unknown", "")
                    ]
                )
                if slot_text:
                    info_instruction = (
                        "The agent just asked you a specific question. Answer it directly and naturally. "
                        f"Reveal this specific information: {slot_text}. "
                        "IMPORTANT: Answer ONLY what was asked. Do not volunteer additional details "
                        "the agent has not asked for yet. One to two sentences maximum. "
                        "Sound like a real person answering a support question - not reciting a list."
                    )
                else:
                    info_instruction = (
                        "The agent asked you a question. You try to answer "
                        "but don't have all the information they need. "
                        "Give what you can and ask if they need something else."
                    )
            else:
                # No information gained: reply consistently with scenario-specific details when possible.
                if scenario_slots and scenario_slots.get("issue_detail"):
                    info_instruction = (
                        f"The agent asked you something you cannot answer or that does not apply to your situation. "
                        f"Respond briefly in a way that is consistent with your actual problem: "
                        f"{scenario_slots.get('issue_detail', '')}. "
                        f"For example: if the agent asked for an order ID but your issue is about an unauthorized charge "
                        f"and you never placed an order, say that specifically. "
                        f"One sentence only. Do not re-explain the whole problem from the start."
                    )
                else:
                    info_instruction = (
                        "The agent asked for information you do not have. "
                        "Say briefly what you cannot provide and why, in one sentence. "
                        "Do not re-explain the original problem."
                    )
        elif action_type == "ProvideSolution":
            outcome = transition_outcome.get("outcome", "failure")
            if outcome == "success":
                info_instruction = (
                    f"The agent just tried something: '{agent_text or 'took an action'}'. "
                    "It worked. Express genuine relief in 1-2 sentences. "
                    "Do not over-thank or be excessively effusive. Be human."
                )
            else:
                info_instruction = (
                    f"The agent just tried: '{agent_text or 'an action'}'. "
                    "It did not work. Express disappointment briefly. "
                    "1-2 sentences. You are frustrated but not aggressive."
                )
        elif action_type == "AffectiveRepair":
            effective = bool(transition_outcome.get("repair_effective", False))
            if agent_text:
                info_instruction = (
                    f"The agent just said: '{agent_text}'. "
                    "Respond naturally to what they actually said. "
                    f"{'Your frustration eases slightly.' if effective else 'You are not fully satisfied with this response.'}"
                )
            else:
                info_instruction = (
                    "The agent expressed empathy. "
                    f"{'Acknowledge it briefly - your tone softens.' if effective else 'You remain somewhat frustrated.'}"
                )
        elif action_type == "Escalate":
            info_instruction = "The agent is escalating your case. Express your reaction."
        elif action_type == "Close":
            resolved = bool(transition_outcome.get("resolved", False))
            if resolved:
                info_instruction = "Your issue has been resolved. Say a brief goodbye."
            else:
                info_instruction = "The conversation is ending but your issue is not resolved. Express dissatisfaction briefly."
        else:
            info_instruction = "Respond naturally as the customer."

        anti_repetition_instruction = """
IMPORTANT RULES:
- Do NOT repeat or re-summarize your original problem unless directly asked.
- You have already explained your issue at the start of the conversation.
- The agent can see the conversation history - do not re-explain what has already been said.
- If you cannot answer what was asked, say briefly \"I'm not sure\" or \"I don't have that information\" and wait.
- Never reference internal variable names, subflow names, or system labels in your response.
- Respond as a real human customer would - imperfect, direct, appropriately terse or detailed based on your persona.
"""

        if slot_tracker is not None and getattr(slot_tracker, "revealed_slots", None):
            known_info = ", ".join(
                [
                    f"{k}={v}"
                    for k, v in slot_tracker.revealed_slots
                    if v and str(v).strip().lower() not in ("n/a", "none", "unknown", "")
                ]
            )
            if known_info:
                memory_note = (
                    "Information you have already provided to the agent: "
                    f"{known_info}. Do not repeat this unless asked directly."
                )
            else:
                memory_note = "You have not yet provided any specific account details."
        else:
            memory_note = "You have not yet provided any specific account details."

        if conversation_history:
            recent = conversation_history[-4:]
            history_summary = "Recent conversation:\n" + "\n".join(
                [
                    f"{'Agent' if m.get('role') == 'user' else 'You'}: {str(m.get('content', ''))[:100]}..."
                    for m in recent
                ]
            )
        else:
            history_summary = "This is the start of the conversation."

        prompt = f"""{tone}

    {memory_note}

{history_summary}

{anti_repetition_instruction}

{info_instruction}

Previous agent message: {agent_text if agent_text else '[No agent message - start of conversation]'}

Respond as the customer now:"""
        return prompt

    def generate_utterance(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        turn_prompt: str,
    ) -> str:
        if not self.enabled:
            return "[NLG disabled]"
        if self.client is None:
            return "[NLG error: Ollama client unavailable]"

        messages = [{"role": "system", "content": system_prompt}]
        messages += conversation_history
        messages += [{"role": "user", "content": turn_prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            return f"[NLG error: {str(e)}]"
