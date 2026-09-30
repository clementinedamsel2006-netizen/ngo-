"""SkillConnect AI Assistant.

The assistant uses a language model for conversation/reasoning, while the
application's own CSV data and matching engine remain the source of truth.
"""

import os
import re
import pandas as pd


def _clean(value):
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _safe_records(df, columns=None, limit=80):
    if df is None or df.empty:
        return []
    frame = df.copy()
    if columns:
        keep = [c for c in columns if c in frame.columns]
        frame = frame[keep]
    frame = frame.head(limit)
    return frame.fillna("").to_dict(orient="records")


def build_context(role, volunteer, ngo, opportunities, applications, saved, ngos, rank_opportunities, skill_lexicon):
    """Build a compact, factual context packet for the model."""
    context = {
        "role": role,
        "volunteer": {},
        "ngo": {},
        "opportunities": [],
        "applications": [],
        "saved": [],
        "ngos": [],
        "top_matches": [],
    }

    if role == "Volunteer" and volunteer is not None:
        fields = [
            "Volunteer_ID", "Name", "Skills", "Interests", "Experience",
            "Qualification", "Bio", "Availability", "Preferred_Mode", "Location"
        ]
        context["volunteer"] = {
            k: _clean(volunteer.get(k, "")) for k in fields if _clean(volunteer.get(k, ""))
        }

        open_ops = opportunities.copy()
        if "Status (Open/Closed)" in open_ops.columns:
            open_ops = open_ops[
                open_ops["Status (Open/Closed)"].astype(str).str.lower().isin(["open", ""])
            ]

        ranked = rank_opportunities(
            volunteer,
            open_ops,
            lexicon=skill_lexicon(None, opportunities),
        )[:8]
        context["top_matches"] = [
            {
                "opportunity_id": str(row.get("Opportunity_ID", "")),
                "role_title": _clean(row.get("Role_Title", "Opportunity")),
                "ngo_id": str(row.get("NGO_ID", "")),
                "score": result.get("overall"),
                "reasons": result.get("reasons", [])[:4],
                "skill_gaps": result.get("skill_gap", {}).get("missing", [])[:5],
            }
            for row, result in ranked
        ]

        my_id = str(volunteer.get("Volunteer_ID", ""))
        if "Volunteer_ID" in applications.columns:
            context["applications"] = _safe_records(
                applications[applications["Volunteer_ID"].astype(str) == my_id],
                ["Application_ID", "Opportunity_ID", "Application_Date", "Status", "Match_Score"],
            )
        if "Volunteer_ID" in saved.columns:
            context["saved"] = _safe_records(
                saved[saved["Volunteer_ID"].astype(str) == my_id],
                ["Opportunity_ID"],
            )

    else:
        if ngo is not None:
            fields = ["SrNo", "Name", "Area_of_Work", "Address", "Working_Since"]
            context["ngo"] = {
                k: _clean(ngo.get(k, "")) for k in fields if _clean(ngo.get(k, ""))
            }
            ngo_id = str(ngo.get("SrNo", ""))
            my_ops = opportunities[
                opportunities["NGO_ID"].astype(str) == ngo_id
            ] if "NGO_ID" in opportunities.columns else opportunities.iloc[0:0]
            context["opportunities"] = _safe_records(
                my_ops,
                ["Opportunity_ID", "Role_Title", "Description", "Skills_Required", "Area", "Location", "Mode (Online/Offline/Hybrid)", "Time_Commitment", "Status (Open/Closed)"],
            )
            my_apps = applications[applications["Opportunity_ID"].astype(str).isin(my_ops["Opportunity_ID"].astype(str))] if not my_ops.empty else applications.iloc[0:0]
            context["applications"] = _safe_records(
                my_apps,
                ["Application_ID", "Volunteer_ID", "Opportunity_ID", "Application_Date", "Status", "Match_Score"],
            )

    # Give the model enough directory context to answer NGO/cause questions,
    # but do not dump passwords or unrelated private fields.
    context["ngos"] = _safe_records(ngos, ["SrNo", "Name", "Area_of_Work", "Address"], limit=60)
    if role == "Volunteer":
        context["opportunities"] = _safe_records(
            opportunities,
            ["Opportunity_ID", "NGO_ID", "Role_Title", "Description", "Skills_Required", "Area", "Location", "Mode (Online/Offline/Hybrid)", "Time_Commitment", "Status (Open/Closed)"],
            limit=80,
        )

    return context


def _local_answer(user_text, context):
    """Handle common requests without an API call when possible."""
    text = user_text.lower()
    if "my application" in text or "applications" in text:
        apps = context.get("applications", [])
        if not apps:
            return "You don't have any applications recorded yet."
        lines = ["Here are your current applications:"]
        for app in apps[:8]:
            lines.append(f"• Opportunity {app.get('Opportunity_ID', '')}: {app.get('Status', 'Unknown')}")
        return "\n".join(lines)

    if "saved" in text and context.get("role") == "Volunteer":
        saved = context.get("saved", [])
        if not saved:
            return "You don't have any saved opportunities yet."
        return "Your saved opportunity IDs are: " + ", ".join(str(x.get("Opportunity_ID", "")) for x in saved)

    return None


def ask_ai(user_text, history, context):
    """Ask the configured model while grounding it in current app data."""
    local = _local_answer(user_text, context)
    if local:
        return local

    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "The AI Assistant is ready, but an OpenAI API key has not been configured yet.\n\n"
            "Add `OPENAI_API_KEY` to Streamlit Cloud → Settings → Secrets, then restart the app."
        )

    try:
        from openai import OpenAI
    except ImportError:
        return "The AI package is not installed yet. Add `openai` to requirements.txt and redeploy."

    client = OpenAI(api_key=api_key)

    instructions = """
You are SkillConnect AI, an intelligent volunteering assistant inside a real NGO/volunteer platform.

Your job is to reason over the supplied application data and help the user make informed decisions.
Do not invent NGOs, opportunities, requirements, application statuses, scores, or user information.
The supplied matching results are computed by the application's matching engine and are the source of truth for match scores.
Do not expose the internal mathematical score breakdown unless the user explicitly asks how matching works.
When a user asks for recommendations, explain useful trade-offs rather than merely repeating a list.
When information is missing, say that it is missing and suggest what profile information would improve the answer.
You can help with: finding opportunities, explaining matches, comparing opportunities, identifying skill gaps, checking application readiness, understanding applications, and improving a volunteer profile.
For NGO users, you can help interpret their opportunities and applicant data, identify common skills, and summarize applicant patterns.
Be concise but useful. Use bullets when comparing several items.
"""

    prompt = "CURRENT APP DATA (JSON):\n" + __import__("json").dumps(context, ensure_ascii=False, default=str)
    prompt += "\n\nUSER MESSAGE:\n" + user_text

    messages = []
    for item in (history or [])[-8:]:
        if item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": item.get("content", "")})
    messages.append({"role": "user", "content": prompt})

    response = client.responses.create(
        model=os.getenv("SKILLCONNECT_AI_MODEL", "gpt-5.5"),
        instructions=instructions,
        input=messages,
        max_output_tokens=700,
    )
    return response.output_text.strip()


def render_ai_assistant(role, volunteer, ngo, opportunities, applications, saved, ngos, rank_opportunities, skill_lexicon):
    """Render the Streamlit chat page."""
    import streamlit as st

    st.header("🤖 SkillConnect AI")
    st.caption("Your AI volunteering assistant — understand matches, compare roles, find skill gaps, and get application guidance.")

    context = build_context(
        role, volunteer, ngo, opportunities, applications, saved, ngos,
        rank_opportunities, skill_lexicon
    )

    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = []

    suggestions = (
        [
            "What opportunities fit me best and why?",
            "What skills am I missing for the strongest opportunities?",
            "Compare my top two opportunities.",
            "Am I ready to apply for my best match?",
        ] if role == "Volunteer" else [
            "Summarize my current opportunities.",
            "What skills are common among my applicants?",
            "How can I improve my opportunity descriptions?",
        ]
    )

    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, key=f"ai_suggestion_{i}", use_container_width=True):
            st.session_state.ai_pending_prompt = suggestion

    for message in st.session_state.ai_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state.pop("ai_pending_prompt", None)
    user_text = st.chat_input("Ask SkillConnect AI anything about volunteering...")
    if pending and not user_text:
        user_text = pending

    if user_text:
        st.session_state.ai_chat_messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.spinner("SkillConnect AI is thinking..."):
                answer = ask_ai(user_text, st.session_state.ai_chat_messages[:-1], context)
            st.markdown(answer)
        st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})
