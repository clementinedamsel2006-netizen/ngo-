"""SkillConnect AI - first-layer Career & Volunteering Advisor.

This module is intentionally read-only: it reasons over the app's existing
profile, opportunity, application, saved-item and matching data. It does not
change CSVs or submit applications.
"""

import json
import os
import re


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


def _opportunity_fields(row):
    """Return the fields the advisor needs for constraint reasoning."""
    wanted = [
        "Opportunity_ID", "NGO_ID", "Role_Title", "Description",
        "Skills_Required", "Interests", "Area", "Location",
        "Mode (Online/Offline/Hybrid)", "Time_Commitment",
        "Availability", "Experience_Required", "Qualification_Required",
        "Status (Open/Closed)",
    ]
    return {k: _clean(row.get(k, "")) for k in wanted if _clean(row.get(k, ""))}


def build_context(role, volunteer, ngo, opportunities, applications, saved,
                  ngos, rank_opportunities, skill_lexicon):
    """Build factual context for the AI advisor from current app data."""
    context = {
        "role": role,
        "volunteer": {},
        "ngo": {},
        "opportunities": [],
        "top_matches": [],
        "applications": [],
        "saved": [],
        "ngos": [],
    }

    if role == "Volunteer" and volunteer is not None:
        fields = [
            "Volunteer_ID", "Name", "Skills", "Interests", "Experience",
            "Qualification", "Bio", "Availability", "Preferred_Mode", "Location"
        ]
        context["volunteer"] = {
            k: _clean(volunteer.get(k, ""))
            for k in fields if _clean(volunteer.get(k, ""))
        }

        open_ops = opportunities.copy()
        if "Status (Open/Closed)" in open_ops.columns:
            status = open_ops["Status (Open/Closed)"].astype(str).str.strip().str.lower()
            open_ops = open_ops[status.isin(["open", ""])]

        # The existing matching engine remains the source of truth for match scores.
        try:
            ranked = rank_opportunities(
                volunteer,
                open_ops,
                lexicon=skill_lexicon(None, opportunities),
            )[:10]
        except Exception:
            ranked = []

        context["top_matches"] = []
        for row, result in ranked:
            context["top_matches"].append({
                "opportunity": _opportunity_fields(row),
                "match_score": result.get("overall"),
                "match_reasons": result.get("reasons", [])[:5],
                "skill_gaps": result.get("skill_gap", {}).get("missing", [])[:8],
            })

        # Keep all currently open opportunities available to the advisor so it
        # can reason about explicit constraints rather than only the top scores.
        context["opportunities"] = [
            _opportunity_fields(row) for _, row in open_ops.head(100).iterrows()
        ]

        my_id = str(volunteer.get("Volunteer_ID", ""))
        if "Volunteer_ID" in applications.columns:
            my_apps = applications[applications["Volunteer_ID"].astype(str) == my_id]
            context["applications"] = _safe_records(
                my_apps,
                ["Application_ID", "Opportunity_ID", "Application_Date", "Status", "Match_Score"],
                limit=30,
            )
        if "Volunteer_ID" in saved.columns:
            my_saved = saved[saved["Volunteer_ID"].astype(str) == my_id]
            context["saved"] = _safe_records(my_saved, ["Opportunity_ID"], limit=50)

    elif ngo is not None:
        fields = ["SrNo", "Name", "Area_of_Work", "Address", "Working_Since"]
        context["ngo"] = {
            k: _clean(ngo.get(k, ""))
            for k in fields if _clean(ngo.get(k, ""))
        }
        ngo_id = str(ngo.get("SrNo", ""))
        my_ops = (
            opportunities[opportunities["NGO_ID"].astype(str) == ngo_id]
            if "NGO_ID" in opportunities.columns else opportunities.iloc[0:0]
        )
        context["opportunities"] = [
            _opportunity_fields(row) for _, row in my_ops.head(100).iterrows()
        ]
        if not my_ops.empty and "Opportunity_ID" in my_ops.columns:
            my_apps = applications[
                applications["Opportunity_ID"].astype(str).isin(my_ops["Opportunity_ID"].astype(str))
            ]
            context["applications"] = _safe_records(
                my_apps,
                ["Application_ID", "Volunteer_ID", "Opportunity_ID", "Application_Date", "Status", "Match_Score"],
                limit=100,
            )

    context["ngos"] = _safe_records(
        ngos, ["SrNo", "Name", "Area_of_Work", "Address"], limit=60
    )
    return context


def _local_answer(user_text, context):
    """Answer simple factual queries without spending an API call."""
    text = user_text.lower().strip()

    if context.get("role") == "Volunteer" and ("my application" in text or "my applications" in text):
        apps = context.get("applications", [])
        if not apps:
            return "You don't have any applications recorded yet."
        lines = ["Here are your current applications:"]
        for app in apps[:10]:
            lines.append(
                f"• Opportunity {app.get('Opportunity_ID', '')}: "
                f"{app.get('Status', 'Status not recorded')}"
            )
        return "\n".join(lines)

    if context.get("role") == "Volunteer" and "saved" in text:
        saved = context.get("saved", [])
        if not saved:
            return "You don't have any saved opportunities yet."
        ids = [str(x.get("Opportunity_ID", "")) for x in saved if x.get("Opportunity_ID", "")]
        return "Your saved opportunities are: " + ", ".join(ids)

    return None


def _instructions(role):
    base = """
You are SkillConnect AI, the Career & Volunteering Advisor inside a real NGO
volunteering platform. You are a decision-support agent, not merely a generic
chatbot.

GROUNDING RULES
- Use only the supplied current app data for claims about this user's profile,
  opportunities, NGOs, applications, saved items, requirements, availability,
  locations, modes and match scores.
- Never invent an opportunity, NGO, requirement, application status, profile
  detail, availability slot or score.
- The application's matching engine is the source of truth for match scores.
- Do not expose the internal mathematical component breakdown unless the user
  explicitly asks how the score is calculated.
- If a required field is missing, say it is not recorded rather than guessing.
- Distinguish an exact requirement from a reasonable suggestion.

CONSTRAINT REASONING
When the user gives constraints such as hours, day, skills, children,
location, travel distance, remote/online mode or experience, explicitly check
those constraints against the opportunity data. Do not select an opportunity
only because its semantic match score is high.

COMPARISON
When comparing opportunities, discuss concrete trade-offs across:
- skill compatibility
- interest/cause alignment
- time and availability fit
- location/travel and work mode
- experience/qualification requirements
- responsibilities described by the opportunity
- known skill gaps
Do not turn this into a generic score-only recommendation.

APPLICATION READINESS
When asked "Can I apply?", separate:
1. requirements clearly met by the profile,
2. requirements not currently demonstrated,
3. requirements that are missing from the data.
Then give practical application advice without claiming the NGO will accept or
reject the application.

APPLICATION DRAFTING
When asked to help apply, create a personalized draft using only the user's
recorded skills, experience, qualification, interests and the selected
opportunity. Never invent achievements. Clearly label it as a draft the user
can edit before submitting. Ask for missing information only when it is needed.

SKILL DEVELOPMENT
For "what should I improve?", connect documented gaps to opportunities. If
suggesting a new skill, explain that it is a development suggestion, not a
requirement unless the opportunity explicitly lists it.

STYLE
Be clear, practical and concise. Use headings and bullets when helpful. Do
not repeat the raw dataset. Explain the reasoning behind your answer.
"""
    if role == "NGO":
        base += "\nFor NGO users, focus on opportunity quality, applicant patterns, skills, and profile/opportunity improvements."
    else:
        base += "\nFor volunteers, focus on fit, constraints, readiness, skill development and application preparation."
    return base


def ask_ai(user_text, history, context):
    """Run the first-layer AI advisor using current app context."""
    local = _local_answer(user_text, context)
    if local:
        return local

    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
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

    data_prompt = json.dumps(context, ensure_ascii=False, default=str)
    user_prompt = (
        "CURRENT SKILLCONNECT APP CONTEXT:\n"
        + data_prompt
        + "\n\nUSER REQUEST:\n"
        + user_text
    )

    messages = []
    for item in (history or [])[-10:]:
        if item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": item.get("content", "")})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.responses.create(
            model=os.getenv("SKILLCONNECT_AI_MODEL", "gpt-5.5"),
            instructions=_instructions(context.get("role", "Volunteer")),
            input=messages,
            max_output_tokens=900,
        )
        answer = response.output_text.strip()
        return answer or "I couldn't produce an answer from the available SkillConnect data."
    except Exception as exc:
        return (
            "I couldn't reach the AI service right now. Please check the OpenAI API "
            "configuration and try again.\n\n"
            f"Technical detail: {type(exc).__name__}"
        )


def render_ai_assistant(role, volunteer, ngo, opportunities, applications, saved,
                        ngos, rank_opportunities, skill_lexicon):
    """Render the first-layer SkillConnect AI Advisor page."""
    import streamlit as st

    st.header("🤖 SkillConnect AI")
    st.caption(
        "Your AI volunteering advisor — find, compare, understand and prepare for opportunities."
    )

    context = build_context(
        role, volunteer, ngo, opportunities, applications, saved, ngos,
        rank_opportunities, skill_lexicon
    )

    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = []

    if role == "Volunteer":
        suggestions = [
            "🔎 Find opportunities that fit my skills, availability and preferences",
            "⚖️ Compare my top two opportunities and explain the trade-offs",
            "📋 Can I apply for my best matching opportunity?",
            "📝 Help me prepare an application for my best match",
            "📊 What skills should I improve for more technical opportunities?",
            "💡 What should I improve in my profile?",
        ]
        st.markdown("**What can I help with?**")
    else:
        suggestions = [
            "📋 Summarize my current opportunities",
            "📊 What skills are common among my applicants?",
            "💡 How can I improve my opportunity descriptions?",
            "🔎 What patterns do you see in my applicants?",
        ]
        st.markdown("**What can I help with?**")

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(
            suggestion,
            key=f"ai_suggestion_{i}",
            use_container_width=True,
        ):
            st.session_state.ai_pending_prompt = suggestion

    for message in st.session_state.ai_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state.pop("ai_pending_prompt", None)
    placeholder = (
        "Tell me what you want to find, compare, understand or prepare..."
        if role == "Volunteer"
        else "Ask about your opportunities or applicants..."
    )
    user_text = st.chat_input(placeholder)
    if pending and not user_text:
        user_text = pending

    if user_text:
        previous_history = list(st.session_state.ai_chat_messages)
        st.session_state.ai_chat_messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.spinner("SkillConnect AI is thinking..."):
                answer = ask_ai(user_text, previous_history, context)
            st.markdown(answer)
        st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})
