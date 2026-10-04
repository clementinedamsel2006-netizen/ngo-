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
    """Ask the free Gemini model while grounding it in current app data."""
    local = _local_answer(user_text, context)
    if local:
        return local

    # Read the key from Streamlit Cloud Secrets first, then environment variables.
    api_key = None
    secret_source = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            secret_source = "Streamlit Secrets"
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            secret_source = "environment variable"

    if isinstance(api_key, str):
        api_key = api_key.strip()

    if not api_key:
        return (
            "Gemini configuration error: `GEMINI_API_KEY` was not found.\n\n"
            "Add it to Streamlit Cloud → Settings → Secrets as:\n"
            '`GEMINI_API_KEY = "YOUR_KEY"`\n\n'
            "Then reboot/redeploy the app."
        )

    try:
        from google import genai
    except ImportError:
        return (
            "The Gemini AI package is not installed. "
            "Add `google-genai` to requirements.txt and redeploy."
        )

    instructions = """
You are SkillConnect AI, the conversational AI assistant inside a real
NGO/volunteer platform.

Your job is to reason over the supplied application data and help the user
make informed decisions.

IMPORTANT RULES:
- The supplied CSV/application data is the source of truth.
- Do not invent NGOs, opportunities, requirements, application statuses,
  scores, or user information.
- The supplied matching results are computed by the application's matching
  engine and are the source of truth for match scores.
- Do not expose the internal mathematical score breakdown unless the user
  explicitly asks how matching works.
- When recommending opportunities, explain useful reasons and trade-offs.
- When information is missing, say that it is missing.
- You can help with finding opportunities, explaining matches, comparing
  opportunities, identifying skill gaps, checking application readiness,
  understanding applications, and improving a volunteer profile.
- For NGO users, help interpret their opportunities and applicant data,
  identify common skills, and summarize applicant patterns.
- Be conversational, concise, friendly, and useful.
"""

    prompt = (
        instructions
        + "\n\nCURRENT APP DATA (JSON):\n"
        + __import__("json").dumps(context, ensure_ascii=False, default=str)
    )

    if history:
        prompt += "\n\nRECENT CONVERSATION:\n"
        for item in history[-8:]:
            if item.get("role") in {"user", "assistant"}:
                prompt += (
                    f"{item['role'].upper()}: "
                    f"{item.get('content', '')}\n"
                )

    prompt += "\n\nUSER MESSAGE:\n" + user_text

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv(
                "SKILLCONNECT_AI_MODEL",
                "gemini-3.5-flash-lite"
            ),
            contents=prompt,
            config={
                "max_output_tokens": 700,
                "temperature": 0.5,
            },
        )

        answer = getattr(response, "text", None)
        if answer:
            return answer.strip()

        return "I couldn't generate a response right now. Please try again."

    except Exception as e:
        # Expose the actual Gemini failure without exposing the API key.
        error_text = str(e)
        combined = error_text.lower()
        status = getattr(e, "code", None) or getattr(e, "status_code", None)
        if status is not None:
            combined = f"{status} {combined}"

        if ("429" in combined or "resource_exhausted" in combined or
                "quota" in combined or "rate limit" in combined):
            return (
                "Gemini API error: 429 / QUOTA OR RATE LIMIT.\n\n"
                "The key was accepted, but the Gemini project/key has reached a usage limit.\n\n"
                f"Technical detail: {error_text[:600]}"
            )

        if ("401" in combined or "unauthenticated" in combined or
                "invalid api key" in combined or "api key not valid" in combined):
            return (
                "Gemini API error: 401 / UNAUTHENTICATED.\n\n"
                f"The request used {secret_source or 'a configured credential'}, but Google rejected it. "
                "Create a fresh Gemini API key and replace the Streamlit secret.\n\n"
                f"Technical detail: {error_text[:600]}"
            )

        if ("403" in combined or "permission denied" in combined or
                "permission_denied" in combined or "forbidden" in combined):
            return (
                "Gemini API error: 403 / PERMISSION DENIED.\n\n"
                "The key exists, but Google is refusing access. Check the Google project/API access "
                "and the restrictions on this key.\n\n"
                f"Technical detail: {error_text[:600]}"
            )

        if "404" in combined or "not found" in combined:
            return (
                "Gemini API error: 404 / NOT FOUND.\n\n"
                "The configured Gemini model or endpoint is not available to this API/project.\n\n"
                f"Technical detail: {error_text[:600]}"
            )

        return (
            "Gemini API error: the request failed.\n\n"
            f"Credential source: {secret_source or 'unknown'}\n"
            f"Error type: {type(e).__name__}\n"
            f"Technical detail: {error_text[:800]}"
        )


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
