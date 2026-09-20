import re
from functools import lru_cache

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
#              SENTENCE TRANSFORMER MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    """Load the Sentence Transformer once and reuse it."""
    return SentenceTransformer(MODEL_NAME)


# ============================================================
#                    TEXT HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    text = str(value)

    if text.lower() in ["nan", "none", "nat"]:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def normalise(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9+#&./ -]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_items(value):
    """
    Split skills/interests/availability into simple items.
    Supports commas, semicolons, slashes and line breaks.
    """
    text = clean_text(value)

    if not text:
        return []

    parts = re.split(r"[,;|\n]+", text)

    result = []

    for part in parts:
        part = normalise(part)

        if part:
            result.append(part)

    return result


def contains_phrase(text, phrase):
    text = normalise(text)
    phrase = normalise(phrase)

    if not text or not phrase:
        return False

    return phrase in text


# ============================================================
#                 SKILL LEXICON
# ============================================================

DEFAULT_SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "machine learning",
    "artificial intelligence",
    "data analysis",
    "data analytics",
    "data science",
    "excel",
    "google sheets",
    "spreadsheet",
    "power bi",
    "tableau",
    "data entry",
    "web development",
    "website maintenance",
    "programming",
    "coding",
    "it support",
    "digital tools",
    "digital literacy",
    "graphic design",
    "canva",
    "adobe photoshop",
    "adobe illustrator",
    "branding",
    "photography",
    "photo editing",
    "video editing",
    "content writing",
    "content creation",
    "social media",
    "communication",
    "marketing",
    "fundraising",
    "event management",
    "event coordination",
    "teaching",
    "mentoring",
    "tutoring",
    "research",
    "documentation",
    "legal documentation",
    "community outreach",
    "public speaking",
    "counselling",
    "counseling",
    "psychology",
    "healthcare",
    "first aid",
    "finance",
    "accounting",
    "project management",
    "operations",
    "environment",
    "sustainability",
    "waste management",
    "recycling",
]


def skill_lexicon(volunteers, opportunities):
    """
    Build a skill vocabulary from the actual CSV data plus a small
    general vocabulary. This is used for transparent skill matching.
    """
    terms = set(DEFAULT_SKILLS)

    if volunteers is not None and not volunteers.empty:
        for value in volunteers.get("Skills", []):
            for item in split_items(value):
                if len(item) >= 2:
                    terms.add(item)

    if opportunities is not None and not opportunities.empty:
        for value in opportunities.get("Skills_Required", []):
            for item in split_items(value):
                if len(item) >= 2:
                    terms.add(item)

    return sorted(terms, key=lambda x: (-len(x), x))


def extract_skills(text, lexicon=None):
    """
    Lightweight skill extraction.

    It does not require spaCy. It checks the free-form text against
    the project skill lexicon and returns the skills that occur.
    """
    text = normalise(text)

    if not text:
        return []

    if lexicon is None:
        lexicon = DEFAULT_SKILLS

    found = []

    for skill in lexicon:
        if contains_phrase(text, skill):
            found.append(skill)

    # Remove shorter duplicate matches where a longer phrase contains it.
    found = sorted(set(found), key=lambda x: (-len(x), x))

    final = []

    for skill in found:
        if not any(
            skill != other and skill in other
            for other in final
        ):
            final.append(skill)

    return sorted(final)


# ============================================================
#              TEXT USED FOR SEMANTIC MATCHING
# ============================================================

def volunteer_text(volunteer):
    """
    Combine the volunteer's meaningful profile information.
    """
    fields = [
        volunteer.get("Skills", ""),
        volunteer.get("Interests", ""),
        volunteer.get("Experience", ""),
        volunteer.get("Qualification", ""),
        volunteer.get("Bio", ""),
        volunteer.get("Availability", ""),
        volunteer.get("Preferred_Mode", ""),
        volunteer.get("Location", ""),
    ]

    return " ".join(
        clean_text(value)
        for value in fields
        if clean_text(value)
    )


def opportunity_text(opportunity):
    """
    Combine the opportunity information that describes the role.
    """
    fields = [
        opportunity.get("Role_Title", ""),
        opportunity.get("Description", ""),
        opportunity.get("Skills_Required", ""),
        opportunity.get(
            "Qualification (optional but useful)",
            opportunity.get("Qualification", "")
        ),
        opportunity.get("Experience_Required", ""),
        opportunity.get("Area", ""),
        opportunity.get("Location", ""),
        opportunity.get("Mode (Online/Offline/Hybrid)", ""),
        opportunity.get("Time_Commitment", ""),
    ]

    return " ".join(
        clean_text(value)
        for value in fields
        if clean_text(value)
    )


# ============================================================
#                  SCORE COMPONENTS
# ============================================================

def semantic_similarity(volunteer, opportunity):
    """
    Sentence Transformer + cosine similarity.

    Returns a value between 0 and 1.
    """
    volunteer_text_value = volunteer_text(volunteer)
    opportunity_text_value = opportunity_text(opportunity)

    if not volunteer_text_value or not opportunity_text_value:
        return 0.0

    model = get_model()

    embeddings = model.encode(
        [volunteer_text_value, opportunity_text_value],
        normalize_embeddings=True
    )

    score = float(
        cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]
    )

    # Keep the UI score in the expected 0-100 range.
    return max(0.0, min(1.0, score))


def skill_similarity(volunteer, opportunity, lexicon=None):
    """
    Compare volunteer skills with required opportunity skills.

    Score = matched required skills / total required skills.
    """
    required = extract_skills(
        opportunity.get("Skills_Required", ""),
        lexicon
    )

    volunteer_skills_text = " ".join([
        clean_text(volunteer.get("Skills", "")),
        clean_text(volunteer.get("Experience", "")),
        clean_text(volunteer.get("Bio", "")),
    ])

    volunteer_skills = extract_skills(
        volunteer_skills_text,
        lexicon
    )

    if not required:
        return 0.5, [], []

    matched = [
        skill
        for skill in required
        if skill in volunteer_skills
    ]

    score = len(matched) / len(required)

    return score, matched, required


def interest_similarity(volunteer, opportunity, lexicon=None):
    """
    Compare volunteer interests with the opportunity's area,
    title, description and required skills.
    """
    interests = extract_skills(
        volunteer.get("Interests", ""),
        lexicon
    )

    opportunity_interest_text = " ".join([
        clean_text(opportunity.get("Role_Title", "")),
        clean_text(opportunity.get("Area", "")),
        clean_text(opportunity.get("Description", "")),
        clean_text(opportunity.get("Skills_Required", "")),
    ])

    opportunity_interests = extract_skills(
        opportunity_interest_text,
        lexicon
    )

    if not interests:
        return 0.5, []

    matched = [
        item
        for item in interests
        if item in opportunity_interests
    ]

    score = len(matched) / len(interests)

    return score, matched


def location_mode_score(volunteer, opportunity):
    """
    Combined location/mode component.

    Location:
      exact/contained locality match = 1
      Mumbai-level match = 0.5
      otherwise = 0

    Mode:
      exact preference match = 1
      Hybrid is partially compatible with Online/Offline = 0.5
      otherwise = 0
    """
    volunteer_location = normalise(volunteer.get("Location", ""))
    opportunity_location = normalise(opportunity.get("Location", ""))

    volunteer_mode = normalise(volunteer.get("Preferred_Mode", ""))
    opportunity_mode = normalise(
        opportunity.get("Mode (Online/Offline/Hybrid)", "")
    )

    location_score = 0.0

    if volunteer_location and opportunity_location:
        if (
            volunteer_location == opportunity_location
            or volunteer_location in opportunity_location
            or opportunity_location in volunteer_location
        ):
            location_score = 1.0
        elif "mumbai" in volunteer_location and "mumbai" in opportunity_location:
            location_score = 0.5

    mode_score = 0.0

    if volunteer_mode and opportunity_mode:
        if volunteer_mode == opportunity_mode:
            mode_score = 1.0
        elif volunteer_mode == "hybrid" or opportunity_mode == "hybrid":
            mode_score = 0.5

    # Equal importance inside the 5% component.
    combined = (location_score + mode_score) / 2

    reasons = []

    if location_score == 1.0:
        reasons.append("Location matches your preference")
    elif location_score == 0.5:
        reasons.append("Both are within Mumbai")

    if mode_score == 1.0:
        reasons.append("Mode matches your preference")
    elif mode_score == 0.5:
        reasons.append("Preferred mode is partly compatible")

    return combined, reasons


def availability_score(volunteer, opportunity):
    """
    Simple text-based availability compatibility.

    This is intentionally conservative because the CSV currently stores
    availability as free-form text rather than structured time slots.
    """
    volunteer_availability = normalise(
        volunteer.get("Availability", "")
    )

    opportunity_time = normalise(
        opportunity.get("Time_Commitment", "")
    )

    if not volunteer_availability or not opportunity_time:
        return 0.5

    # Look for useful broad compatibility signals.
    weekday_words = [
        "weekday",
        "weekdays",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
    ]

    weekend_words = [
        "weekend",
        "weekends",
        "saturday",
        "sunday",
    ]

    volunteer_weekday = any(
        word in volunteer_availability
        for word in weekday_words
    )

    volunteer_weekend = any(
        word in volunteer_availability
        for word in weekend_words
    )

    opportunity_lower = opportunity_time

    if "weekend" in opportunity_lower and volunteer_weekend:
        return 1.0

    if "weekday" in opportunity_lower and volunteer_weekday:
        return 1.0

    # If no structured signal is present, remain neutral rather than
    # penalising the volunteer.
    return 0.5


# ============================================================
#                    FINAL MATCH RESULT
# ============================================================

WEIGHTS = {
    "semantic": 0.60,
    "skill": 0.20,
    "interest": 0.10,
    "location_mode": 0.05,
    "availability": 0.05,
}


def match_result(volunteer, opportunity, lexicon=None):
    """
    Main AI matching function.

    Final score:
        Semantic similarity  60%
        Skill match          20%
        Interest match       10%
        Location/mode         5%
        Availability          5%
    """
    semantic = semantic_similarity(
        volunteer,
        opportunity
    )

    skill, matched_skills, required_skills = skill_similarity(
        volunteer,
        opportunity,
        lexicon
    )

    interest, matched_interests = interest_similarity(
        volunteer,
        opportunity,
        lexicon
    )

    location_mode, location_mode_reasons = location_mode_score(
        volunteer,
        opportunity
    )

    availability = availability_score(
        volunteer,
        opportunity
    )

    overall = (
        semantic * WEIGHTS["semantic"]
        + skill * WEIGHTS["skill"]
        + interest * WEIGHTS["interest"]
        + location_mode * WEIGHTS["location_mode"]
        + availability * WEIGHTS["availability"]
    )

    overall_percentage = int(round(overall * 100))

    reasons = []

    if matched_skills:
        readable = ", ".join(
            skill.title()
            for skill in matched_skills[:3]
        )
        reasons.append(
            readable + " matches required skills"
        )

    if matched_interests:
        readable = ", ".join(
            item.title()
            for item in matched_interests[:2]
        )
        reasons.append(
            readable + " matches your interests"
        )

    reasons.extend(location_mode_reasons)

    if availability == 1.0:
        reasons.append("Availability appears compatible")

    if not reasons:
        reasons.append(
            "Your profile has some semantic similarity with this opportunity"
        )

    return {
        "overall": overall_percentage,
        "semantic": round(semantic * 100, 1),
        "skill": round(skill * 100, 1),
        "interest": round(interest * 100, 1),
        "location_mode": round(location_mode * 100, 1),
        "availability": round(availability * 100, 1),
        "matched_skills": matched_skills,
        "required_skills": required_skills,
        "matched_interests": matched_interests,
        "reasons": reasons,
    }


# ============================================================
#                       RANKING
# ============================================================

def rank_opportunities(volunteer, opportunities, lexicon=None):
    """
    Calculate the AI score for every opportunity and return them
    from highest to lowest overall match.
    """
    if opportunities is None or opportunities.empty:
        return []

    results = []

    for _, opportunity in opportunities.iterrows():
        result = match_result(
            volunteer,
            opportunity,
            lexicon=lexicon
        )

        results.append(
            (
                opportunity,
                result
            )
        )

    results.sort(
        key=lambda item: item[1]["overall"],
        reverse=True
    )

    return results


# ============================================================
#                  EXPLAINABLE AI DISPLAY
# ============================================================

def render_why_this_match(match, title=None):
    """
    Display the score breakdown and human-readable reasons in Streamlit.
    """
    import streamlit as st

    if title:
        st.subheader(title)

    st.progress(
        max(0.0, min(1.0, match["overall"] / 100))
    )

    st.caption(
        "Semantic similarity "
        + str(match["semantic"])
        + "% · Skill "
        + str(match["skill"])
        + "% · Interest "
        + str(match["interest"])
        + "% · Location/mode "
        + str(match["location_mode"])
        + "% · Availability "
        + str(match["availability"])
        + "%"
    )

    if match.get("reasons"):
        st.markdown("**Why this match?**")

        for reason in match["reasons"][:4]:
            st.write("✓ " + reason)
