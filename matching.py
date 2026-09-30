import re
from functools import lru_cache

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

SKILL_ALIASES = {
    "teach": "teaching",
    "teaches": "teaching",
    "taught": "teaching",
    "mentored": "mentoring",
    "mentor": "mentoring",
    "coded": "coding",
    "programmed": "programming",
    "designing": "graphic design",
    "designer": "graphic design",
    "photograph": "photography",
    "photographed": "photography",
    "analyze": "data analysis",
    "analysing": "data analysis",
    "analyzing": "data analysis",
}


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

    # Handle a small set of transparent word-form aliases so that
    # "teach" can still be recognised as the stored skill "teaching".
    for phrase, canonical in SKILL_ALIASES.items():
        if contains_phrase(text, phrase) and canonical in lexicon:
            found.append(canonical)

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
#                 AI TEXT UTILITIES
# ============================================================

def _embedding_similarity(text_a, text_b):
    """Semantic similarity for two arbitrary text snippets."""
    a = clean_text(text_a)
    b = clean_text(text_b)
    if not a or not b:
        return 0.0

    model = get_model()
    embeddings = model.encode(
        [a, b],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return max(0.0, min(1.0, score))


def _calibrate_semantic(score):
    """Map raw cosine similarity to a more useful recommendation signal.

    The transformation keeps ordering intact while separating weak,
    moderate and strong semantic matches more clearly.
    """
    score = max(0.0, min(1.0, float(score)))
    return 1.0 / (1.0 + math.exp(-8.0 * (score - 0.45)))


def _text_tokens(value):
    text = normalise(value)
    return {token for token in re.findall(r"[a-z0-9+#]+", text) if len(token) > 2}


def _token_overlap(text_a, text_b):
    a = _text_tokens(text_a)
    b = _text_tokens(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def interest_semantic_score(volunteer, opportunity):
    """Semantic interest/cause compatibility independent of the skill lexicon."""
    volunteer_interest = clean_text(volunteer.get("Interests", ""))
    opportunity_interest = " ".join([
        clean_text(opportunity.get("Area", "")),
        clean_text(opportunity.get("Role_Title", "")),
        clean_text(opportunity.get("Description", "")),
    ]).strip()

    if not volunteer_interest or not opportunity_interest:
        return 0.5, []

    semantic = _embedding_similarity(volunteer_interest, opportunity_interest)
    overlap = _token_overlap(volunteer_interest, opportunity_interest)
    score = (semantic * 0.75) + (overlap * 0.25)

    reasons = []
    if score >= 0.72:
        reasons.append("Your interests strongly align with this cause")
    elif score >= 0.55:
        reasons.append("Your interests align with this opportunity")

    return max(0.0, min(1.0, score)), reasons


def profile_completeness(volunteer):
    """Estimate how much useful profile information is available to the AI."""
    fields = [
        "Skills", "Interests", "Experience", "Qualification", "Bio",
        "Availability", "Preferred_Mode", "Location"
    ]
    filled = sum(1 for field in fields if clean_text(volunteer.get(field, "")))
    return filled / len(fields)

# ============================================================
#                  SCORE COMPONENTS
# ============================================================

def semantic_similarity(volunteer, opportunity):
    """Sentence Transformer semantic compatibility, calibrated for ranking."""
    raw = _embedding_similarity(
        volunteer_text(volunteer),
        opportunity_text(opportunity)
    )
    return _calibrate_semantic(raw)


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
    """Semantic + lexical cause/interest compatibility."""
    return interest_semantic_score(volunteer, opportunity)


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
    """Estimate compatibility of free-form availability and time commitment."""
    volunteer_availability = normalise(volunteer.get("Availability", ""))
    opportunity_time = normalise(opportunity.get("Time_Commitment", ""))

    if not volunteer_availability or not opportunity_time:
        return 0.5

    weekend = {"weekend", "weekends", "saturday", "sunday"}
    weekdays = {"weekday", "weekdays", "monday", "tuesday", "wednesday", "thursday", "friday"}

    v_weekend = any(w in volunteer_availability for w in weekend)
    v_weekday = any(w in volunteer_availability for w in weekdays)
    o_weekend = any(w in opportunity_time for w in weekend)
    o_weekday = any(w in opportunity_time for w in weekdays)

    day_score = 0.5
    if o_weekend and v_weekend:
        day_score = 1.0
    elif o_weekday and v_weekday:
        day_score = 1.0
    elif (o_weekend and v_weekday) or (o_weekday and v_weekend):
        day_score = 0.0

    # Compare explicit hour counts when both sides state one.
    v_hours = re.findall(r"(?:up to|maximum|max|for)?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", volunteer_availability)
    o_hours = re.findall(r"(?:about|around|approximately|up to|for)?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", opportunity_time)

    hour_score = 0.5
    if v_hours and o_hours:
        vh = max(float(x) for x in v_hours)
        oh = max(float(x) for x in o_hours)
        if vh >= oh:
            hour_score = 1.0
        elif vh >= oh * 0.75:
            hour_score = 0.6
        else:
            hour_score = 0.0

    return (day_score * 0.6) + (hour_score * 0.4)


# ============================================================
#                 SKILL-GAP ANALYSIS
# ============================================================

def skill_gap_analysis(volunteer, opportunity, lexicon=None):
    """
    Compare required skills with skills detected in the volunteer's
    profile. This is a separate explainable feature from the overall
    recommendation score.
    """
    if lexicon is None:
        lexicon = DEFAULT_SKILLS

    required = extract_skills(
        opportunity.get("Skills_Required", ""),
        lexicon
    )

    volunteer_text_value = " ".join([
        clean_text(volunteer.get("Skills", "")),
        clean_text(volunteer.get("Experience", "")),
        clean_text(volunteer.get("Bio", "")),
    ])

    detected = extract_skills(volunteer_text_value, lexicon)

    matched = [skill for skill in required if skill in detected]
    missing = [skill for skill in required if skill not in detected]

    coverage = (
        len(matched) / len(required)
        if required else 1.0
    )

    return {
        "required": required,
        "detected": detected,
        "matched": matched,
        "missing": missing,
        "coverage": round(coverage * 100, 1),
    }


# ============================================================
#              APPLICANT RANKING / DECISION SUPPORT
# ============================================================

def experience_score(volunteer, opportunity):
    """Transparent experience-fit score based on simple text signals."""
    required = normalise(opportunity.get("Experience_Required", ""))
    actual = normalise(volunteer.get("Experience", ""))

    if not required or required in {"not specified", "none", "na", "n/a"}:
        return 0.5

    if not actual:
        return 0.0

    # Exact phrase overlap is strong evidence in the small CSV-based system.
    if required in actual or actual in required:
        return 1.0

    # Look for common experience-level signals.
    levels = [
        ("advanced", ["advanced", "expert", "senior"]),
        ("intermediate", ["intermediate", "1-3 years", "2-3 years", "3 years"]),
        ("beginner", ["beginner", "basic", "fresher", "student", "entry"]),
    ]

    for _, words in levels:
        if any(word in required for word in words):
            if any(word in actual for word in words):
                return 1.0

    # Having some stated experience is better than an empty profile,
    # but it is not treated as a full match.
    return 0.5


def qualification_score(volunteer, opportunity):
    """Compare qualification text conservatively; unknown stays neutral."""
    required = normalise(
        opportunity.get(
            "Qualification (optional but useful)",
            opportunity.get("Qualification", "")
        )
    )
    actual = normalise(volunteer.get("Qualification", ""))

    if not required or required in {"not specified", "optional", "none", "na", "n/a"}:
        return 0.5

    if not actual:
        return 0.0

    if required in actual or actual in required:
        return 1.0

    # Token overlap prevents a completely unrelated qualification from
    # receiving a full score while keeping this rule explainable.
    required_tokens = set(required.split())
    actual_tokens = set(actual.split())

    if not required_tokens:
        return 0.5

    overlap = len(required_tokens & actual_tokens) / len(required_tokens)

    if overlap >= 0.5:
        return 1.0
    if overlap > 0:
        return 0.5
    return 0.0


APPLICANT_WEIGHTS = {
    "semantic": 0.45,
    "skill": 0.20,
    "interest": 0.10,
    "experience": 0.10,
    "qualification": 0.05,
    "location_mode": 0.05,
    "availability": 0.05,
}


def applicant_match_result(volunteer, opportunity, lexicon=None):
    """
    Full NGO-side applicant ranking score.

    This is intentionally separate from volunteer opportunity ranking:
    NGOs are evaluating an applicant against one specific opportunity,
    so experience and qualification are given explicit components.
    """
    base = match_result(
        volunteer,
        opportunity,
        lexicon=lexicon
    )

    experience = experience_score(volunteer, opportunity)
    qualification = qualification_score(volunteer, opportunity)

    overall = (
        (base["semantic"] / 100) * APPLICANT_WEIGHTS["semantic"]
        + (base["skill"] / 100) * APPLICANT_WEIGHTS["skill"]
        + (base["interest"] / 100) * APPLICANT_WEIGHTS["interest"]
        + experience * APPLICANT_WEIGHTS["experience"]
        + qualification * APPLICANT_WEIGHTS["qualification"]
        + (base["location_mode"] / 100) * APPLICANT_WEIGHTS["location_mode"]
        + (base["availability"] / 100) * APPLICANT_WEIGHTS["availability"]
    )

    result = dict(base)
    result.update({
        "experience": round(experience * 100, 1),
        "qualification": round(qualification * 100, 1),
        "overall": int(round(overall * 100)),
        "skill_gap": skill_gap_analysis(
            volunteer,
            opportunity,
            lexicon=lexicon
        ),
    })

    result["rows"] = {
        "semantic": result["semantic"],
        "skill": result["skill"],
        "interest": result["interest"],
        "experience": result["experience"],
        "qualification": result["qualification"],
        "location_mode": result["location_mode"],
        "availability": result["availability"],
    }

    return result


def rank_applicants(applicants, opportunity, lexicon=None):
    """
    Rank volunteer rows for one opportunity.

    Returns a list of (volunteer_row, application_row, result) tuples.
    """
    if applicants is None or len(applicants) == 0:
        return []

    ranked = []

    for item in applicants:
        if isinstance(item, tuple) and len(item) == 2:
            volunteer, application = item
        else:
            volunteer = item
            application = None

        result = applicant_match_result(
            volunteer,
            opportunity,
            lexicon=lexicon
        )

        ranked.append((volunteer, application, result))

    ranked.sort(
        key=lambda item: item[2]["overall"],
        reverse=True
    )

    return ranked


# ============================================================
#                K-MEANS VOLUNTEER CLUSTERS
# ============================================================

def cluster_volunteers(volunteers, n_clusters=3, lexicon=None):
    """
    Discover volunteer groups from Sentence Transformer embeddings.

    Returns a dataframe with Cluster and Cluster_Label columns plus
    summary information. K-Means is used for unsupervised learning;
    labels are generated afterward from the most common extracted skills.
    """
    if volunteers is None or volunteers.empty:
        return pd.DataFrame(), {}

    rows = volunteers.copy().reset_index(drop=False)
    rows["_text"] = rows.apply(volunteer_text, axis=1)
    rows = rows[rows["_text"].astype(str).str.strip() != ""].copy()

    if len(rows) < 2:
        return pd.DataFrame(), {}

    k = max(2, min(int(n_clusters), len(rows)))

    model = get_model()
    embeddings = model.encode(
        rows["_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(embeddings)

    rows["Cluster"] = labels

    if len(set(labels)) > 1 and len(rows) > k:
        score = float(silhouette_score(embeddings, labels))
    else:
        score = None

    if lexicon is None:
        lexicon = DEFAULT_SKILLS

    cluster_names = {}

    for cluster_id in sorted(rows["Cluster"].unique()):
        cluster_rows = rows[rows["Cluster"] == cluster_id]
        counts = {}

        for _, row in cluster_rows.iterrows():
            text = " ".join([
                clean_text(row.get("Skills", "")),
                clean_text(row.get("Interests", "")),
                clean_text(row.get("Bio", "")),
                clean_text(row.get("Experience", "")),
            ])

            for skill in extract_skills(text, lexicon):
                counts[skill] = counts.get(skill, 0) + 1

        top_skills = [
            skill for skill, _ in sorted(
                counts.items(),
                key=lambda pair: (-pair[1], pair[0])
            )[:3]
        ]

        if top_skills:
            label = " & ".join(skill.title() for skill in top_skills)
        else:
            label = "Volunteer Group " + str(cluster_id + 1)

        cluster_names[cluster_id] = label

    rows["Cluster_Label"] = rows["Cluster"].map(cluster_names)

    summary = {
        "n_clusters": k,
        "silhouette_score": score,
        "cluster_names": cluster_names,
        "cluster_counts": rows["Cluster_Label"].value_counts().to_dict(),
    }

    return rows, summary


# Volunteer-side weights intentionally consider the full profile.
WEIGHTS = {
    "semantic": 0.40,
    "skill": 0.20,
    "interest": 0.12,
    "experience": 0.08,
    "qualification": 0.05,
    "location_mode": 0.08,
    "availability": 0.07,
}


def match_result(volunteer, opportunity, lexicon=None):
    """Comprehensive AI-assisted volunteer ↔ opportunity match."""
    if lexicon is None:
        lexicon = DEFAULT_SKILLS

    semantic = semantic_similarity(volunteer, opportunity)
    skill, matched_skills, required_skills = skill_similarity(
        volunteer, opportunity, lexicon
    )
    interest, interest_reasons = interest_similarity(
        volunteer, opportunity, lexicon
    )
    experience = experience_score(volunteer, opportunity)
    qualification = qualification_score(volunteer, opportunity)
    location_mode, location_reasons = location_mode_score(
        volunteer, opportunity
    )
    availability = availability_score(volunteer, opportunity)

    overall = (
        semantic * WEIGHTS["semantic"]
        + skill * WEIGHTS["skill"]
        + interest * WEIGHTS["interest"]
        + experience * WEIGHTS["experience"]
        + qualification * WEIGHTS["qualification"]
        + location_mode * WEIGHTS["location_mode"]
        + availability * WEIGHTS["availability"]
    )

    reasons = []
    if semantic >= 0.78:
        reasons.append("Your profile strongly matches the role's overall requirements")
    elif semantic >= 0.60:
        reasons.append("Your profile is semantically aligned with this role")

    if matched_skills:
        reasons.append(
            "Matching skills: " + ", ".join(skill.title() for skill in matched_skills[:4])
        )

    reasons.extend(interest_reasons)
    reasons.extend(location_reasons)

    if experience >= 0.9:
        reasons.append("Your experience fits the stated experience requirement")
    if qualification >= 0.9:
        reasons.append("Your qualification aligns with the opportunity")
    if availability >= 0.9:
        reasons.append("Your stated availability fits the time commitment")

    gap = skill_gap_analysis(volunteer, opportunity, lexicon=lexicon)

    return {
        "overall": int(round(overall * 100)),
        "semantic": round(semantic * 100, 1),
        "skill": round(skill * 100, 1),
        "interest": round(interest * 100, 1),
        "experience": round(experience * 100, 1),
        "qualification": round(qualification * 100, 1),
        "location_mode": round(location_mode * 100, 1),
        "availability": round(availability * 100, 1),
        "reasons": reasons[:6],
        "skill_gap": gap,
        "matched_skills": matched_skills,
        "required_skills": required_skills,
        "profile_completeness": round(profile_completeness(volunteer) * 100, 1),
        "rows": {
            "semantic": round(semantic * 100, 1),
            "skill": round(skill * 100, 1),
            "interest": round(interest * 100, 1),
            "experience": round(experience * 100, 1),
            "qualification": round(qualification * 100, 1),
            "location_mode": round(location_mode * 100, 1),
            "availability": round(availability * 100, 1),
        },
    }



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

    caption_parts = [
        "Semantic similarity " + str(match["semantic"]) + "%",
        "Skill " + str(match["skill"]) + "%",
        "Interest " + str(match["interest"]) + "%",
    ]

    if "experience" in match:
        caption_parts.append("Experience " + str(match["experience"]) + "%")

    if "qualification" in match:
        caption_parts.append("Qualification " + str(match["qualification"]) + "%")

    caption_parts.extend([
        "Location/mode " + str(match["location_mode"]) + "%",
        "Availability " + str(match["availability"]) + "%",
    ])

    st.caption(" · ".join(caption_parts))

    if match.get("reasons"):
        st.markdown("**Why this match?**")

        for reason in match["reasons"][:4]:
            st.write("✓ " + reason)
