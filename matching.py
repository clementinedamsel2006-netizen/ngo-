import re
from functools import lru_cache

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
    # Applicant ranking is more skills-focused than general discovery.
    # Semantic similarity remains important, while explicit required skills
    # receive the strongest non-semantic weight.
    "semantic": 0.40,
    "skill": 0.25,
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

    applicant_reasons = list(base.get("reasons", []))

    if experience == 1.0:
        applicant_reasons.append("Experience appears compatible with the requirement")
    elif experience == 0.0 and normalise(opportunity.get("Experience_Required", "")):
        applicant_reasons.append("Experience requirement is not clearly met")

    if qualification == 1.0:
        applicant_reasons.append("Qualification appears compatible with the requirement")
    elif qualification == 0.0 and normalise(
        opportunity.get(
            "Qualification (optional but useful)",
            opportunity.get("Qualification", "")
        )
    ):
        applicant_reasons.append("Qualification does not clearly match the requirement")

    matched_count = len(result["skill_gap"].get("matched", []))
    required_count = len(result["skill_gap"].get("required", []))
    if required_count:
        applicant_reasons.append(
            str(matched_count)
            + " of "
            + str(required_count)
            + " required skills matched"
        )

    result["reasons"] = applicant_reasons[:6]

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

    gap = skill_gap_analysis(
        volunteer,
        opportunity,
        lexicon=lexicon
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
        "skill_gap": gap,
        "rows": {
            "semantic": round(semantic * 100, 1),
            "skill": round(skill * 100, 1),
            "interest": round(interest * 100, 1),
            "location_mode": round(location_mode * 100, 1),
            "availability": round(availability * 100, 1),
        },
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
    """Display the final score explanation without a redundant component breakdown."""
    import streamlit as st

    if title:
        st.subheader(title)

    if match.get("reasons"):
        for reason in match["reasons"][:4]:
            st.write("✓ " + reason)
    else:
        st.caption(
            "No specific matching reason was identified from the available "
            "profile information."
        )
