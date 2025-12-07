import re

def route_question(question: str) -> str:
    """
    Intelligent routing logic for patient-specific RAG.
    
    Priority-based design:
    1. Explicit condition-summary queries
    2. Next review / appointment queries
    3. Recent event / timeline queries
    4. Medication / care plan queries
    5. Notes / symptoms
    6. Fallback to 'all'
    """

    q = question.lower().strip()

    # -------------------------------
    # 1. CONDITION / SUMMARY QUERIES
    # -------------------------------
    condition_keywords = [
        "condition", "conditions", "diagnosis", "diagnoses",
        "summary", "summarize", "overview", "what is wrong", 
        "health status"
    ]
    if any(k in q for k in condition_keywords):
        return "summary"

    # --------------------------------------------
    # 2. NEXT REVIEW / DATE / FOLLOW-UP QUERIES
    # --------------------------------------------
    next_review_patterns = [
        r"next review", r"follow[- ]?up", r"next appointment",
        r"when.*review", r"when.*see", r"when.*visit"
    ]
    if any(re.search(p, q) for p in next_review_patterns):
        # These answers ALWAYS live in the CARE PLAN, not timeline.
        return "care_plan"

    # ---------------------------------
    # 3. TIMELINE / RECENT EVENTS
    # ---------------------------------
    timeline_keywords = [
        "recent", "recently", "what happened", "last event",
        "timeline", "history", "latest", "last visit"
    ]
    if any(k in q for k in timeline_keywords):
        return "timeline"

    # ---------------------------------
    # 4. MEDICATION / CARE PLAN QUERIES
    # ---------------------------------
    careplan_keywords = [
        "plan", "care plan", "treatment", "therapy",
        "med", "medication", "drug", "pill", "dose", "instruction",
        "what to do", "how to manage", "what should i do"
    ]
    if any(k in q for k in careplan_keywords):
        return "care_plan"

    # -----------------------------
    # 5. PATIENT NOTES / SYMPTOMS
    # -----------------------------
    notes_keywords = [
        "symptom", "symptoms", "feeling", "complain", "issues",
        "note", "notes", "report", "fatigue", "dizzy"
    ]
    if any(k in q for k in notes_keywords):
        return "notes"

    # -----------------------------
    # 6. FALLBACK → SEARCH ALL
    # -----------------------------
    return "all"
