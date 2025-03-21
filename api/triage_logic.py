# api/triage_logic.py

def triage_questions():
    """
    Return a list of possible questions for the user based on
    the mnemonics (WWHAM, ASMETHOD, ENCORE, SIT DOWN SIR).
    This can be part of a conversation flow if building a chatbot.
    """
    return [
        "What are the symptoms?",
        "How long have they been present?",
        "Are you taking any other medications?",
        "Where exactly is the issue located?",
        # ...
    ]

def generate_triage_recommendation(age, appearance, symptoms):
    """
    Simplified logic: 
    - If serious keywords are found, refer to doctor.
    - Otherwise, return an OTC suggestion placeholder.
    """
    danger_signs = ["chest pain", "severe headache", "unconscious", "bleeding"]
    if any(ds in symptoms.lower() for ds in danger_signs):
        return "Referral to doctor recommended due to danger symptoms."
    else:
        # This is a placeholder; real logic would be more detailed
        return f"OTC recommendation placeholder for symptoms: {symptoms}"
