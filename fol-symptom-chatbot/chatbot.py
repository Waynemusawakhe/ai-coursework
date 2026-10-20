"""Rule-based symptom chatbot using first-order-logic style inference (CMPG313, 2025).

Rule form: for every disease D, if all symptoms of D are present in the user's input,
then the user might have D.  (forall D: symptoms(D) is a subset of input -> possible(D))

Educational only. This is not medical advice.
"""

knowledge_base = {
    
    "flu": {"fever", "cough", "sore_throat"},
    "common_cold": {"sneezing", "runny_nose", "mild_fever"},
    "malaria": {"fever", "chills", "sweating", "headache"},
    "covid19": {"fever", "cough", "shortness_of_breath", "loss_of_taste"},
    "strep_throat": {"sore_throat", "swollen_lymph_nodes", "fever"}
    
}

advice_base = {
    "flu": [
        "Stay hydrated and rest.",
        "Take over-the-counter flu medicine.",
        "Consult a doctor if symptoms worsen."
    ],
    "common_cold": [
        "Drink warm fluids.",
        "Get plenty of rest.",
        "Use saline nasal drops."
    ],
    "malaria": [
        "Seek immediate medical attention.",
        "Take anti-malarial medications as prescribed.",
        "Avoid mosquito bites."
    ],
    "covid19": [
        "Self-isolate and monitor symptoms.",
        "Get tested and wear a mask.",
        "Seek emergency care if you have trouble breathing."
    ],
    "strep_throat": [
        "Consult a doctor for antibiotics.",
        "Drink warm liquids to soothe your throat.",
        "Avoid sharing utensils or cups."
    ]
}

def get_user_symptoms():
    print("Welcome to HealthBot! (educational demo, not medical advice)")
    print(" Enter your symptoms (comma-separated): ")
    user_input = input("Symptoms: ").lower()
    
    symptoms = {sym.strip().replace(" ", "_") for sym in user_input.split(",")}
    return symptoms


def infer_disease(user_symptoms):
    possible_diseases = []

    # FOL-style rule: a disease is inferred when all of its symptoms are present

    for disease, symptoms in knowledge_base.items():
        if symptoms.issubset(user_symptoms):
            possible_diseases.append(disease)

    return possible_diseases

def run_chatbot():
    user_symptoms = get_user_symptoms()
    diseases = infer_disease(user_symptoms)

    if diseases:
        print("\nBased on your symptoms, you might have:")
        for disease in diseases:
            print(f"- {disease.title()}")
            for tip in advice_base.get(disease, ["Please consult a doctor."]):
                print(f"  - {tip}")
    else:
        print("\nNo matching disease found.")
        print("Please consult a healthcare professional.")

if __name__ == "__main__":
    run_chatbot()
