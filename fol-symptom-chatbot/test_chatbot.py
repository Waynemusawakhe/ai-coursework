from chatbot import infer_disease


def test_all_symptoms_present_infers_disease():
    assert infer_disease({"fever", "cough", "sore_throat"}) == ["flu"]


def test_partial_symptoms_infer_nothing():
    assert infer_disease({"fever"}) == []


def test_overlapping_symptoms_can_match_several_diseases():
    found = infer_disease({"fever", "cough", "sore_throat", "swollen_lymph_nodes"})
    assert set(found) == {"flu", "strep_throat"}
