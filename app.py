import streamlit as st
from langgraph.graph import StateGraph
from main import compiled 

# Titel der App
st.title("🍽️ KI-Nährwertanalyse Bot")

# Texteingabe
user_input = st.text_area("Beschreibe deine Mahlzeit:", 
    placeholder="z. B. Heute habe ich 2 Brötchen mit Butter, Käse und 1 Glas Orangensaft gegessen")

# Button zur Analyse
if st.button("Analyse starten"):
    if user_input.strip() == "":
        st.warning("Bitte gib eine Beschreibung deiner Mahlzeit ein.")
    else:
        # Initial State
        state = {
            "meal_description": user_input,
            "food_items": [],
            "nutrition_info": [],
            "messages": [],
        }

        # Workflow ausführen
        with st.spinner("Nährwertanalyse läuft..."):
            result = compiled.invoke(state)
        
        # Ausgabe anzeigen
        st.success("Analyse abgeschlossen:")
        st.markdown(result["messages"][-1]["content"])
