import requests
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
import json
import re

# Definiert Datenstruktur
class NutritionState(TypedDict):
    meal_description: str
    food_items: List[Dict]  # Liste von Objekten mit mindestens 'name', 'quantity' optional 'is_basic'
    nutrition_info: List[Dict[str, str]]
    messages: List[Dict[str, str]]

# Initialisiert KI-Modell
model = init_chat_model("mistral:instruct", model_provider="ollama")

# Schritt 1, KI Extrahiert wichtigste Informationen aus dem Text
def extract_food_items(state: NutritionState) -> NutritionState:
    """
    Extrahiert Mengen- und Typinformationen über Lebensmittel aus gegebenem Text
    """
    prompt = (
        "Extrahiere alle Nahrungsmittel aus folgendem Text als Liste von Objekten.\n"
        "Jedes Objekt soll folgende Felder haben:\n"
        "- name: Name des Lebensmittels\n"
        "- quantity: Menge in Gramm, falls genau angegeben, oder eine Schätzung in Gramm falls ungenau (z.B. 1 Glas = 200g)\n"
        "Die Antwort MUSS gültiges JSON sein. KEINE Rechnungen oder Ausdrücke, NUR Zahlen als Werte.\n"
        "Wenn eine Menge eine Rechnung enthält (z.B. \"2 * 100\"), dann rechne das Ergebnis aus (also 200) und schreibe nur die Zahl als Wert, NICHT die Rechnung.\n"
        "Kontrolliere noch mal, ob du sicher gültiges JSON abgibst, das heißt KEINE Rechnungen oder Kommentare!!\n"
        "Schreibe keine Anführungszeichen um Zahlen.\n"
        "KEINE zusätzlichen Texte, Kommentare oder Erklärungen.\n"
        "Nur die reine JSON-Liste.\n\n"
        f"Text:\n{state['meal_description']}\n\n"
        "Beispiel:\n"
        "[\n"
        "  {\"name\": \"Brötchen\", \"quantity\": 200},\n"
        "  {\"name\": \"Butter\", \"quantity\": 50},\n"
        "  {\"name\": \"Ei\", \"quantity\": 50}\n"
        "]"
        "Explizit NICHT machen, NEGATIVBEISPIEL!:\n"
        "[\n"
        "  {\"name\": \"Brötchen\", \"quantity\": 10 * 200},\n"
        "  {\"name\": \"Butter\", \"quantity\": 3 * 50} // Ich schreibe hier noch einen Kommentar!,\n"
        "  {\"name\": \"Nutella\", \"quantity\": 10 * 200}\n"
        "]"
    )


    response = model.invoke([HumanMessage(content=prompt)])
    response_text = response.content
    print("antwort extract: ", response_text)
    try:
        items = json.loads(response_text)
    except json.JSONDecodeError:
        items = []
    state["food_items"] = items
    return state

# Schritt 2, Klassifizierung der Lebensmittel in Grundnahrungsmittel und industriell Verarbeitetes
def classify_food_items(state: NutritionState) -> NutritionState:
    """
    KI teilt Nahrungsmittel in Grundnahrungsmittel sowie industriell verarbeitete Lebensmittel ein. Lebensmittel werden mit True/False eingeordnet
    """

    items = state.get("food_items", [])
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        print("Warnung: 'food_items' ist keine Liste von Objekten")
        return state

    simplified_list = [{"name": i["name"]} for i in items]

    prompt = (
        "Klassifiziere **ausschließlich** die folgenden Lebensmittel anhand der folgenden Regeln:\n\n"
        "✅ is_basic = true → natürliche, unverarbeitete Lebensmittel (z. B. Ei, Apfel, Karotte, Brokkoli, Milch)\n"
        "❌ is_basic = false → verarbeitete Lebensmittel (z. B. Butter, Brötchen, Saft, Käse, Margarine)\n\n"
        "🚫 Nutze nur die Lebensmittel aus der unten stehenden Liste. Keine Beispiele hinzufügen. Keine zusätzlichen Produkte.\n"
        "✅ Antworte **nur** mit einer JSON-Liste im Format:\n"
        "[{\"name\": \"...\", \"is_basic\": true/false}, ...]\n\n"
        f"Lebensmittel:\n{json.dumps([{'name': i['name']} for i in items], ensure_ascii=False)}"
    )

    response = model.invoke([HumanMessage(content=prompt)])
    cleaned_content = response.content

    try:
        classified = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print("Fehler beim Parsen der JSON-Antwort:", e)
        classified = [{"name": item["name"], "is_basic": False} for item in simplified_list]

    # Merge is_basic zurück in Originalstruktur
    for orig_item in items:
        matched = next((cl for cl in classified if cl["name"].lower() == orig_item["name"].lower()), None)
        if matched:
            orig_item["is_basic"] = matched["is_basic"]
        else:
            orig_item["is_basic"] = False

    state["food_items"] = items
    print("Finale Items:", items)
    return state

# Schritt 3, Informationen über Lebensmittel werden gesammelt
def get_nutrition_info(state: NutritionState) -> NutritionState:
    """
    Funktion sucht Kalorieninformationen über sämtliche Lebensmittel des State heraus; dies geschieht über OpenFoodFacts. Sollte es sich um ein Grundnahrungsmittel handeln, so werden diese separat behandelt.
    """
    nutrition_results = []

    for item in state.get("food_items", []):
        product_name = item.get("name", "")
        is_basic = item.get("is_basic", False)

        # Grundnahrungsmittel wird nicht durch OpenFoodFacts gesucht
        if is_basic:
            kcal = get_kcal_for_basic_food(product_name)
            if kcal is not None:
                info = f"{product_name} (Grundnahrungsmittel): ca. {round(kcal)} kcal"
            else:
                info = f"{product_name} (Grundnahrungsmittel): Kalorieninfo nicht verfügbar"
            nutrition_results.append({
                "original": product_name,
                "info": info
            })
            continue

        # Daten durch OpenFoodFacts ermitteln
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "search_terms": product_name,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 1,
            "tagtype_0": "countries",
            "tag_contains_0": "contains",
            "tag_0": "de",
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
        except Exception as e:
            data = {"count": 0}

        if data.get("count", 0) > 0:
            product = data["products"][0]
            nutriments = product.get("nutriments", {})
            kcal = nutriments.get("energy-kcal_100g") or nutriments.get("energy-kcal")
            brand = product.get("brands", "Unbekannt")
            if kcal:
                info_str = f"{product_name} ({brand}): {kcal} kcal pro 100g"
            else:
                info_str = f"{product_name} ({brand}): Keine Kalorienangabe gefunden"
        else:
            brand = "Unbekannt"
            info_str = f"{product_name} ({brand}): Kein Produkt gefunden"

        nutrition_results.append({
            "original": f"{product_name} von {brand}",
            "info": info_str
        })

    state["nutrition_info"] = nutrition_results
    return state

# Schritt 4, gesammelte Informationen werden zurückgegeben
def summarize_nutrition(state: NutritionState) -> NutritionState:
    summary = "🧾 Hier ist deine Nährwertanalyse:\n\n"
    total_kcal = 0

    # food_items nach Namen in ein Dict für schnellen Zugriff der Menge
    quantities = {item["name"].lower(): item.get("quantity", 0) for item in state.get("food_items", [])}

    for entry in state.get("nutrition_info", []):
        info_text = entry.get("info", "Lebensmittel")
        kcal_per_100g = None
        kcal_total = None

        # kcal pro 100g extrahieren (Zahl vor "kcal")
        words = info_text.split()
        for i, word in enumerate(words):
            if "kcal" in word.lower():
                try:
                    kcal_per_100g = float(words[i-1])
                except:
                    pass

        # Menge zuordnen anhand Name im original Text
        name_lower = entry.get("original", "").lower()
        quantity = 0
        for food_name, qty in quantities.items():
            if food_name in name_lower:
                quantity = qty
                break

        if kcal_per_100g is not None and quantity > 0:
            kcal_total = (quantity / 100) * kcal_per_100g
            total_kcal += kcal_total
            kcal_display = round(kcal_total)
            summary += f"- {info_text}: insgesamt also {kcal_display} kcal ({quantity}g)\n"
        else:
            summary += f"- {info_text}: kcal-Wert oder Menge unbekannt\n"

    summary += f"\n👉 Geschätzte Gesamtkalorien: **{round(total_kcal)} kcal**"
    state.setdefault("messages", []).append({"type": "summary", "content": summary})
    return state

def get_kcal_for_basic_food(product_name: str):
    """
    Hilfsfunktion, gibt Nährwert von Grundnahrungsmitteln anhand einer Liste zurück
    """
    # Auswahl an Grundnahrungsmitteln, sollte künftig durch Datenbankanbindung ausgetauscht werden
    basic_foods_nutrition = {
        "reis": 130,
        "weizenmehl": 340,
        "haferflocken": 380,
        "gerste": 354,
        "dinkel": 338,
        "quinoa": 368,
        "hirse": 378,
        "linsen": 116,
        "kichererbsen": 164,
        "schwarze bohnen": 341,
        "grüne bohnen": 31,
        "brokkoli": 34,
        "spinat": 23,
        "karotten": 41,
        "tomaten": 18,
        "gurke": 16,
        "paprika": 31,
        "zwiebeln": 40,
        "knoblauch": 149,
        "süßkartoffeln": 86,
        "kartoffeln": 77,
        "kohl": 25,
        "blumenkohl": 25,
        "salat": 15,
        "grünkohl": 49,
        "apfel": 52,
        "banane": 89,
        "birne": 57,
        "orange": 47,
        "mandarine": 53,
        "erdbeeren": 32,
        "himbeeren": 52,
        "blaubeeren": 57,
        "ei": 137,
        "trauben": 69,
        "pfirsich": 39,
        "kirschen": 50,
        "avocado": 160,
        "walnüsse": 654,
        "mandeln": 575,
        "cashewkerne": 553,
        "sonnenblumenkerne": 584,
        "chia-samen": 486,
        "leinsamen": 534,
        "hefe": 105,
        "sojabohnen": 173,
        "tofu": 76,
        "edamame": 121,
        "mais": 86,
        "erbsen": 81,
        "sellerie": 16,
        "fenchel": 31,
        "rote beete": 43,
        "rüben": 43,
        "kürbis": 26,
        "artischocken": 47,
        "spargel": 20,
        "aubergine": 25,
        "zucchini": 17,
    }
    kcal_per_100g = basic_foods_nutrition.get(product_name.lower())
    if kcal_per_100g is None:
        return None  # Kein Wert vorhanden
    return kcal_per_100g


# Graph definieren
graph = StateGraph(NutritionState)
graph.add_node("extract_food", extract_food_items)
graph.add_node("classify", classify_food_items)
graph.add_node("get_nutrition", get_nutrition_info)
graph.add_node("summarize", summarize_nutrition)

graph.add_edge(START, "extract_food")
graph.add_edge("extract_food", "classify")
graph.add_edge("classify", "get_nutrition")
graph.add_edge("get_nutrition", "summarize")
graph.add_edge("summarize", END)

compiled = graph.compile()

# Inital State, Beginn des Workflows
initial_state: NutritionState = {
    # Man muss alles in Gramm angeben, sonst bekommt er Probleme!
    "meal_description": "Heute habe ich 200g Nutella und dazu 60g Ja! Schokolade vernascht. Danach hatte ich noch 100 Gramm Apfel",
    "food_items": [],
    "nutrition_info": [],
    "messages": [],
}

result = compiled.invoke(initial_state)

# Rückgabe des Ergebnisses
print(result["messages"][-1]["content"])
