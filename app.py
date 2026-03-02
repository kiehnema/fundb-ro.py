import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import os
from datetime import datetime

st.set_page_config(page_title="Fundbüro App", layout="wide")

st.title("🔎 Digitales Fundbüro")

# -------------------------
# Ordner erstellen
# -------------------------
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# -------------------------
# Datenbank verbinden
# -------------------------
conn = sqlite3.connect("fundbuero.db", check_same_thread=False)
c = conn.cursor()

# Tabelle erstellen
c.execute("""
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    category TEXT,
    confidence REAL,
    date TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER,
    email TEXT,
    date TEXT
)
""")

conn.commit()

# -------------------------
# Modell laden
# -------------------------
@st.cache_resource
def load_model_and_labels():
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_model_and_labels()

# -------------------------
# Navigation
# -------------------------
menu = st.sidebar.selectbox("Menü", ["Fund hochladen", "Funde durchsuchen"])

# =====================================================
# 1️⃣ FUND HOCHLADEN
# =====================================================
if menu == "Fund hochladen":

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        # Bild vorbereiten
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Vorhersage
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])

        st.success(f"Erkannt: {class_name}")
        st.write(f"Sicherheit: {round(confidence_score*100,2)} %")

        if st.button("Im Fundbüro speichern"):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"uploads/{timestamp}_{uploaded_file.name}"
            image.save(filename)

            c.execute("INSERT INTO items (filename, category, confidence, date) VALUES (?, ?, ?, ?)",
                      (filename, class_name, confidence_score, datetime.now()))
            conn.commit()

            st.success("Fund wurde gespeichert!")

# =====================================================
# 2️⃣ FUNDE DURCHSUCHEN
# =====================================================
if menu == "Funde durchsuchen":

    categories = ["Alle", "Flasche", "Schuhe", "Pullover"]
    selected_category = st.selectbox("Kategorie filtern", categories)

    if selected_category == "Alle":
        c.execute("SELECT * FROM items ORDER BY date DESC")
    else:
        c.execute("SELECT * FROM items WHERE category LIKE ? ORDER BY date DESC",
                  (f"%{selected_category}%",))

    items = c.fetchall()

    if items:
        for item in items:
            item_id, filename, category, confidence, date = item

            st.image(filename, width=200)
            st.write(f"**Kategorie:** {category}")
            st.write(f"**Gefunden am:** {date}")
            st.write(f"**Sicherheit:** {round(confidence*100,2)} %")

            email = st.text_input(f"E-Mail eingeben falls es dir gehört (ID {item_id})")

            if st.button(f"Anspruch senden für ID {item_id}"):
                if email:
                    c.execute("INSERT INTO claims (item_id, email, date) VALUES (?, ?, ?)",
                              (item_id, email, datetime.now()))
                    conn.commit()
                    st.success("Anspruch wurde registriert! Das Fundbüro wird dich kontaktieren.")
                else:
                    st.error("Bitte eine gültige E-Mail eingeben.")

            st.markdown("---")
    else:
        st.info("Keine Funde vorhanden.")
