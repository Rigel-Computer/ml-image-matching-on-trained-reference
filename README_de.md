# 🧠 image-matching-on-trained-reference (Deutsch)

Vergleicht Bilder mit einem vortrainierten Referenzdatensatz auf visuelle Ähnlichkeit – basierend auf ResNet18.  
Das Tool läuft vollständig offline und benötigt keine GPU.

> 🇬🇧 [English version here](README.md)

Eine ausführliche Darstellung des gesamten Projekts finden Sie unter folgendem Medium-Link:

👉 [Zu viele Igel, zu wenig Zeit? 🦔🐇🐭](https://medium.com/rigel-computer-com/zu-viele-igel-zu-wenig-zeit-f70e9cbb0600)

---

## 📝 Tagline
> Effiziente, lokale Bilderkennung durch Ähnlichkeitsvergleich mit einem trainierten Referenzdatensatz.

---

## 📌 Kurzbeschreibung
Dieses Tool analysiert neue Bilder und vergleicht sie mit einem zuvor erstellten Referenzsatz anhand von ResNet18-Feature-Vektoren.  
Liegt der Ähnlichkeitswert über einem Schwellenwert, wird das Bild als „Match“ erkannt und umbenannt.

---

## 🎯 Anwendungsfall

Entwickelt zur Erkennung von gescannten Briefumschlägen anhand ihres visuellen Erscheinungsbilds (z. B. Form, Kontrast, Layout, Anzahl).  
Es vergleicht neue Bilder mit einem bekannten Referenzsatz – gesucht wird visuelle Ähnlichkeit, **nicht** exakte Duplikate.

Typische Einsätze:
- Erkennung von Bildern mit gleichem Layout oder Aufbau
- Markierung bekannter Bildtypen (z. B. Umschläge)
- Vorsortierung und Kategorisierung gescannter Daten nach Muster

---

## 🔧 Funktionen
- Läuft vollständig offline
- Nutzt vortrainiertes ResNet18-Modell zur Merkmalsextraktion
- Durchsucht Ordner rekursiv
- Erkennt und markiert ähnliche Bilder
- Schwellenwert einstellbar

---

## 📁 Projektstruktur

```
project-root/
├── images/              # Trainings-/Referenzbilder
├── zu_pruefen/          # Zu scannende neue Bilder
├── traning.py           # Erstellt features.npy und filenames.npy
├── main.py              # Vergleicht Bilder und benennt Treffer
├── features.npy         # Gespeicherte Vektoren
└── filenames.npy        # Gespeicherte Dateinamen
```

---

## 🧪 Nutzung

1. Trainingsbilder in `images/` ablegen.
2. Trainingsskript starten:
   ```bash
   python traning.py
   ```
3. Neue Bilder in `zu_pruefen/` einfügen (Unterordner erlaubt).
4. Hauptskript ausführen:
   ```bash
   python main.py
   ```

---

## ⚙️ Abhängigkeiten

Installation aller Bibliotheken:

```bash
pip install torch torchvision scikit-learn numpy pillow
```

Alle Abhängigkeiten sind auch in `requirements.txt` aufgeführt.

---

## 🧠 Internes

- Feature-Vektoren sind 512-dimensional (ResNet18)
- Vergleich erfolgt über Cosinus-Ähnlichkeit
- Default-Schwelle: 0.89
- Treffer werden umbenannt zu `Match_<originalname>.jpg`

---

## 📌 Hinweise

- Keine Internetverbindung erforderlich
- Ideal für konsistente Bildreihen (z. B. gescannte Dokumente)
- ResNet18-Gewichte werden lokal über torchvision geladen
