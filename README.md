Datenschutzfreundliche künstliche Intelligenz mithilfe von homomorpher Verschlüsselung
===

Motivation
---
Ist wichtig und so weiter

Inhalt
---
Um die Verwendung von verschlüsselten Daten im Maschinellen Lernen zu zeigen wurde die Thematik in zwei Themengebiete aufgeteilt: die Inferenz und das Training. Beide Teile werden mit Beispielen aus der Linearen Regression und den Neuronalen Netzen vorgestellt.

1. **Inferenz** 
Die Inferenz beschreibt im Maschinellen Lernen Umfeld das Auswerten von Eingabedaten durch ein bereits vorhandenes Modell. Im Kontext der verschlüsselten Daten werden diese mit einem trainierten Modell ausgewertet und können, dann mit den Ergebnis der unverschlüsselten Inferenz verglichen werden.<br />
Dies kann anhand einer kleinen [Linearen Regression Aufgabe](Inferenz/Inferenz_Lineare_Regression_iris.ipynb) als ein Einstieg betrachtet werden. Verschlüsselte Daten können aber auch durch neuronale Netze ausgewertet werden, wofür die Grundlagen durch die [Simulation eines XOR Gatters](Inferenz/Inferenz_Neuronale_Netze_XOR.ipynb) dargestellt werden. Die Erkenntnisse aus dem XOR Beispiel werden in weiteren [Inferenz Beispielen](Inferenz/Inferenz_Neuronale_Netze.ipynb) mit Neuronalen Netzen genutzt.

2. **Training**
Das Training bezieht sich im Maschinellen Lernen auf die Erstellung oder Verbesserung von Modellen, welche wiederum in der Inferenz genutzt werden können. In Verbindung mit Verschlüsselung liegen die Daten für das Training nur in verschlüsselter Form vor.<br />
Dies wurde ebenfalls zuerst als [Training einer Linearer Regression](Training/Training_Lineare_Regression_iris.ipynb) ausprobiert, um mögliche Probleme und Hindernisse Im Training zu identifizieren. Der Bogen zu den Neuronalen Netzen wurde mit dem [Training des XOR Gatters](Training/Training_Neuronale_Netze_XOR.ipynb) geschlagen. Weiterführende Beispiel gibt es noch weitere Beispiele zum [Training von Neuronalen Netze](Training/Training_Neuronale_Netze.ipynb) durch verschlüsselte Daten.

3. **Sensitiver Datensatz**
Um nun das gelernt zu übertragen auf einen relevanten Anwendungsfall, in welchem Verschlüsselung und Privatsphäre wichtig ist, wurde ein öffentlicher Herzversagen Datensatz von Kaggle ausgewählt. Darauf wurden verschiedene [Modelle erstellt und die Inferenz sowie das Training](Neuronale_Netze_Herzversagen.ipynb) getestet.

4. **Vergleich**
Zur Evaluation werden noch [normale und verschlüsselte Inferenz/Training](Neuronale_Netze_Vergleich.ipynb) verglichen.

Fazit
---
Ja ist toll gell