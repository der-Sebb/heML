Datenschutzfreundliche künstliche Intelligenz mithilfe von homomorpher Verschlüsselung
===

Motivation
---
In der Welt der künstlichen Intelligenz beziehungsweise des Maschninellen Lernens sind Daten die Währung. Nur ist das in vielen Fällen ein Nachteil für die Individuen, wenn zum Beispiel personenbezogene Daten von Firmen ausgewertet werden. Dabei ist die Auswertung nicht immer zwangsweise ein negativer Vorgang und kann beitragen, ein Modell zu erstellen, welches einen wichtigen Beitrag in zum Beispiel der Medizin liefern kann. Ein weiterer Fall kann sein, wenn eine Firma nicht die Mittel besitzt, ein Modell mit ihren Daten zu erstellen und deswegen Hilfe von einer dritten Instanz benötigt, in Form von Recheneinheiten. Deswegen ist es von Vorteil, einen Weg zu wählen, personenbezogene oder proprietäre Daten verschlüsselt zu verarbeiten. Hierfür kann die sogenannte homomorphe Verschlüsselung verwendet werden, die es ermöglicht, Daten mit einem öffentlichen Schlüssel zu verschlüsseln und auf den verschlüsselten Daten bestimmte arithmetische Operationen auszuführen. Mit Hilfe eines privaten Schlüssels können die Ergebnisse wieder entschlüsselt werden und somit hat keiner außer dem Besitzer Zugriff zu den Daten.

Inhalt
---
Um die Verwendung von verschlüsselten Daten im Maschinellen Lernen zu zeigen, wurde die Thematik in zwei Themengebiete aufgeteilt: die Inferenz und das Training. Beide Teile werden mit Beispielen aus der Linearen Regression und den Neuronalen Netzen vorgestellt.

1. **Inferenz** 
Die Inferenz beschreibt im Maschinellen Lernen Umfeld das Auswerten von Eingabedaten durch ein bereits vorhandenes Modell. Im Kontext der verschlüsselten Daten werden diese mit einem trainierten Modell ausgewertet und können dann mit den Ergebnissen der unverschlüsselten Inferenz verglichen werden.<br />
Dies kann anhand einer kleinen [Linearen Regression Aufgabe](Inferenz/Inferenz_Lineare_Regression_iris.ipynb) als ein Einstieg betrachtet werden. Verschlüsselte Daten können aber auch durch neuronale Netze ausgewertet werden, wofür die Grundlagen durch die [Simulation eines XOR Gatters](Inferenz/Inferenz_Neuronale_Netze_XOR.ipynb) dargestellt werden. Die Erkenntnisse aus dem XOR Beispiel werden in weiteren [Inferenz Beispielen](Inferenz/Inferenz_Neuronale_Netze.ipynb) mit Neuronalen Netzen genutzt.

2. **Training**
Das Training bezieht sich im Maschinellen Lernen auf die Erstellung oder Verbesserung von Modellen, welche wiederum in der Inferenz genutzt werden können. In Verbindung mit Verschlüsselung liegen die Daten für das Training nur in verschlüsselter Form vor.<br />
Dies wurde ebenfalls zuerst als [Training einer Linearer Regression](Training/Training_Lineare_Regression_iris.ipynb) ausprobiert, um mögliche Probleme und Hindernisse Im Training zu identifizieren. Der Bogen zu den Neuronalen Netzen wurde mit dem [Training des XOR Gatters](Training/Training_Neuronale_Netze_XOR.ipynb) geschlagen. Weiterführende gibt es noch Beispiele zum [Training von Neuronalen Netze](Training/Training_Neuronale_Netze.ipynb) durch verschlüsselte Daten.

3. **Sensitiver Datensatz**
Um nun das gelernt zu übertragen auf einen relevanten Anwendungsfall, in welchen Verschlüsselung und Privatsphäre wichtig ist, wurde ein öffentlicher Datensatz über Herzversagen von Kaggle ausgewählt. Darauf wurden verschiedene [Modelle erstellt und die Inferenz sowie das Training](Neuronale_Netze_Herzversagen.ipynb) getestet.

4. **Vergleich**
Zur Evaluation werden [normale und verschlüsselte Inferenz/Training](Neuronale_Netze_Vergleich.ipynb) verglichen.

Zusammenfassung
---
In dieser Projektarbeit wurde sich mit dem Einsatz von homomorpher Verschlüsselung im Maschinellen Lernen beschäftigt. Getestet wurde Inferenz und Training anhand von Beispielen der Linearen Regression und den Neuronalen Netzen.<br />
Die Inferenz in der Linearen Regression kann ohne Probleme durchgeführt werden. Anders sieht es bei den Neuronalen Netze aus, die eine spezielle Struktur brauchen, damit alle Operationen durch die limitierten arithmetischen Möglichkeiten ausgeführt werden können. Das Training der Linearen Regression hat gezeigt, dass um die klassische iteratitive Herangehensweise im Maschniellen Lernen einzusetzen, eine Methode gebraucht wird, welche das Rauschen in verschlüsselten Daten mindert. Ansonsten ist das Berechnen passender Gewichte für das Modell eingeschränkt. Dies ist bei den neuronalen Netzen nicht anders und wird durch die Größe des Netzes zum Beispiel im Deep Learning verstärkt.<br />
Die Genauigkeit bei der Inferenz und dem Training weicht zwischen der unverschlüsselten und verschlüsselten Variante nicht voneinander ab, solange die Architektur in beiden Fällen gleich war. Das heißt, die verschlüsselte Daten haben nicht direkt einen Einfluss auf die Performance, aber sorgen dafür, dass Konzepte umgedacht werden müssen, um kompatibel zu sein. Die Laufzeit sowohl in der Inferenz als auch während des Trainings ist merklich länger als vergleichbare unverschlüsselte Modelle. Dies kann wiederum die Suche von passenden Architekturen und Hyperparametern beeinflussen, da die Evaluation lange dauert.<br />
Insgesamt kann man aber schließen, dass die homomorphe Verschlüsselung viel Potenzial hat im Maschinellen Lernen und könnte bereits für verschlüsselte Inferenz und in manchen Fällen auch für verschlüsseltes Training eingesetzt werden. Nur sind weitere Optimierungen in den Verschlüsselungsmethoden oder den Architekturen und verwendeten Methoden im Maschinellen Lernen notwendig, um es in alltäglichen Umfeldern einsatzfähig zu machen. Dazu ist auf den verschlüsselten Daten keine Datenanalyse möglich, wodurch die Vorverarbeitung bereits durch den Datenbesitzer durchgeführt werden muss.
