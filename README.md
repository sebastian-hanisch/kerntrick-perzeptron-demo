# Kerntrick im Perceptron: ein Skalarprodukt austauschen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kerntrick-perzeptron-demo.streamlit.app/)** -->

Erste Demo der Reihe „Kernmethoden: Skalarprodukte tauschen“ in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ und Wurzel ihres Astes „Trennen“.
Aufgehoben wird eine Annahme der Vorgängerin `perceptron-demo`: *das Perceptron trennt nur linear*. Die Nachfolgerin der Reihe fragt, welche der vielen trennenden Flächen die beste ist (harte Marge und duales QP).
Vehikel sind vier ausgedachte 2D-Beispiele: vier Punkte (XOR), zwei Ringe, Kunden im Liefergebiet, zwei Monde. Es gibt keine reale Datenquelle; in den gelesenen Quellen gab es keine Messung, ob Kernmethoden
in Logistik, Lager oder Fertigung bessere Entscheidungen liefern. Gezählt werden Korrekturen, Epochen, Zufalls-Seeds und Multiplikationen; was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet.

## Warum dieses Problem

Das Perceptron rechnet nur mit Skalarprodukten. Wer sie durch eine Kernfunktion ersetzt (der **Kerntrick**), trennt ohne Änderung der Regel auch Mengen, die eine Gerade nicht trennt, sofern der Kern im Merkmalsraum trennt (Monde: Polynom 2 nicht, Polynom 3 und Gauß-Kern ja). Die Kernfrage der Demo ist die zweite:
woran merkt man, dass es klappt? Bei kleiner Bandbreite (σ = 0,05 und 0,1) sagt „konvergiert“ nicht, ob die Etiketten Signal oder Rauschen sind: auch mit Zufallsetiketten konvergieren 59 und 58 von 60 Läufen innerhalb von 3 000 Epochen; bei σ = 0,5 trennt die Meldung mit 3 000 Epochen (0 von 60 mit Zufallsetiketten gegen 60 von 60 mit wahren Etiketten, Ringe, n = 100). Bei einem Kern, der kein Skalarprodukt ist, kommt vom Lauf selbst keine Warnung (die Seite zeigt den kleinsten Eigenwert der Gram-Matrix).
In der Reihe kehrt dieses Motiv wieder („der Löser meldet nichts“); hier ist es die schwache Form: die Meldung ist da, sie sagt nur weniger, als man liest. Hauptbeleg ist Experiment 5.

## Befunde und Korrekturen gegenüber dem Plan

Die Zahlen des Plans stammen aus Vormessungen mit dem Zufallsstrom von NumPy; die Seite rechnet mit SplitMix64 (`ktp_rng.py`). Alle Zahlen unten sind mit dem Strom der Seite neu gemessen (Satz A: Zufalls-Seeds 1 bis 30, Satz B: 1001 bis 1030, frisch). Für die Monde, das Liefergebiet mit n = 200 und die vertauschten Etiketten gelten je Satz 20 Zufalls-Seeds (ab 1 bzw. ab 1001), das Sigmoid nutzt nur Satz A.
Abweichungen vom Plan:

- **Marge der gefundenen Trennung** im Median 0,167 und 0,194 der größten (Ringe, Gauß σ = 1, zwei Sätze, Maß mit freiem b); der Plan nannte 0,181 und 0,178. Gleiche Größenordnung, die Zahl ist ein Band über Zufalls-Seeds, kein Punktwert.
- **Monde:** mit dem eigenen Erzeuger (gleichverteilte Winkel, beschränktes Rauschen) konvergiert Polynom 2 je Rauschstufe (0 und 0,1) in 0 von 40 Läufen (20 je Satz, zwei Sätze); eine Prüfung mit einem anderen Erzeuger hatte 1 von 20 gefunden. Die Zeile hängt am Erzeuger.
- **Zufallsetiketten, σ = 0,5:** mit der Messgrenze 50 000 Epochen und 300 000 Korrekturen konvergieren 12 von 60 Läufen; die Korrekturgrenze von 300 000 greift in 41 von 60 Läufen. Der Plan nannte nur die Grenze 3 000 Epochen (dort 0 von 60 bei σ = 0,5).
  Bei σ = 0,25 liegt der größte Wert bei 166 997 Korrekturen (Plan: 197 098), die Korrekturgrenze bindet dort nicht.
- **10 % vertauschte Etiketten im Liefergebiet (n = 200):** Testfehler gegen die wahre Klasse 0,158 bis 0,172 in den Gauß-Zeilen (Plan: 0,148 bis 0,164); der Negativbefund bleibt: die Meldung trennt gute und schlechte Läufe nicht.
- **Korrekturen im Liefergebiet bei n = 200** im Median 797 bis 1 107,5 (Plan: 966). Die Schranke ist dort im Median das 93,5- bis 129,8-Fache der Korrekturen.
- Sigmoid a = 0,5, c = −1: 9 von 30 Läufen konvergieren (Plan: 5 von 30); die Zeile hängt am Erzeuger.

## Modell

Punkte x_i ∈ ℝ², Etiketten y_i ∈ {−1, +1}. **Regel:** wenn y_i f(x_i) ≤ 0 (der Fall 0 zählt als Fehler), steigt der Zähler α_i um 1 (**Korrektur**); f(x) = Σ_j α_j y_j K̃(x_j, x) mit K̃ = K + 1 (der Bias als Zusatzkoordinate).
Start α = 0, deshalb hat eine Lernrate keine Wirkung auf die Entscheidungen. **Epoche** = ein Durchlauf in fester Reihenfolge (je Zufalls-Seed eine feste Zufallsfolge; Mischen in jeder Epoche ist ein Schalter, nicht gemessen);
**konvergiert** = eine Epoche ohne Korrektur; „Grenze erreicht“ heißt nicht „nicht trennbar“. **Der Kerntrick besteht darin, das Skalarprodukt überall durch K̃ zu ersetzen und sonst nichts zu ändern.**
Kerne: linear, Polynom (x·y + 1)^d mit d = 2, 3, Gauß exp(−‖x − y‖²/(2σ²)) mit der Bandbreite σ (γ = 1/(2σ²)), Sigmoid tanh(a x·y + c) nur als Falle.
**Marge heißt 2/‖w‖** (Breite zwischen den Klassen; viele Lehrtexte nennen 1/‖w‖ Marge, das ist hier der Abstand zur Ebene). Die **Schranke** k ≤ R̃²‖w*‖² (R̃² = max K̃(x, x), w* die größte Trennung in (φ, 1), also mit dem Bias als Koordinate)
ist eine Rechnung dieser Seite, keine Quellenaussage (die Beweisschritte stehen im 📐-Expander und sind in `tests/test_handbeispiele.py` nachgerechnet). „Marge in (φ, 1)“ ist nicht die Marge mit freiem Versatz b;
beide sind bei XOR gleich (Σαy = 0); beim Gauß-Kern liegen die gefundenen Margen-Verhältnisse beider Maße nah beieinander (Ringe, σ = 1: Median 0,167 gegen 0,165 und 0,194 gegen 0,192), beim Polynom 2 nicht (0,185 bis 0,191 gegen 0,254 bis 0,262). Experiment 3 zeigt das Maß mit freiem b. Die Formeln stehen im Expander „📐 Mathematische Formulierung“.

## Methodik

- **Duale Form** mit mitgeführtem Vektor s = K̃(α ∘ y): eine Prüfung kostet O(1), eine Korrektur O(n) (`ktp_perceptron.py`); eine schnelle und eine Punkt-für-Punkt-Fassung liefern dieselben Läufe (Test).
- **Größte Trennung** (`ktp_margin.py`): ein kleines duales QP mit Aktivmengenverfahren (nur NumPy) und Zertifikat (Dualitätslücke ≤ 10⁻⁶·max(1, ‖w*‖²), kleinste Randbedingung ≥ 1 − 10⁻⁴); ohne Zertifikat steht keine Zahl.
  Orakel: Aufzählung aller Stützmengen, `SVC` und nichtnegative kleinste Quadrate (nur in Tests).
- **Erzeuger** (`ktp_scenario.py`): XOR (vier Punkte), Ringe mit Lücke (Radius 1 und 2, Breite ±0,1; Lücke 0,8 im Radius), Liefergebiet (Kreis; Kunden gleichverteilt in [−1, 1]², Klasse +1 im Kreis mit halber Fläche), Monde (zwei Halbkreisbögen, beschränktes gleichverteiltes Rauschen).
  Zufallsquelle überall SplitMix64 (`ktp_rng.py`, Kopie der Vorlage mit Kopier-Test); kein `numpy.random`.
- **Eingefrorene Messreihen** (`ktp_sweep.json`, `tools/ktp_make_sweep.py`, `--check` rechnet neu und vergleicht): zwei Sätze Zufalls-Seeds, Zähleinheit Korrekturen, Epochen, Läufe.

## Befunde (gemessen, keine Behauptungen)

Zufallsergebnisse sind Bänder über die genannten Zufalls-Seeds; jede Zahl nennt ihre Zähleinheit. Wo zwei Zahlen stehen, sind es die Mediane der beiden Sätze. Grenze 3 000 Epochen, wenn nicht anders genannt.

| Frage | Befund | Test |
|---|---|---|
| Konvergiert das lineare Perceptron auf XOR? | Nein, exakt: Reihenfolge a, b, c, d gibt 4 Korrekturen in Epoche 1 und danach wieder w = 0 (Zyklus der Länge einer Epoche); in keiner der 24 Reihenfolgen konvergiert es | `test_handbeispiele.py`, `test_edge_cases.py` |
| Und mit Polynom 2? | 4 Korrekturen in jeder der 24 Reihenfolgen, α = (1, 1, 1, 1), f(x) = 8 x₁x₂, Marge 2√2 = 2,828, Schranke R̃²‖w*‖² = 5 (4,5 mit K statt K̃) | `test_handbeispiele.py` |
| Gauß-Kern auf XOR? | Marge 1 − exp(−2/σ²) = 0,999665 / 0,864665 / 0,393469 / 0,117503 für σ = 0,5 / 1 / 2 / 4; Korrekturen in der Reihenfolge a, b, c, d: 6 / 6 / 10 / 34, Schranke 8,01 / 10,70 / 51,67 / 579,42 | `test_oracle_kerntrick.py` |
| Rechnet der Kern wirklich im Merkmalsraum? | Duales Perceptron mit Polynomkern, primales im expliziten Merkmalsraum und `sklearn.linear_model.Perceptron`: gleiche Korrekturen, Epochen und Konvergenz in 200 von 200 Instanzen (4 Zeilen zu 50) aus der Messreihe `ktp_sweep.json` (Zufalls-Seeds 1 bis 50 je Zeile; die Suite prüft nur je 10 Zufalls-Seeds in 4 Zeilen mit kleineren Instanzen); w relativ höchstens 2,7·10⁻¹⁴ (dual gegen primal); w gegen `sklearn` bitgleich (`numpy.array_equal`, Messung dieser README mit Python 3.14 unter Windows, Linux-Lauf steht aus; die Tests prüfen relativ 10⁻⁹) | `test_claims.py`, `test_oracle_kerntrick.py` |
| Aufwand? | p = 10, Polynom 3: 286 Merkmale gegen 12 Multiplikationen je Skalarprodukt, nur bei der Auswertung an einem neuen Punkt mit fertigem φ(x); die duale Form ist bis 23 verschiedene korrigierte Punkte billiger (24 · 12 = 288 > 286), mit Bau von φ(x) (Annahme: mindestens eine Multiplikation je Merkmal) bis 47 | `test_kernels.py` |
| Ringe, linear? | 0 von 60 Läufen konvergieren; Korrekturen nach 3 000 Epochen im Median 150 644 bis 150 714,5 | `test_claims.py` |
| Ringe, Kern? | Gauß σ = 0,25 / 0,5 / 1: 60 von 60 mit 16 / 16 / 15 Korrekturen im Median (alle 60 Läufe zwischen 12 und 22); Polynom 2: 14,5 bis 15; Polynom 3: 31 | `test_claims.py` |
| Liefergebiet ohne Lücke? | Polynom 2, n = 100: 60 von 60 mit 193 bis 315 Korrekturen im Median, bei n = 200: 40 von 40 mit 797 bis 1 107,5; die Schranke ist bei n = 200 im Median das 93,5- bis 129,8-Fache der Korrekturen und wird in 40 von 40 Läufen eingehalten | `test_claims.py` |
| Monde (Grenze 2 000 Epochen)? | Linear und Polynom 2: je Rauschstufe (0 und 0,1) 0 von 40 Läufen (20 je Satz, zwei Sätze); Polynom 3: 40 von 40 je Rauschstufe (20 je Satz) mit 36,5 bis 44; Gauß σ = 0,25 und 0,5: 8 bis 10; σ = 1: 37 bis 49 Korrekturen im Median | `test_claims.py` |
| Bandbreite (Ringe, Gauß, 20 000 Epochen)? | U-förmig: 71 bis 72 Korrekturen bei σ = 0,02; 15 bis 16 zwischen σ = 0,25 und 1; 977,5 bis 1 248 bei σ = 8 (Tabelle unten); in 540 von 540 Läufen dieselbe Meldung „konvergiert“ | `test_claims.py` |
| Wird die Schranke eingehalten? | In 540 von 540 zertifizierten Läufen (Ringe, Gauß, 9 Bandbreiten) und 40 von 40 (Liefergebiet n = 200, Polynom 2): keine Überschreitung; die Mediane der Schranke in den Ringe-Zeilen liegen je nach σ zwischen 33,5 und 14 980 | `test_claims.py` |
| Welche Trennung findet das Perceptron? | Irgendeine, die erste: im Median 0,167 und 0,194 der größten Marge (Ringe, Gauß σ = 1), 0,057 (σ = 0,5), Polynom 2: 0,185 bis 0,191 mit freiem b gegen 0,254 bis 0,262 mit dem Bias als Koordinate; Liefergebiet (n = 100, Polynom 2): 0,169 bis 0,171 | `test_claims.py` |
| Hängt die gefundene Trennung an der Reihenfolge? | Der kleinste Kosinus der Normalen zweier Reihenfolgen derselben Instanz liegt im Median bei 0,819 bis 0,822 (Gauß σ = 1), 0,676 bis 0,693 (σ = 0,5), 0,914 bis 0,915 (Polynom 2), 0,994 bis 0,995 (Liefergebiet) | `test_claims.py` |
| Zufallsetiketten, Gauß-Kern? | Alle 60 Instanzen sind trennbar (verschiedene Punkte, Gram-Matrix positiv definit, K c = y mit Residuum unter 10⁻³ gelöst); „konvergiert“ bei 50 000 Epochen: 60 / 60 / 58 / 12 von 60 für σ = 0,05 / 0,1 / 0,25 / 0,5, **bei 3 000 Epochen aber nur 59 / 58 / 39 / 0 von 60**; Korrekturen im Median 118 bis 121 / 359 bis 379 / 6 624 bis 8 505 gegen 36 bis 38 / 20 / 16 mit den wahren Etiketten | `test_claims.py` |
| Hilft „konvergiert“ bei 10 % vertauschten Etiketten? | Nein: Testfehler gegen die wahre Klasse in den Gauß-Zeilen 0,158 bis 0,172, ob alle 40, 20 von 40 oder keiner der 40 Läufe konvergieren (Tabelle unten) | `test_claims.py` |
| Sigmoid, kein Skalarprodukt (Grenze 1 000 Epochen)? | Ringe, n = 100 > d = 2: Gram-Matrix in 30 von 30 Zufalls-Seeds indefinit in allen 7 Parametersätzen; der Lauf meldet nichts; 0, 9 und 30 von 30 konvergieren bei a = 1, c = −1 / a = 0,5, c = −1 / a = 0,1, c = −1 (dort 41 Korrekturen) | `test_claims.py` |

**Ringe, Gauß-Kern, nach Bandbreite** (je 30 Zufalls-Seeds in zwei Sätzen, Grenze 20 000 Epochen; alle 540 Läufe konvergieren): Median der beiden Sätze, Bereich über alle 60 Läufe, Epochen im Median, Schranke R̃²‖w*‖² im Median

| σ | Korrekturen (Median) | Bereich | Epochen (Median) | Schranke (Median) |
|---|---|---|---|---|
| 0,02 | 71 bis 72 | 60 bis 80 | 4 bis 5 | 190,8 bis 191,5 |
| 0,05 | 36 bis 38 | 30 bis 46 | 3 | 160,8 bis 161,3 |
| 0,1 | 20 | 18 bis 24 | 2 | 113,0 bis 113,6 |
| 0,25 | 16 | 12 bis 18 | 2 | 53,6 bis 53,7 |
| 0,5 | 16 | 12 bis 22 | 2 | 33,5 bis 33,6 |
| 1 | 15 | 13 bis 17 | 2 | 37,1 bis 37,2 |
| 2 | 24 | 17 bis 40 | 2,5 bis 3 | 110,2 bis 110,3 |
| 4 | 100 bis 142 | 49 bis 206 | 4 bis 5 | 1 050,3 bis 1 055,3 |
| 8 | 977,5 bis 1 248 | 179 bis 2 281 | 23 bis 29 | 14 968,0 bis 14 979,8 |

Die Aussage „kleine Bandbreite verlängert das Perceptron nicht“ gilt nur bis σ = 0,25 (der Bereich der ersten Vormessung): darunter wächst die Zahl der Korrekturen (σ = 0,1: 20, σ = 0,05: 36 bis 38, σ = 0,02: 71 bis 72).

**Zufallsetiketten auf den Ringen (n = 100)**, 60 Zufalls-Seeds in zwei Sätzen zu je 30, Messgrenze 50 000 Epochen und 300 000 Korrekturen:

| σ | konvergiert | innerhalb 3 000 Epochen | Korrekturen (Median) | Bereich | wahre Etiketten (Median) | Korrekturgrenze erreicht | Epochengrenze erreicht | Verhältnis der Mediane |
|---|---|---|---|---|---|---|---|---|
| 0,05 | 60 von 60 | 59 von 60 | 118 bis 121 | 80 bis 11 630 | 36 bis 38 | 0 | 0 | 3,2 bis 3,3 |
| 0,1 | 60 von 60 | 58 von 60 | 359 bis 379 | 102 bis 22 135 | 20 | 0 | 0 | 17,9 bis 18,9 |
| 0,25 | 58 von 60 | 39 von 60 | 6 624 bis 8 505 | 952 bis 166 997 | 16 | 0 | 2 | 414 bis 532 |
| 0,5 | 12 von 60 | 0 von 60 | 300 000 (untere Schranke) | 22 674 bis 300 000 | 16 | 41 | 7 | über 18 000 (untere Schranke) |

„Grenze erreicht“ ist nicht „nicht trennbar“: bei σ = 0,25 meldet die Seite mit 3 000 Epochen in 21 von 60 Zufalls-Seeds „Grenze erreicht“, obwohl alle 60 trennbar sind (bei 50 000 Epochen: 2 von 60). Bei σ = 0,5 ist der Median
eine untere Schranke, weil die Korrekturgrenze in 41 von 60 Läufen greift.

**Liefergebiet n = 200 mit 10 % vertauschten Etiketten** (40 Zufalls-Seeds in zwei Sätzen zu je 20, Grenze 3 000 Epochen, 4 000 frische Testpunkte; Fehler gegen die **wahre** Klasse):

| Kern | konvergiert | Korrekturen (Median) | Trainingsfehler (Mittel) | Testfehler (Mittel, Standardfehler) |
|---|---|---|---|---|
| Gauß σ = 0,05 | 40 von 40 | 163 bis 165 | 0,000 | 0,158 (0,003) |
| Gauß σ = 0,1 | 40 von 40 | 693 bis 923,5 | 0,000 | 0,163 (0,003) |
| Gauß σ = 0,2 | 20 von 40 | 16 192 bis 17 990,5 | 0,022 | 0,172 (0,004) |
| Gauß σ = 0,4 | 0 von 40 | 81 017,5 bis 82 890 | 0,140 | 0,170 (0,007) |
| Gauß σ = 0,8 | 0 von 40 | 118 021,5 bis 118 900,5 | 0,194 | 0,165 (0,011) |
| Polynom 2 | 0 von 40 | 147 428,5 bis 148 895,5 | 0,248 | 0,187 (0,014) |

**Ehrlicher Negativbefund:** die Zeilen ohne Konvergenz (0,170; 0,165) sind nicht besser als die mit Konvergenz (0,158; 0,163) und in diesen Zahlen auch nicht erkennbar schlechter; die Differenz benachbarter Gauß-Zeilen ist höchstens 0,009
(1,9 Standardfehler der Differenz, innerhalb der Streuung). Der Zeilenvergleich vermengt Konvergenz mit σ; innerhalb der Zeile σ = 0,2 ist der Testfehler konvergierter und nicht konvergierter Läufe gleich (0,171 gegen 0,173, 20 Läufe je Gruppe; Kontrollmessung, nicht in `ktp_sweep.json`, nicht im Test). Was unterscheidet, ist die Zahl der Korrekturen derselben Punkte mit wahren und mit zufälligen Etiketten (Tabelle oben), nicht die Meldung. Die Aussage „Auswendiglernen schadet“ ist hier
**nicht** belegt; ohne ein Vergleichsverfahren mit abgestimmtem C darf sie nicht stehen.

## Zahl → Test

Jede Zahl der Befunde-Tabelle steht in `ktp_sweep.json` (eingefroren) und wird in `tests/test_claims.py` auf zwei Weisen geprüft: der Text dieser README enthält die Zahl aus der Datei, und eine Neurechnung mit wenigen Zufalls-Seeds
liegt im Band der Datei (Anteile mit der Streuung, Ganzzahlen und Brüche exakt). Die vollständige Neurechnung beider Sätze leistet `python tools/ktp_make_sweep.py --check` (Messung dieser README, nicht Teil der Testsuite).
Nicht im Test: die Verhältnisse in den Spalten „Verhältnis der Mediane“ (aus den Medianen der Tabelle gerechnet), die Mittelwerte der Trainingsfehler, die Kosinus-Zahlen (Messung dieser README, nicht im Test).

## Ehrliche Grenzen

- Nur zweidimensionale, ausgedachte Beispiele. Größere n, höhere Dimension, Bandbreite im Liefergebiet und in den Monden sind nicht gemessen; die Bandbreitenkurve gilt für die Ringe.
- Kein Regler C, λ oder ν und kein Vergleich mit einem abgestimmten Verfahren: deshalb keine Aussage „Auswendiglernen schadet“. Die Zahl „Perceptron gegen größte Marge“ eines anderen Erzeugers (linear, zwei Gebiete mit Puffer) führt die Nachfolgerin der Reihe;
  die Ringe mit Lücke hier haben ihre eigene Zeile, und beide Zahlen dürfen nicht als Widerspruch gelesen werden.
- Gilt nur für die gemessene Regel: feste Reihenfolge je Zufalls-Seed, Bias als K + 1, Epochengrenze wie genannt (Messgrenze 50 000 gegen den Regler 3 000), Zähleinheiten Korrekturen, Zufalls-Seeds und Multiplikationen. Mischen je Epoche ist als Schalter angeboten, nicht gemessen.
- Das Live-Modell ist nicht die Studie: die Seite rechnet eine Instanz, die eingefrorenen Tabellen fassen 20 bis 60 Zufalls-Seeds zusammen. Ein einzelner Zufalls-Seed ist kein Befund.
- Dass die Lücke zwischen den Ringen der Hebel für die kleine Korrekturzahl ist (gegenüber dem Liefergebiet ohne Lücke), ist eine Deutung: Ringe und Liefergebiet unterscheiden sich auch in der Form der Klassen.
- Der Gewinn des Kerntricks ist Rechenaufwand bei großem D und der Zugang zu unendlichdimensionalen Merkmalsräumen; in 2D (p = 2) ist der Kern ab 3 verschiedenen korrigierten Punkten teurer (Polynom 3 und Polynom 2); bei Polynom 2 und 2 Punkten ist er gleich teuer, also nicht mehr billiger (Experiment 2). Die Aufwandszahlen betreffen nur die Auswertung, nicht das Training.
- Die Zeilen zu Plattformen: die Zahlen wurden mit Python 3.14 unter Windows gemessen; ein Linux-Lauf steht aus. Entscheidungen bei Werten nahe 0 (y f ≤ 0 bei f = 0) können bei Gleitkomma-Gram-Matrizen kippen, die Tests prüfen Bänder.
- Ein Nutzen von Kernmethoden in Logistik, Lager oder Fertigung ist nicht belegt.

## Verifikation

- **Kern:** das Perceptron gegen Handrechnung (zwei Punkte, XOR), schnelle gegen Punkt-für-Punkt-Fassung, Eigenschaften (Spiegeln der Etiketten, Umnummerieren, Drehen, Skalieren der Gram-Matrix, Lernrate), Randfälle (`tests/test_perceptron.py`, `tests/test_edge_cases.py`).
- **Orakel** (`tests/test_oracle_kerntrick.py`): dual gegen primal im Tensorraum gegen `sklearn.linear_model.Perceptron`; Kern gegen Merkmalsraum; größte Trennung gegen Aufzählung aller Stützmengen, `SVC` und nichtnegative kleinste Quadrate; Gauß-Marge auf XOR gegen die geschlossene Form;
  kleinster Eigenwert gegen `numpy.linalg.eigvalsh`; Literaturwert Cortes/Vapnik 1995 (Polynomkern, Grad-2-Merkmal).
- **Handbeispiele** (`tests/test_handbeispiele.py`, `tests/handbeispiel_exakt.py`): 304 Prüfungen in exakten Brüchen, dann die App gegen dieselben Zahlen.
- **Zufall:** Kopier-Test von `ktp_rng.py`, eingefrorene Werte je Erzeuger, Eigenschaften der Erzeuger (`tests/test_rng_copy.py`, `tests/test_scenario.py`).
- **Verdikte** als reine Funktionen, jede Zahl im Text an handgesetzten Werten (`tests/test_verdicts.py`); je Schnellstart-Szenario die erwartete Verdikt-Zeile (`tests/test_app.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Enden), Footer-Test, Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Szenario hervorgehoben, jede Reglerwahl hebt die Markierung auf), Laufzeit-Importe, `modul_frisch`.
- Die Testzahl: 227 Tests (`pytest --collect-only -q tests`); `tests/test_claims.py` prüft, dass die Zahl hier mit der Suite übereinstimmt.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Grenzen, Formeln |
| `ktp_rng.py` | SplitMix64 (Kopie der Vorlage, Kopier-Test) |
| `ktp_kernels.py` | Kerne, Gram-Matrizen, Merkmalsraum des Polynomkerns, Aufwandszahlen |
| `ktp_perceptron.py` | Perceptron dual (Kerntrick) und primal, Protokoll |
| `ktp_margin.py` | größte Trennung (duales QP mit Zertifikat), Schranke, Marge der gefundenen Trennung |
| `ktp_scenario.py` | Erzeuger: XOR, Ringe, Liefergebiet, Monde, Etikettenoperationen, Reihenfolgen |
| `ktp_evaluation.py` | Analysen je Experiment, Verdikte, Tabellentexte, Formeln |
| `ktp_presets.py` | Permalink, Schnellstart-Szenarien (aktives hervorgehoben), Zufalls-Seed-Button |
| `ktp_constants.py` | Regler-Grenzen, Kernlisten, Szenarien |
| `ktp_layout.py` | Fester Kopf (Stylesheet) |
| `ktp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ktp_sweep.json` | eingefrorene Messreihen (Satz A und B) |
| `modul_frisch.py` | lädt geänderte Hilfsmodule nach einem Update neu |
| `tools/ktp_make_sweep.py` | rechnet die Messreihen neu (`--check` vergleicht) |
| `requirements.txt` | Laufzeit: streamlit, numpy, plotly (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scikit-learn und scipy (nur Fremdreferenz in Tests und im Werkzeug) |
| `tests/` | Perceptron, Kerne, Erzeuger, Zufallskopie, Orakel, Handbeispiele, Verdikte, Claims, Randfälle, Szenarien, Layout, AppTest, Footer, Laufzeit-Importe, `modul_frisch` |

## Bewusst nicht umgesetzt

Nicht angeboten und nicht gemessen; nirgends ein Regler und kein Versprechen: C, λ und ν; ein Vergleich mit abgestimmtem C; Bandbreite im Liefergebiet und in den Monden, größere n und höhere Dimension; Mischen je Epoche als Messreihe;
die Prüfung und Reparatur unzulässiger Kerne; die Näherung der Gram-Matrix; TabPFN (nur Quelle); Gastbeiträge (Kernel-PCA, Kernel-k-Means, MMD, HSIC, Graphkerne); ein PDF-Export; ein Nutzen im Betrieb.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html).
