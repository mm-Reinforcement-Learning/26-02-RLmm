# Die Markov-Eigenschaft

## Definition

Die **Markov-Eigenschaft** (auch *Gedächtnislosigkeit* genannt) besagt, dass der zukünftige Zustand eines Systems **ausschließlich vom aktuellen Zustand** abhängt – und **nicht** von der Geschichte, wie man dorthin gelangt ist.

Formal:

$$P(X_{t+1} = s \mid X_t = x_t, X_{t-1} = x_{t-1}, \ldots, X_0 = x_0) = P(X_{t+1} = s \mid X_t = x_t)$$

> „Die Zukunft ist bedingt unabhängig von der Vergangenheit, gegeben die Gegenwart."

---

## Anschauliches Beispiel: Wetter

Angenommen, das Wetter kann drei Zustände annehmen:

| Zustand | Symbol |
|---------|--------|
| Sonnig  | ☀️ S   |
| Bewölkt | ☁️ B   |
| Regnerisch | 🌧️ R |

### Übergangsmatrix

Die Wahrscheinlichkeit, von einem Zustand in den nächsten zu wechseln:

|         | → S   | → B   | → R   |
|---------|-------|-------|-------|
| **S** ☀️ | 0.7   | 0.2   | 0.1   |
| **B** ☁️ | 0.3   | 0.4   | 0.3   |
| **R** 🌧️ | 0.2   | 0.3   | 0.5   |

### Markov-Eigenschaft angewendet

Angenommen, die Wetterhistorie der letzten Tage war:

```
Montag: ☀️  →  Dienstag: ☁️  →  Mittwoch: 🌧️
```

**Frage:** Wie wahrscheinlich ist es, dass es am **Donnerstag** regnet?

**Mit Markov-Eigenschaft:**

Wir schauen **nur** auf den aktuellen Zustand (Mittwoch = 🌧️):

$$P(\text{Donnerstag} = R \mid \text{Mittwoch} = R) = 0.5$$

Die Vergangenheit (Montag = ☀️, Dienstag = ☁️) wird **vollständig ignoriert**. Es spielt keine Rolle, wie wir zum Mittwoch-Regen gekommen sind.

---

## Gegenbeispiel: Verletzung der Markov-Eigenschaft

Stell dir vor, du modellierst die **Körpertemperatur** eines Patienten.

Wenn ein Patient seit 3 Tagen Fieber hat, ist die Wahrscheinlichkeit, dass er morgen noch Fieber hat, **höher** als wenn er erst seit gestern Fieber hat. Das System hat also ein *Gedächtnis* – die Markov-Eigenschaft ist **verletzt**.

$$P(X_{t+1} \mid X_t) \neq P(X_{t+1} \mid X_t, X_{t-1}, X_{t-2})$$

In diesem Fall wäre ein **Markov-Prozess höherer Ordnung** oder ein anderes Modell nötig.

---

## Markov-Kette: Schritt-für-Schritt

### Szenario: Aktienkurs vereinfacht

Zustände: `Steigt` (↑), `Fällt` (↓)

Übergangswahrscheinlichkeiten:

```
P(↑ → ↑) = 0.6    P(↑ → ↓) = 0.4
P(↓ → ↑) = 0.3    P(↓ → ↓) = 0.7
```

**Startbedingung:** Heute steigt der Kurs (↑)

**Berechnung für übermorgen (2 Schritte):**

```
Schritt 1 – morgen:
  P(morgen = ↑) = 0.6
  P(morgen = ↓) = 0.4

Schritt 2 – übermorgen:
  P(übermorgen = ↑) = P(morgen=↑)·P(↑→↑) + P(morgen=↓)·P(↓→↑)
                    = 0.6 · 0.6 + 0.4 · 0.3
                    = 0.36 + 0.12 = 0.48

  P(übermorgen = ↓) = 0.6 · 0.4 + 0.4 · 0.7
                    = 0.24 + 0.28 = 0.52
```

> Nur der aktuelle Zustand geht in die Rechnung ein – kein Blick weiter zurück.

---

## Stationäre Verteilung

Auf lange Sicht konvergiert eine ergodische Markov-Kette gegen eine **stationäre Verteilung** π, bei der gilt:

$$\pi = \pi \cdot P$$

Für das Aktienbeispiel:

$$\pi_\uparrow = 0.3 / (0.3 + 0.4) \approx 0.43 \qquad \pi_\downarrow \approx 0.57$$

Das bedeutet: Unabhängig vom Startzustand wird der Kurs auf lange Sicht ~43 % der Zeit steigen und ~57 % der Zeit fallen.

---

## Zusammenfassung

| Eigenschaft | Bedeutung |
|-------------|-----------|
| **Gedächtnislosigkeit** | Nur der aktuelle Zustand zählt |
| **Übergangsmatrix** | Beschreibt alle Wahrscheinlichkeiten |
| **Stationäre Verteilung** | Langfristiges Gleichgewicht |
| **Ordnung** | Markov-1: 1 Zustand; Markov-n: n Zustände zurück |

### Typische Anwendungen

- Sprachmodelle (n-Gramm-Modelle)
- Reinforcement Learning (MDP – Markov Decision Process)
- Warteschlangentheorie
- Bioinformatik (Hidden Markov Models)
- Neuromorphes Computing (Zustandsübergänge in SNNs)
