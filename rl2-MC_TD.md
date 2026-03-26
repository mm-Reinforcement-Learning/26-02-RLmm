# Reinforcement Learning: Monte-Carlo & Temporal-Difference Learning

**Technischer Trainer | 14 Unterrichtsstunden à 45 Min.**

---

## 📚 Kursübersicht & Zeitplan

| Einheit | Thema | Stunden |
|---|---|---|
| 1–2 | Grundlagen & Wiederholung (MDP, Bellman) | 2 |
| 3–4 | Monte-Carlo Methoden | 2 |
| 5–6 | Temporal-Difference Learning (TD(0)) | 2 |
| 7–8 | On-Policy: SARSA | 2 |
| 9–10 | On-Policy: Expected SARSA | 2 |
| 11–12 | Off-Policy: Q-Learning | 2 |
| 13–14 | Vergleich, Praxis & Abschluss | 2 |

---

## 🧱 Einheit 1–2: Grundlagen & Wiederholung

### Intuitive Erklärung

Bevor wir MC und TD verstehen, müssen wir das **Grundproblem** kennen:  
Ein Agent lebt in einer Welt, macht Aktionen, bekommt Belohnungen – und will lernen, **wie er sich verhalten soll**, um langfristig möglichst viel Belohnung zu sammeln.

**Die drei zentralen Fragen:**
1. *Wie gut ist es, in Zustand S zu sein?* → **Value Function V(s)**
2. *Wie gut ist es, in Zustand S die Aktion A zu wählen?* → **Action-Value Function Q(s,a)**
3. *Welche Aktion soll ich wählen?* → **Policy π**

**Bellman-Gleichung (intuitiv):**  
> „Der Wert eines Zustands = Sofortige Belohnung + Wert des nächsten Zustands (abgezinst)"

```
V(s) = R(s) + γ · V(s')
```

### Visualisierungsidee 🎨

```
┌─────────────────────────────────────────────────┐
│  Schachbrett / Gridworld                        │
│                                                 │
│  [ ][ ][ ][🏆]                                  │
│  [ ][🧱][ ][ ]                                  │
│  [ ][ ][ ][ ]                                   │
│  [🤖][ ][ ][💀]                                │
│                                                 │
│  Agent (🤖) → Ziel (🏆), vermeide Tod (💀)     │
└─────────────────────────────────────────────────┘
```

Diese **4×4 Gridworld** wird der **rote Faden für den ganzen Kurs**.

### Beispiel

```python
# Einfache Gridworld-Umgebung
states = [(i, j) for i in range(4) for j in range(4)]
actions = ['up', 'down', 'left', 'right']
gamma = 0.9  # Discount-Faktor

# Belohnungen
R = {(0,3): +10,   # Ziel
     (3,3): -10}   # Tod
# alle anderen: 0
```

### Häufige Missverständnisse ⚠️

| Missverständnis | Richtigstellung |
|---|---|
| „Belohnung = Wert" | Nein! Belohnung ist **sofortig**, Wert ist **langfristig** |
| „γ=1 ist am besten" | Kann zu Instabilität führen, kein Anreiz für schnelle Lösung |
| „Policy = deterministisch" | Kann stochastisch sein: π(a\|s) ∈ [0,1] |

### 🧩 Mini-Quiz

> **Frage 1:** Warum brauchen wir einen Discount-Faktor γ < 1?  
> a) Damit der Agent schneller lernt  
> b) Damit zukünftige Belohnungen weniger zählen als sofortige  
> c) Damit die Policy deterministisch wird  
> **→ Antwort: b)**

> **Frage 2:** V(s) = 5, γ = 0.9, V(s') = 8. Wie lautet R(s)?  
> → R(s) = V(s) - γ·V(s') = 5 - 0.9×8 = **-2.2**

---

## 🎲 Einheit 3–4: Monte-Carlo Methoden

### Intuitive Erklärung

**Kernidee:** Lernen durch **vollständige Erfahrung**.  
Wie ein Schüler, der erst **nach dem Test** erfährt, was richtig war – er schaut auf sein komplettes Ergebnis.

> 🎯 „Spiele das Spiel bis zum Ende, dann schau was insgesamt rausgekommen ist."

**Monte-Carlo Prinzip:**
1. Führe eine **komplette Episode** durch (S₀,A₀,R₁,S₁,...,Sₜ)
2. Berechne den **tatsächlichen Return** rückwärts
3. Update den Schätzwert V(s) in Richtung des tatsächlichen Returns

**Return berechnen:**
```
Gₜ = Rₜ₊₁ + γ·Rₜ₊₂ + γ²·Rₜ₊₃ + ... + γᵀ⁻ᵗ·Rₜ
```

**Update-Regel:**
```
V(s) ← V(s) + α · [G - V(s)]
         ↑          ↑
    Lernrate    "Überraschung"
```

### Visualisierungsidee 🎨

```
Episode 1:  🤖→→→↑→🏆
            Belohnungen: 0, 0, 0, 0, +10
            G (rückwärts berechnet):

Zeitschritt:  t=0   t=1   t=2   t=3   t=4
Reward:        0     0     0     0    +10
Return G:     7.29  8.1   9.0  10.0  10.0
              ←─────────────────────────(γ=0.9)

NACH der Episode: Update aller besuchten Zustände!
```

### First-Visit vs. Every-Visit MC

```
Episode: S₁ → S₂ → S₁ → S₃ → Ende
                    ↑
           S₁ wird zweimal besucht!

First-Visit MC:   Nur 1. Besuch von S₁ zählt
Every-Visit MC:   Beide Besuche von S₁ zählen
```

### Beispiel – Monte-Carlo in Python

```python
import numpy as np
from collections import defaultdict

def monte_carlo_prediction(env, policy, episodes=1000, gamma=0.9, alpha=0.1):
    V = defaultdict(float)  # Value Function

    for ep in range(episodes):
        # 1. Episode generieren
        episode = []
        state = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # 2. Return rückwärts berechnen (First-Visit MC)
        G = 0
        visited_states = set()

        for state, action, reward in reversed(episode):
            G = reward + gamma * G  # Return akkumulieren

            if state not in visited_states:  # First-Visit
                visited_states.add(state)
                # 3. Value Function updaten
                V[state] += alpha * (G - V[state])

    return V
```

### MC für Control (Q-Werte)

```python
def monte_carlo_control(env, episodes=5000, gamma=0.9, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))

    def epsilon_greedy(state):
        if np.random.random() < epsilon:
            return env.random_action()          # Exploration
        return max(Q[state], key=Q[state].get)  # Exploitation

    for ep in range(episodes):
        episode = []
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                Q[state][action] += 0.1 * (G - Q[state][action])

    return Q
```

### Vor- und Nachteile MC

| ✅ Vorteile | ❌ Nachteile |
|---|---|
| Kein Bias – lernt aus echten Returns | Braucht **komplette Episoden** |
| Einfach zu verstehen | **Hohe Varianz** |
| Kein Modell der Umgebung nötig | Kein Online-Learning möglich |
| Unabhängig von Initialwerten | Nur für episodische Tasks |

### Häufige Missverständnisse ⚠️

| Missverständnis | Richtigstellung |
|---|---|
| „MC ist langsam, also schlecht" | Für viele echte Probleme (z.B. Brettspiele) sehr effektiv |
| „First-Visit ist besser als Every-Visit" | Beide konvergieren – Kontext entscheidet |
| „MC lernt aus jedem Schritt" | **Nein!** Nur nach kompletter Episode |

### 🧩 Mini-Quiz

> **Frage 3:** Warum hat MC eine hohe Varianz?  
> **→** Weil der Return G von **vielen zufälligen Schritten** abhängt – kleine Zufälle summieren sich auf.

> **Frage 4:** Kann man MC für ein Schachspiel einsetzen?  
> **→ Ja!** Schach hat klare Endpunkte (Episoden). MC ist hier sogar klassisch (z.B. Monte-Carlo Tree Search).

> **Frage 5:** Was ist α (alpha) in der Update-Regel?  
> a) Der Discount-Faktor  
> b) Die Lernrate (Schrittgröße)  
> c) Die Explorations-Rate  
> **→ Antwort: b)**

---

## ⚡ Einheit 5–6: Temporal-Difference Learning (TD(0))

### Intuitive Erklärung

**Kernidee:** Lernen **während** der Episode, nach **jedem Schritt**.

> Vergleich MC vs. TD:  
> - **MC:** Schüler lernt erst nach dem **gesamten Schuljahr** (Jahreszeugnis)  
> - **TD:** Schüler bekommt nach **jeder Hausaufgabe** Feedback

**TD(0) Update:**
```
V(s) ← V(s) + α · [R + γ·V(s') - V(s)]
                    └──────────────────┘
                       "TD-Error" (δ)
```

**TD-Error δ:**  
Das ist die **Überraschung** – wie viel besser/schlechter war die Realität vs. unsere Schätzung?

```
δ = R + γ·V(s') - V(s)
    └──────────┘   └──┘
    Was ich bekam  Was ich erwartet hatte
```

### Visualisierungsidee 🎨

```
MC:
t=0  t=1  t=2  t=3  t=4(Ende)
 S₀ → S₁ → S₂ → S₃ → 🏆
 ←─────────────────────────── Update NACH Episode
TD:
t=0  t=1  t=2  t=3  t=4(Ende)
 S₀ → S₁ → S₂ → S₃ → 🏆
 ↑    ↑    ↑    ↑    ↑
Update nach JEDEM Schritt!
```

**Bootstrapping:**
```
MC:  V(s) → G              (echter Return, keine Schätzung)
TD:  V(s) → R + γ·V(s')   (Schätzung nutzt Schätzung = Bootstrapping)
```

### Beispiel – TD(0)

```python
def td_zero(env, policy, episodes=1000, gamma=0.9, alpha=0.1):
    V = defaultdict(float)

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)

            # TD-Update nach JEDEM Schritt!
            td_error = reward + gamma * V[next_state] - V[state]
            V[state] += alpha * td_error

            state = next_state

    return V
```

### MC vs. TD Vergleich

```
Beispiel: Agent geht 5 Schritte, dann Belohnung +10, gamma=0.9, alpha=0.1

MC (nach Episode):
V(S₀) ← V(S₀) + α[7.29 - V(S₀)]  # echter Return
V(S₁) ← V(S₁) + α[8.1  - V(S₁)]
V(S₂) ← V(S₂) + α[9.0  - V(S₂)]

TD (nach jedem Schritt):
Schritt 1: V(S₀) ← V(S₀) + α[0 + γ·V(S₁) - V(S₀)]
Schritt 2: V(S₁) ← V(S₁) + α[0 + γ·V(S₂) - V(S₁)]
...
Schritt 5: V(S₄) ← V(S₄) + α[10 + γ·0    - V(S₄)]
           ↑ Erst hier "weiß" TD vom Reward!
```

### Häufige Missverständnisse ⚠️

| Missverständnis | Richtigstellung |
|---|---|
| „TD ist immer besser als MC" | TD hat mehr **Bias** durch Bootstrapping, MC mehr **Varianz** |
| „TD-Error ist ein Fehler" | Eher ein **Lern-Signal** – auch positiv möglich |
| „TD braucht keine Episodes" | TD kann mit kontinuierlichen Tasks umgehen, muss aber nicht |

### 🧩 Mini-Quiz

> **Frage 6:** Was ist Bootstrapping?  
> **→** Eine Schätzung mit einer **anderen Schätzung** verbessern (V(s) nutzt V(s'))

> **Frage 7:** Ein Agent in S₀, macht Aktion, landet in S₁, bekommt R=2.  
> V(S₀)=3, V(S₁)=5, γ=0.9, α=0.1. Wie lautet der neue Wert V(S₀)?  
> → δ = 2 + 0.9×5 - 3 = **3.5**  
> → V(S₀) = 3 + 0.1×3.5 = **3.35**

---

## 🔄 Einheit 7–8: On-Policy – SARSA

### Intuitive Erklärung

**SARSA = State-Action-Reward-State-Action**  
Jetzt lernen wir nicht nur V(s), sondern **Q(s,a)** – den Wert einer Aktion in einem Zustand.

> 🔑 **On-Policy:** Das Tupel, aus dem wir lernen, verwendet **dieselbe Policy**, die auch handelt.

**SARSA Name erklärt:**
```
(Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁, Aₜ₊₁)
  S    A    R      S'     A'
```
→ Wir brauchen **5 Elemente** für ein Update!

**Update-Regel:**
```
Q(s,a) ← Q(s,a) + α · [R + γ·Q(s',a') - Q(s,a)]
                               ↑
                    Nächste Aktion NACH aktueller Policy!
```

### Visualisierungsidee 🎨

```
SARSA - Lernschleife:

  ┌─────────────────────────────────────────┐
  │                                         │
  ▼                                         │
Zustand S ──π(s)──► Aktion A               │
    │                   │                  │
    │              Umgebung                │
    │                   │                  │
    ▼                   ▼                  │
Zustand S' ◄──── Reward R                 │
    │                                      │
    └──π(s')──► Aktion A' ────────────────►┘
                    │
                    └─► UPDATE Q(S,A) mit (S,A,R,S',A')
```

### Beispiel – SARSA

```python
def sarsa(env, episodes=1000, gamma=0.9, alpha=0.1, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))

    def epsilon_greedy(state):
        if np.random.random() < epsilon:
            return env.random_action()
        actions = env.available_actions(state)
        return max(actions, key=lambda a: Q[state][a])

    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy(state)  # A aus S mit Policy
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(next_state)  # A' aus S' mit Policy

            # SARSA Update: benutzt tatsächliche nächste Aktion
            td_error = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            action = next_action  # A' wird neues A!

    return Q
```

### Wichtiges Verhalten: SARSA ist vorsichtig

```
Klippen-Problem (Cliff Walking):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  │  │  ← sicherer Weg
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│🤖│💀│💀│💀│💀│💀│💀│💀│💀│🏆│  ← Klippe (R=-100)
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

SARSA lernt:  OBEN LANG gehen (sicherer Weg)
              → Weil ε-greedy manchmal in die Klippe fällt
              → SARSA "weiß" das und ist vorsichtig!
```

### Häufige Missverständnisse ⚠️

| Missverständnis | Richtigstellung |
|---|---|
| „On-Policy = besser" | On/Off-Policy sind **verschiedene Werkzeuge**, nicht besser/schlechter |
| „SARSA lernt die optimale Policy" | SARSA lernt die **optimale Policy bzgl. ε-greedy** (nicht die greedy-optimale) |
| „A' muss die beste Aktion sein" | Nein! A' ist die Aktion **wie sie die Policy wählen würde** |

### 🧩 Mini-Quiz

> **Frage 8:** Warum heißt SARSA "on-policy"?  
> **→** Weil der Update `Q(s',a')` die **gleiche ε-greedy Policy** verwendet, die den Agenten steuert.

> **Frage 9:** Was passiert mit SARSA wenn ε→0?  
> **→** SARSA nähert sich Q-Learning an (fast greedy). Aber Exploration geht verloren!

---

## 🎯 Einheit 9–10: On-Policy – Expected SARSA

### Intuitive Erklärung

**Problem mit SARSA:**  
In `Q(s',a')` hängt das Update von einer **zufällig gewählten** nächsten Aktion ab → hohe Varianz.

**Expected SARSA Lösung:**  
Statt einer zufälligen Aktion nehmen wir den **Erwartungswert über alle möglichen nächsten Aktionen**!

```
SARSA:
  Q(s,a) ← Q(s,a) + α[R + γ · Q(s',a')              - Q(s,a)]
                                       ↑
                              eine zufällige Aktion

Expected SARSA:
  Q(s,a) ← Q(s,a) + α[R + γ · Σ π(a'|s')·Q(s',a')  - Q(s,a)]
                                       ↑
                              Erwartungswert aller Aktionen
```

**Intuition:**
> Statt zu würfeln, was der Nachbar morgen tut – frage ihn, was er **durchschnittlich** tut! 🎲→📊

### Visualisierungsidee 🎨

```
SARSA Update:                   Expected SARSA Update:

    S' ──a'₁──► Q(s',a'₁)         S' ──a'₁──► Q(s',a'₁) × π(a'₁|s')
         └─── (zufällig!)               a'₂──► Q(s',a'₂) × π(a'₂|s')
                                         a'₃──► Q(s',a'₃) × π(a'₃|s')
                                                ─────────────────────
                                                         Summe ↑
```

### Beispiel – Expected SARSA

```python
def expected_sarsa(env, episodes=1000, gamma=0.9, alpha=0.1, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))

    def get_action_probs(state):
        """ε-greedy Wahrscheinlichkeiten für alle Aktionen"""
        actions = env.available_actions(state)
        n = len(actions)
        probs = {a: epsilon / n for a in actions}  # Basis-Exploration

        # Greedy-Aktion bekommt extra Wahrscheinlichkeit
        best_action = max(actions, key=lambda a: Q[state][a])
        probs[best_action] += (1 - epsilon)
        return probs

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # Aktion nach ε-greedy wählen
            action_probs = get_action_probs(state)
            action = np.random.choice(
                list(action_probs.keys()),
                p=list(action_probs.values())
            )

            next_state, reward, done = env.step(action)

            # Expected Value berechnen
            next_probs = get_action_probs(next_state)
            expected_q = sum(
                prob * Q[next_state][a]
                for a, prob in next_probs.items()
            )

            # Update mit Erwartungswert
            td_error = reward + gamma * expected_q - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state

    return Q
```

### SARSA vs. Expected SARSA Vergleich

| Aspekt | SARSA | Expected SARSA |
|---|---|---|
| **Target** | Q(s',a') (eine Aktion) | Σ π(a'\|s')·Q(s',a') (Erwartung) |
| **Varianz** | Höher | **Niedriger** |
| **Bias** | Gleich | Gleich |
| **Rechenaufwand** | Geringer | Höher (Σ über alle Aktionen) |
| **Stabilität** | Weniger stabil | **Stabiler** |

### Häufige Missverständnisse ⚠️

| Missverständnis | Richtigstellung |
|---|---|
| „Expected SARSA = greedy" | Nein! Die **Policy bleibt ε-greedy**, nur der Update nutzt den Erwartungswert |
| „Immer besser als SARSA" | In kleinen Aktionsräumen kaum Unterschied; bei großen: klar besser |
| „Gleich wie Q-Learning" | Nur wenn ε=0 (greedy)! Sonst andere Gewichtung |

### 🧩 Mini-Quiz

> **Frage 10:** Expected SARSA mit ε=0 (greedy Policy) – was ergibt sich?  
> **→ Q-Learning!** Wenn nur eine Aktion Wahrscheinlichkeit 1 hat (greedy), ist der Erwartungswert = max Q(s',a')

> **Frage 11:** Warum hat Expected SARSA geringere Varianz als SARSA?  
> **→** Weil wir **mitteln** statt einer **zufälligen Stichprobe** zu nehmen.

---

## 🔍 Einheit 11–12: Off-Policy – Q-Learning

### Intuitive Erklärung

**Kernidee:** Der Agent **verhält sich** nach einer Policy (behavior policy), lernt aber für eine **andere Policy** (target policy).

> 🔑 **Off-Policy:** „Ich beobachte einen anderen Fahrer, aber lerne für meine **optimale** Fahrweise."

**Q-Learning Update:**
```
Q(s,a) ← Q(s,a) + α · [R + γ · max_a' Q(s',a') - Q(s,a)]
                               └────────────────┘
                                  Bestes Q im nächsten Zustand
                                  (egal was tatsächlich gemacht wird!)
```

**Der Unterschied zu SARSA:**
```
SARSA:      R + γ · Q(s', a')        ← a' von ε-greedy Policy
Q-Learning: R + γ · max Q(s', a')    ← IMMER die beste Aktion!
```

### Visualisierungsidee 🎨

```
Q-Learning lernt IMMER die optimale greedy Policy:

Behavior Policy (ε-greedy):      Target Policy (greedy):

  S ──ε──► zufällige Aktion         S ──► max Q(s,a)
  S ──1-ε► beste Aktion

        ↓                                  ↓
  tatsächliche                    was beim Update
    Aktion                        angenommen wird

→ Q-Learning lernt Q* direkt – unabhängig von Exploration!
```

**Cliff Walking: Q-Learning vs. SARSA**
```
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  │  │  ← SARSA (vorsichtig, oben)
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│🤖│💀│💀│💀│💀│💀│💀│💀│💀│🏆│  ← Q-Learning (optimal, aber riskant!)
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Q-Learning: Lernt den OPTIMALEN (kurzen) Weg unten entlang
SARSA:      Lernt den SICHEREN Weg (wegen eigener ε-Fehler)
```

### Beispiel – Q-Learning

```python
def q_learning(env, episodes=1000, gamma=0.9, alpha=0.1, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))

    def epsilon_greedy(state):
        if np.random.random() < epsilon:
            return env.random_action()
        actions = env.available_actions(state)
        return max(actions, key=lambda a: Q[state][a])

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy(state)  # Behavior Policy
            next_state, reward, done = env.step(action)

            # Q-Learning Update: IMMER max über alle nächsten Aktionen
            next_actions = env.available_actions(next_state)
            max_next_q = max(Q[next_state][a] for a in next_actions)

            td_error = reward + gamma * max_next_q - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state

    return Q
```

### Q-Table Visualisierung

```python
def visualize_q_table(Q, grid_size=4):
    """Zeigt Pfeil-Richtungen basierend auf Q-Werten"""
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for i in range(grid_size):
        row = ""
        for j in range(grid_size):
            state = (i, j)
            if Q[state]:
                best_action = max(Q[state], key=Q[state].get)
                row += f" {arrows.get(best_action, '?')} "
            else:
                row += " · "
        print(row)
```

### Häufige Missverständnisse ⚠️

| Missverständnis | Richtigstellung |
|---|---|
| „Off-Policy ist immer besser" | Off-Policy kann mit **hoher Varianz** und Instabilität kämpfen |
| „Q-Learning findet immer die optimale Policy" | Nur mit ausreichender **Exploration** und Konvergenzbedingungen |
| „max Q ist greedy = kein Lernen mehr" | Die **Exploration** (ε) sorgt weiterhin für neue Erfahrungen |
| „SARSA und Q-Learning konvergieren zum Gleichen" | Q-Learning → Q*, SARSA → Qᵉ (unter ε-greedy Politik) |

### 🧩 Mini-Quiz

> **Frage 12:** Was ist der Hauptunterschied zwischen Q-Learning und SARSA im Update?  
> **→** Q-Learning nutzt `max Q(s',a')` (bestes Q), SARSA nutzt `Q(s',a')` (tatsächlich gewähltes Q)

> **Frage 13:** Ein Student sagt: „Q-Learning ist off-policy weil es eine neue Policy lernt."  
> Ist das richtig?  
> **→ Teilweise.** Es ist off-policy weil die **Behavior Policy** (ε-greedy) sich von der **Target Policy** (greedy/max) unterscheidet.

> **Frage 14:** Q(s,a) Tabelle:
> ```
> Q(S', links)  = 3
> Q(S', rechts) = 7
> Q(S', oben)   = 1
> ```
> Q-Learning Update: R=2, γ=0.9, α=0.1, Q(S,A)=4  
> → δ = 2 + 0.9×**7** - 4 = **4.3**  
> → Q(S,A) = 4 + 0.1×4.3 = **4.43**

---

## 🏁 Einheit 13–14: Gesamtvergleich & Praxis

### Der große Überblick

```
                    Lernt aus:
                   ┌──────────────────────┬────────────────┐
                   │  Komplette Episode   │  Jeden Schritt │
    ┌──────────────┼──────────────────────┼────────────────┤
    │  On-Policy   │   MC (on-policy)     │  SARSA         │
    │              │                      │  Expected SARSA│
    ├──────────────┼──────────────────────┼────────────────┤
    │  Off-Policy  │   MC (off-policy)    │  Q-Learning    │
    └──────────────┴──────────────────────┴────────────────┘
```

### Algorithmen-Vergleich

| Eigenschaft | MC | TD(0) | SARSA | Exp. SARSA | Q-Learning |
|---|---|---|---|---|---|
| **Bootstrapping** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Online Learning** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **On/Off Policy** | Beide | – | On | On | **Off** |
| **Varianz** | Hoch | Mittel | Mittel | **Niedrig** | Mittel |
| **Bias** | Kein | Mittel | Mittel | Mittel | Mittel |
| **Konvergenz-Ziel** | V(s) | V(s) | Qᵉ | Qᵉ | **Q*** |

### Wann welchen Algorithmus?

```
Entscheidungsbaum:

Ist die Task episodisch?
├── Nein → TD-Methoden (SARSA, Q-Learning)
└── Ja →
    Brauche ich schnelles Online-Learning?
    ├── Nein (und viele Daten) → Monte-Carlo
    └── Ja →
        Muss ich konservativ/sicher sein?
        ├── Ja (z.B. Roboter, echte Klippen) → SARSA
        └── Nein →
            Habe ich viele Aktionen?
            ├── Ja → Expected SARSA (geringere Varianz)
            └── Nein → Q-Learning (einfach, konvergiert zu Q*)
```

### Praxis-Projekt: Alle Algorithmen auf Gridworld

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class GridWorld:
    """4x4 Gridworld mit Ziel und Falle"""
    def __init__(self):
        self.size = 4
        self.start = (3, 0)
        self.goal  = (0, 3)
        self.trap  = (3, 3)
        self.actions      = [(-1,0), (1,0), (0,-1), (0,1)]
        self.action_names = ['↑', '↓', '←', '→']

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action_idx):
        dr, dc = self.actions[action_idx]
        r, c = self.state
        nr = max(0, min(self.size-1, r + dr))
        nc = max(0, min(self.size-1, c + dc))
        self.state = (nr, nc)

        if self.state == self.goal:
            return self.state, +10, True
        elif self.state == self.trap:
            return self.state, -10, True
        else:
            return self.state, -0.1, False

    def available_actions(self, state=None):
        return list(range(4))

    def random_action(self):
        return np.random.randint(4)


def run_comparison():
    env = GridWorld()
    results = {}

    algorithms = {
        'Q-Learning':     q_learning,
        'SARSA':          sarsa,
        'Expected SARSA': expected_sarsa,
    }

    for name, algo in algorithms.items():
        Q = algo(env, episodes=500)
        results[name] = Q
        print(f"\n{name} - Gelernte Policy:")
        visualize_q_table(Q)

    return results

results = run_comparison()
```

### Lernkurven visualisieren

```python
def plot_learning_curves(n_episodes=500, n_runs=10):
    env = GridWorld()

    all_rewards = {
        'Q-Learning': np.zeros((n_runs, n_episodes)),
        'SARSA':      np.zeros((n_runs, n_episodes)),
    }

    # ... Training und Plotting
    plt.figure(figsize=(10, 6))
    for name, rewards in all_rewards.items():
        mean_rewards = rewards.mean(axis=0)
        plt.plot(mean_rewards, label=name)

    plt.xlabel('Episode')
    plt.ylabel('Gesamtbelohnung')
    plt.title('Lernkurven: Q-Learning vs. SARSA')
    plt.legend()
    plt.show()
```

### Abschluss: Die RL-Landkarte

```
                    ╔═══════════════════════════════╗
                    ║   REINFORCEMENT LEARNING      ║
                    ╚═══════════════════════════════╝
                               │
               ┌───────────────┴───────────────┐
               │                               │
        Model-Based                      Model-Free
                                               │
                                  ┌────────────┴──────────┐
                                  │                        │
                           Value-Based               Policy-Based
                                  │                (→ weiterführend)
                        ┌─────────┴──────────┐
                        │                    │
                   Monte-Carlo          TD-Methods
                                             │
                                   ┌─────────┴─────────┐
                                   │                    │
                                On-Policy         Off-Policy
                                   │                    │
                            ┌──────┴──────┐        Q-Learning
                            │             │         (→ DQN)
                          SARSA      Exp. SARSA
```

### Häufige Missverständnisse – Gesamtüberblick ⚠️

| Missverständnis | Richtigstellung |
|---|---|
| „Mehr Exploration ist immer besser" | Zu viel Exploration → keine Ausnutzung. **Trade-off!** |
| „Konvergenz = gute Performance" | Konvergenz zu Qᵉ (SARSA) ≠ Konvergenz zu Q* |
| „Q-Learning ist Off-Policy wegen der max-Operation" | **Ja!** Das ist genau der Grund |
| „Mit genug Daten braucht man kein ε" | Ohne Exploration keine Entdeckung unbekannter Zustände |
| „TD und MC lernen das Gleiche" | Bei selben Daten: unterschiedliche Lösungen! TD = konsistenter mit MDP |

### 🧩 Abschluss-Quiz

> **Frage 15:** Welcher Algorithmus ist am besten für einen echten Roboter-Arm geeignet?  
> a) Q-Learning (schnell zur optimalen Policy)  
> b) SARSA (vorsichtig wegen exploration-Risiko)  
> c) MC (echte Returns)  
> **→ b) SARSA** – weil der Roboter bei Exploration nicht beschädigt werden soll!

> **Frage 16:** Das Bias-Varianz-Dilemma in RL:  
> - **Hohe Varianz** → typisch für: **MC**  
> - **Hoher Bias** → typisch für: **TD mit schlechter Initialisierung**

> **Frage 17:** Vervollständige den Update:  
> `Q(s,a) ← Q(s,a) + 0.1 · [3 + 0.9 · ___ - Q(s,a)]`  
> - Für Q-Learning:  → **max_a' Q(s',a')**  
> - Für SARSA:       → **Q(s', a')** (gewählte Aktion nach ε-greedy)

> **Frage 18 (Denksportaufgabe):** Expected SARSA mit ε=1 (vollständig zufällig)?  
> **→** Dann ist jede Aktion gleichwahrscheinlich → Expected SARSA = Durchschnitt aller Q(s',a')  
> Das nennt man **Random Walk Policy** – kaum nützlich, aber theoretisch interessant!

---

## 📖 Empfohlene Ressourcen

| Ressource | Typ | Niveau |
|---|---|---|
| Sutton & Barto – „Reinforcement Learning" (free online) | Buch | Standard |
| OpenAI Gymnasium (früher Gym) | Python Library | Praxis |
| David Silver RL Course (YouTube) | Video | Mittel |
| cleanrl GitHub | Code-Referenz | Praxis |

---

## 🔑 Key Takeaways

```
1. MC          → Lerne aus kompletten Episoden (kein Bias, hohe Varianz)
2. TD(0)       → Lerne aus jedem Schritt durch Bootstrapping
3. SARSA       → On-Policy TD für Q(s,a): vorsichtig, lernt unter ε-greedy
4. Exp. SARSA  → Wie SARSA, aber stabilere Updates (Erwartungswert)
5. Q-Learning  → Off-Policy: lernt Q* direkt (optimal, aber riskant bei ε)

   Alle Methoden → Model-Free, tabellarisch, ohne Neural Networks
```