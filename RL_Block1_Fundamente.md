# Block 1 – Fundamente des Reinforcement Learning
## *„Wie lernt ein Agent überhaupt?"*
### Umfang: ~20 Unterrichtsstunden

---

## Lernziele

Nach Abschluss dieses Blocks können die Studierenden:

- das RL-Paradigma im Kontext des maschinellen Lernens einordnen
- Markov Decision Processes formal definieren und modellieren
- Bellman-Gleichungen herleiten und interpretieren
- Dynamic Programming-Verfahren anwenden (Value Iteration, Policy Iteration)
- Monte Carlo- und Temporal-Difference-Methoden erklären und vergleichen
- einfache Algorithmen (Q-Learning, SARSA) implementieren

---

## Einheit 1.1 – RL im KI-Panorama *(2 Std.)*

### 1.1.1 Was ist Reinforcement Learning?

Reinforcement Learning (RL) ist ein Lernparadigma, bei dem ein **Agent** durch **Interaktion mit einer Umgebung** lernt, eine **Strategie (Policy)** zu optimieren, die den kumulierten **Reward** maximiert.

```
         Aktion a_t
Agent ─────────────────► Umgebung
  ▲                          │
  │    Zustand s_{t+1}       │
  └──────────────────────────┤
  │    Reward r_{t+1}        │
  └──────────────────────────┘
```

### 1.1.2 Einordnung im ML-Panorama

| Paradigma | Datenbasis | Ziel |
|---|---|---|
| Supervised Learning | Labeled Examples | Funktion approximieren |
| Unsupervised Learning | Unlabeled Data | Struktur entdecken |
| **Reinforcement Learning** | **Interaktion + Reward** | **Policy optimieren** |

### 1.1.3 Kernbegriffe

| Begriff | Symbol | Beschreibung |
|---|---|---|
| Agent | — | Lernende Entität |
| Umgebung | $\mathcal{E}$ | Alles außerhalb des Agenten |
| Zustand | $s \in \mathcal{S}$ | Beschreibung der Situation |
| Aktion | $a \in \mathcal{A}$ | Mögliche Entscheidungen |
| Reward | $r \in \mathbb{R}$ | Unmittelbares Feedback |
| Policy | $\pi$ | Strategie des Agenten |
| Return | $G_t$ | Kumulierter zukünftiger Reward |

### 1.1.4 Anwendungsbeispiele

- **Spiele**: AlphaGo, OpenAI Five (Dota 2), Atari Games
- **Robotik**: Manipulation, Laufen, Greifen
- **Autonomes Fahren**: Spurhalten, Überholmanöver
- **Finanzen**: Portfolio-Optimierung, Algorithmic Trading
- **NLP/LLM**: RLHF (Reinforcement Learning from Human Feedback)-(Natural Language Processing/Large Language Model)

---

## Einheit 1.2 – Markov Decision Processes (MDP) *(5 Std.)*

### 1.2.1 Die Markov-Eigenschaft

Ein Zustand $s_t$ hat die **Markov-Eigenschaft**, wenn die Zukunft vollständig durch die Gegenwart beschrieben wird – die Vergangenheit ist gegeben durch $s_t$ irrelevant:

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

> **Intuition:** Der aktuelle Zustand enthält alle relevanten Informationen. Die „Geschichte" muss nicht gespeichert werden.

### 1.2.2 Formale Definition eines MDP

Ein **Markov Decision Process** ist ein Tupel $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

| Komponente | Symbol | Bedeutung |
|---|---|---|
| Zustandsraum | $\mathcal{S}$ | Menge aller möglichen Zustände |
| Aktionsraum | $\mathcal{A}$ | Menge aller möglichen Aktionen |
| Transitionsmodell | $\mathcal{P}$ | $P(s' \mid s, a)$ – Übergangswahrscheinlichkeit |
| Reward-Funktion | $\mathcal{R}$ | $R(s, a, s')$ – unmittelbarer Reward |
| Diskontfaktor | $\gamma \in [0,1]$ | Gewichtung zukünftiger Rewards |

**Transitionsdynamik:**
$$\mathcal{P}_{ss'}^a = P(S_{t+1} = s' \mid S_t = s, A_t = a)$$

**Reward-Funktion:**
$$\mathcal{R}_s^a = \mathbb{E}\left[R_{t+1} \mid S_t = s, A_t = a\right]$$

### 1.2.3 Der Return

Der **Return** $G_t$ ist die gewichtete Summe aller zukünftigen Rewards ab Zeitschritt $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Rekursive Form:**

$$\boxed{G_t = R_{t+1} + \gamma G_{t+1}}$$

**Bedeutung des Diskontfaktors $\gamma$:**

| Wert | Verhalten |
|---|---|
| $\gamma = 0$ | Agent ist vollständig myopisch (nur sofortiger Reward) |
| $\gamma \to 1$ | Agent berücksichtigt weit entfernte Rewards gleichwertig |
| $0 < \gamma < 1$ | Balanciertes Verhalten (Standard in der Praxis) |

### 1.2.4 Policy

Eine **Policy** $\pi$ beschreibt das Verhalten des Agenten:

**Deterministische Policy:**
$$\pi(s) = a$$

**Stochastische Policy:**
$$\pi(a \mid s) = P(A_t = a \mid S_t = s)$$

### 1.2.5 Episodische vs. kontinuierliche Aufgaben

| Typ | Beschreibung | Beispiel |
|---|---|---|
| **Episodisch** | Klar definierter Endzustand $s_T$ | Schachspiel, Atari |
| **Kontinuierlich** | Kein natürlicher Abschluss | Robotersteuerung, Trading |

---

## Einheit 1.3 – Bellman-Gleichungen *(5 Std.)*

### 1.3.1 Value-Funktionen

**State-Value-Funktion** $V^\pi(s)$ – erwarteter Return bei Policy $\pi$ aus Zustand $s$:

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid S_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\Bigg|\; S_t = s\right]$$

**Action-Value-Funktion** $Q^\pi(s, a)$ – erwarteter Return bei Policy $\pi$, Zustand $s$, Aktion $a$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid S_t = s, A_t = a\right]$$

**Beziehung zwischen $V^\pi$ und $Q^\pi$:**

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \cdot Q^\pi(s, a)$$

### 1.3.2 Bellman Expectation Equations

Durch Einsetzen der rekursiven Return-Definition erhält man die **Bellman Expectation Equations**:

**Für $V^\pi$:**

$$\boxed{V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma V^\pi(s')\right]}$$

**Für $Q^\pi$:**

$$\boxed{Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\right]}$$

> **Kern-Idee:** Der Wert eines Zustands setzt sich zusammen aus dem **unmittelbaren Reward** und dem **diskontierten Wert des Nachfolgezustands**.

### 1.3.3 Bellman Optimality Equations

Die **optimale Value-Funktion** $V^*(s)$ ist die maximal erreichbare Value über alle Policies:

$$V^*(s) = \max_\pi V^\pi(s)$$

**Bellman Optimality Equation für $V^*$:**

$$\boxed{V^*(s) = \max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma V^*(s')\right]}$$

**Bellman Optimality Equation für $Q^*$:**

$$\boxed{Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\right]}$$

**Optimale Policy aus $Q^*$:**

$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s, a)$$

### 1.3.4 Backup-Diagramme

```
      s                        s, a
      │                          │
   ┌──┴──┐                    ┌──┴──┐
  a│     │a                 s'│     │s'
   │     │                    │     │
  s'    s'                  s',a'  s',a'

V pi(s) Backup          Q pi(s,a) Backup
```

---

## Einheit 1.4 – Dynamic Programming *(4 Std.)*

> **Voraussetzung:** Vollständiges Modell der Umgebung ($\mathcal{P}$, $\mathcal{R}$) bekannt.

### 1.4.1 Policy Evaluation (Prediction)

Gegeben Policy $\pi$, berechne $V^\pi$ durch **iterative Anwendung** der Bellman Expectation Equation:

$$V_{k+1}(s) \leftarrow \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma V_k(s')\right]$$

**Algorithmus: Iterative Policy Evaluation**

```
Initialisierung: V(s) = 0 für alle s ∈ S
Wiederhole:
    Δ ← 0
    Für jedes s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a)[R(s,a,s') + γ·V(s')]
        Δ ← max(Δ, |v - V(s)|)
Bis Δ < θ  (Konvergenzschwelle)
```

**Konvergenz:** $V_k \to V^\pi$ für $k \to \infty$ (bei $\gamma < 1$ oder episodischen MDPs)

### 1.4.2 Policy Improvement

Gegeben $V^\pi$, verbessere Policy $\pi$ durch **greedy Verbesserung**:

$$\pi'(s) = \arg\max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma V^\pi(s')\right]$$

**Policy Improvement Theorem:**

$$V^{\pi'}(s) \geq V^\pi(s) \quad \forall s \in \mathcal{S}$$

### 1.4.3 Policy Iteration

Abwechselnde Anwendung von Evaluation und Improvement bis zur Konvergenz:

$$\pi_0 \xrightarrow{\text{Eval}} V^{\pi_0} \xrightarrow{\text{Improve}} \pi_1 \xrightarrow{\text{Eval}} V^{\pi_1} \xrightarrow{\text{Improve}} \ldots \xrightarrow{} \pi^* $$

### 1.4.4 Value Iteration

Kombiniert Evaluation und Improvement in **einem Schritt** – direktes Anwenden der Bellman Optimality Equation:

$$\boxed{V_{k+1}(s) \leftarrow \max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma V_k(s')\right]}$$

**Algorithmus: Value Iteration**

```
Initialisierung: V(s) = 0 für alle s ∈ S
Wiederhole:
    Δ ← 0
    Für jedes s ∈ S:
        v ← V(s)
        V(s) ← max_a Σ_s' P(s'|s,a)[R(s,a,s') + γ·V(s')]
        Δ ← max(Δ, |v - V(s)|)
Bis Δ < θ
Policy extrahieren: π*(s) = argmax_a Σ_s' P(s'|s,a)[R(s,a,s') + γ·V(s')]
```

### 1.4.5 Vergleich: Policy Iteration vs. Value Iteration

| Kriterium | Policy Iteration | Value Iteration |
|---|---|---|
| Konvergenzgeschwindigkeit | Schneller (weniger Iterationen) | Langsamer |
| Aufwand pro Iteration | Höher (vollständige Evaluation) | Geringer |
| Berechnung der Policy | Explizit nach jeder Evaluation | Erst am Ende |
| Geeignet für | Kleine Zustandsräume | Größere Zustandsräume |

---

## Einheit 1.5 – Monte Carlo & Temporal Difference *(4 Std.)*

> **Motivation:** DP benötigt ein vollständiges Modell. Was, wenn $\mathcal{P}$ unbekannt ist?

### 1.5.1 Monte Carlo (MC) Methods

**Idee:** Schätze $V^\pi(s)$ durch **Mittelung realer Returns** aus vollständigen Episoden.

**First-Visit MC:**

$$V(s) \leftarrow V(s) + \frac{1}{N(s)}\left(G_t - V(s)\right)$$

Mit inkrementeller Update-Regel (konstantes $\alpha$):

$$V(s) \leftarrow V(s) + \alpha \left(G_t - V(s)\right)$$

**Eigenschaften:**
- Kein Modell der Umgebung nötig (model-free)
- Nur bei episodischen Aufgaben anwendbar
- Hohe Varianz, kein Bias
- Update erst nach Episodenende

### 1.5.2 Temporal Difference (TD) Learning

**Idee:** Kombiniere MC (kein Modell nötig) mit DP (bootstrapping): Update nach **jedem Zeitschritt** basierend auf dem **TD-Target**.

**TD(0) Update:**

$$\boxed{V(s_t) \leftarrow V(s_t) + \alpha \underbrace{\left[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD-Target}} - V(s_t)\right]}_{\text{TD-Fehler } \delta_t}}$$

**TD-Fehler:**

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

### 1.5.3 Vergleich: MC vs. TD vs. DP

| Eigenschaft | DP | MC | TD |
|---|---|---|---|
| Modell nötig? | ✅ Ja | ❌ Nein | ❌ Nein |
| Bootstrapping? | ✅ Ja | ❌ Nein | ✅ Ja |
| Update-Zeitpunkt | Sweep | Episodenende | Jeder Schritt |
| Varianz | Niedrig | Hoch | Mittel |
| Bias | Kein | Kein | Ja (anfangs) |
| Episodisch nötig? | ❌ Nein | ✅ Ja | ❌ Nein |

### 1.5.4 SARSA – On-Policy TD Control

**SARSA** lernt $Q^\pi(s,a)$ direkt (on-policy):

$$\boxed{Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]}$$

Name: **S**tate – **A**ction – **R**eward – **S**tate – **A**ction

### 1.5.5 Q-Learning – Off-Policy TD Control

**Q-Learning** lernt die optimale $Q^*$ direkt (off-policy):

$$\boxed{Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]}$$

**Algorithmus: Q-Learning (Watkins, 1989)**

```
Initialisierung: Q(s, a) = 0 für alle s, a
Für jede Episode:
    s ← Startzustand
    Wiederhole:
        a ← ε-greedy aus Q(s, ·)       # Exploration-Policy
        Führe a aus, beobachte r, s'
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        s ← s'
    Bis s ist Terminalzustand
```

### 1.5.6 Exploration vs. Exploitation

Das grundlegende **Exploration-Exploitation-Dilemma**:

| Strategie | Formel / Beschreibung |
|---|---|
| **$\varepsilon$-greedy** | Wähle $\arg\max_a Q(s,a)$ mit $P = 1-\varepsilon$, zufällige Aktion mit $P = \varepsilon$ |
| **$\varepsilon$-decay** | $\varepsilon_t = \varepsilon_0 \cdot e^{-\lambda t}$ – Exploration nimmt über Zeit ab |
| **Softmax / Boltzmann** | $\pi(a \mid s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$ |
| **UCB** | $a = \arg\max_a \left[Q(s,a) + c\sqrt{\frac{\ln t}{N(s,a)}}\right]$ |

### 1.5.7 On-Policy vs. Off-Policy

| | On-Policy (SARSA) | Off-Policy (Q-Learning) |
|---|---|---|
| Lernziel | Aktuelle Policy $\pi$ | Optimale Policy $\pi^*$ |
| Verwendete Daten | Aktuelle Policy | Beliebige Policy (z.B. $\varepsilon$-greedy) |
| Konservativismus | Sicherer in gefährlichen Umgebungen | Effizienter im Allgemeinen |

---

## Zusammenfassung Block 1

```
Markov-Eigenschaft
        │
        ▼
   MDP-Modell (S, A, P, R, γ)
        │
        ▼
  Bellman Equations
  ┌─────┴─────┐
Expectation  Optimality
     │            │
     ▼            ▼
  Policy     Value
 Iteration  Iteration    ← Dynamic Programming (Modell bekannt)
                │
                ▼
        TD / MC Methods    ← Model-Free (Modell unbekannt)
        ┌──────┴──────┐
      SARSA       Q-Learning
   (On-Policy)  (Off-Policy)
```

---

## Weiterführende Literatur

| Quelle | Empfehlung |
|---|---|
| Sutton & Barto – *Reinforcement Learning: An Introduction* (2018) | Kapitel 1–6 ⭐ Standardwerk |
| David Silver – *UCL RL Lectures* (YouTube) | Lectures 1–4 |
| Csaba Szepesvári – *Algorithms for Reinforcement Learning* | Für formale Vertiefung |
| OpenAI Spinning Up | Praktische Implementierungen |

---

## Übungsaufgaben Block 1

1. **MDP modellieren:** Modellieren Sie das Gridworld-Problem als MDP. Definieren Sie $\mathcal{S}$, $\mathcal{A}$, $\mathcal{P}$, $\mathcal{R}$, $\gamma$.

2. **Bellman-Gleichung:** Leiten Sie die Bellman Expectation Equation für $Q^\pi$ aus der Definition des Returns her.

3. **Policy Iteration:** Führen Sie Policy Iteration auf einem $4 \times 4$ Gridworld-Beispiel von Hand durch (2 Iterationen).

4. **Q-Learning implementieren:** Implementieren Sie Q-Learning in Python für das FrozenLake-Environment (OpenAI Gym). Visualisieren Sie die Konvergenz der Q-Werte.

5. **Vergleich SARSA vs. Q-Learning:** Testen Sie beide Algorithmen auf dem CliffWalking-Environment. Erklären Sie die beobachteten Unterschiede.

6. **Exploration:** Vergleichen Sie $\varepsilon$-greedy mit $\varepsilon$-decay auf dem Bandit-Problem. Welcher Ansatz konvergiert schneller?
