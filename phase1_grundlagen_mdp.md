# Phase 1 · Grundlagen & MDP-Framework
**Umfang:** 14 Stunden · Module M1–M4  
**Zielgruppe:** Fortgeschrittene mit Deep-Learning-Kenntnissen

---

## Lernziele dieser Phase

Nach Abschluss von Phase 1 können die Teilnehmer:
- Den Unterschied zwischen RL, Supervised und Unsupervised Learning erklären
- Ein reales Problem als MDP formalisieren (Zustandsraum, Aktionsraum, Reward, Übergangsmodell)
- Die Bellman-Gleichungen für V* und Q* herleiten und anwenden
- Policy Iteration und Value Iteration implementieren und deren Konvergenz begründen

---

## M1 · Einführung & Motivation *(2 Stunden)*

### Was ist Reinforcement Learning?

Reinforcement Learning ist das dritte Paradigma des maschinellen Lernens neben Supervised und Unsupervised Learning. Ein **Agent** interagiert mit einer **Umgebung**, führt **Aktionen** aus und erhält **Belohnungen** (Rewards) als Feedback. Das Ziel: die kumulative Belohnung über die Zeit zu maximieren.

```
Das RL-Grundprinzip:

  Agent ──(Aktion a_t)──▶ Umgebung
    ◀──(Zustand s_{t+1})──
    ◀──(Reward r_{t+1}) ──
```

<!-- ILLUSTRATION: RL-Kreislauf -->
<svg width="100%" viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Agent Box -->
  <rect x="60" y="90" width="160" height="80" rx="12" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="140" y="124" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="600" fill="#3C3489">Agent</text>
  <text x="140" y="146" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#534AB7">Policy π(a|s)</text>
  <!-- Environment Box -->
  <rect x="460" y="90" width="160" height="80" rx="12" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="540" y="124" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="600" fill="#085041">Umgebung</text>
  <text x="540" y="146" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1D9E75">Zustand, Reward</text>
  <!-- Arrow: Aktion -->
  <path d="M220 110 Q340 60 460 110" fill="none" stroke="#534AB7" stroke-width="2" marker-end="url(#arr1)"/>
  <rect x="295" y="48" width="90" height="26" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
  <text x="340" y="65" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#3C3489">Aktion aₜ</text>
  <!-- Arrow: Zustand -->
  <path d="M460 150 Q340 200 220 150" fill="none" stroke="#1D9E75" stroke-width="2" marker-end="url(#arr1)"/>
  <rect x="285" y="178" width="110" height="26" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.8"/>
  <text x="340" y="195" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#085041">Zustand sₜ₊₁</text>
  <!-- Arrow: Reward -->
  <path d="M460 165 Q340 220 220 165" fill="none" stroke="#D85A30" stroke-width="2" marker-end="url(#arr1)"/>
  <rect x="290" y="210" width="100" height="26" rx="6" fill="#FAECE7" stroke="#D85A30" stroke-width="0.8"/>
  <text x="340" y="227" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#712B13">Reward rₜ₊₁</text>
  <!-- Zeit -->
  <text x="340" y="22" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#888780">Zeitschritt t → t+1 → t+2 → …</text>
</svg>

### RL vs. Supervised Learning vs. Unsupervised Learning

| | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|---|---|---|---|
| **Datenbasis** | Gelabelte Beispiele | Ungelabelte Daten | Interaktion mit Umgebung |
| **Feedback** | Direktes Label | Kein Feedback | Verzögerter Reward |
| **Ziel** | Funktion approximieren | Struktur finden | Kumulative Belohnung max. |
| **Herausforderung** | Generalisierung | Cluster-Validierung | Credit Assignment, Exploration |

### Wichtige Anwendungsfelder

- 🎮 **Games:** AlphaGo, AlphaZero, OpenAI Five (Dota 2), Atari-Benchmark
- 🤖 **Robotik:** Motorisches Lernen, Greifaufgaben, Navigation
- 🗣️ **LLMs & RLHF:** ChatGPT, Claude, Gemini – Policy-Finetuning aus menschlichem Feedback
- 💊 **Medizin:** Personalisierte Behandlungspfade, Dosierungsoptimierung
- 🏭 **Industrie:** Energieoptimierung, Lagerhaltung, autonomes Fahren

---

## M2 · Markov-Entscheidungsprozesse (MDP) *(4 Stunden)*

### Formale Definition

Ein **Markov Decision Process (MDP)** ist definiert durch das Tupel:

$$\mathcal{M} = (\mathcal{S},\ \mathcal{A},\ P,\ R,\ \gamma)$$

| Symbol | Bedeutung | Beispiel (Schach) |
|--------|-----------|-------------------|
| $\mathcal{S}$ | Zustandsraum | Alle möglichen Brettstellungen |
| $\mathcal{A}$ | Aktionsraum | Alle legalen Züge |
| $P(s'\|s,a)$ | Übergangswahrscheinlichkeit | Deterministisch: 1.0 |
| $R(s,a,s')$ | Reward-Funktion | +1 Gewinn, -1 Verlust, 0 sonst |
| $\gamma \in [0,1]$ | Discount-Faktor | z.B. 0.99 |

### Die Markov-Eigenschaft

> **Ein Zustand ist Markov, wenn die Zukunft bedingt unabhängig von der Vergangenheit ist, gegeben der aktuelle Zustand.**

$$P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t)$$

<!-- ILLUSTRATION: MDP-Struktur -->
<svg width="100%" viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- States -->
  <circle cx="100" cy="150" r="38" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="100" y="145" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#0C447C">s₀</text>
  <text x="100" y="162" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">Start</text>

  <circle cx="340" cy="100" r="38" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="340" y="95" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#085041">s₁</text>
  <text x="340" y="112" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">Folge-Zustand</text>

  <circle cx="340" cy="220" r="38" fill="#FAECE7" stroke="#D85A30" stroke-width="1.5"/>
  <text x="340" y="215" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#712B13">s₂</text>
  <text x="340" y="232" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#D85A30">Alt. Zustand</text>

  <circle cx="580" cy="150" r="38" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="580" y="145" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#3C3489">s₃</text>
  <text x="580" y="162" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">Terminal</text>

  <!-- Arrows with probabilities -->
  <path d="M136 130 L304 108" fill="none" stroke="#1D9E75" stroke-width="1.5" marker-end="url(#arr2)"/>
  <text x="200" y="108" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#1D9E75">a₀, P=0.7</text>
  <text x="200" y="121" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#5F5E5A">r=+5</text>

  <path d="M136 168 L304 212" fill="none" stroke="#D85A30" stroke-width="1.5" marker-end="url(#arr2)"/>
  <text x="200" y="206" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#D85A30">a₀, P=0.3</text>
  <text x="200" y="219" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#5F5E5A">r=-1</text>

  <path d="M378 108 L542 138" fill="none" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arr2)"/>
  <text x="470" y="116" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#534AB7">a₁, P=1.0</text>
  <text x="470" y="129" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#5F5E5A">r=+10</text>

  <path d="M378 212 L542 162" fill="none" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arr2)"/>
  <text x="470" y="200" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#534AB7">a₂, P=1.0</text>
  <text x="470" y="213" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#5F5E5A">r=+2</text>

  <!-- Legend -->
  <rect x="40" y="258" width="600" height="30" rx="6" fill="#F1EFE8" stroke="#D3D1C7" stroke-width="0.5"/>
  <text x="340" y="277" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5F5E5A">Kreise = Zustände · Pfeile = Übergänge P(s'|s,a) · Kantengewicht = Reward r(s,a,s')</text>
</svg>

### Discount-Faktor γ

Der Discount-Faktor steuert, wie stark **zukünftige** Rewards gegenüber **sofortigen** gewichtet werden:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

<!-- ILLUSTRATION: Discount-Kurve -->
<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="60" y1="160" x2="620" y2="160" stroke="#888780" stroke-width="1" marker-end="url(#arr3)"/>
  <line x1="60" y1="20" x2="60" y2="165" stroke="#888780" stroke-width="1" marker-end="url(#arr3)"/>
  <text x="630" y="164" font-family="sans-serif" font-size="11" fill="#888780">t</text>
  <text x="40" y="24" font-family="sans-serif" font-size="11" fill="#888780">γᵏ</text>
  <text x="48" y="164" font-family="sans-serif" font-size="10" fill="#888780">0</text>
  <text x="48" y="30" font-family="sans-serif" font-size="10" fill="#888780">1</text>
  <!-- γ=0.99 curve (slow decay) -->
  <polyline points="60,30 130,32 200,36 270,42 340,51 410,63 480,79 550,100 610,122"
    fill="none" stroke="#1D9E75" stroke-width="2"/>
  <text x="615" y="118" font-family="sans-serif" font-size="11" fill="#1D9E75">γ=0.99</text>
  <!-- γ=0.9 curve -->
  <polyline points="60,30 130,38 200,53 270,72 340,96 410,117 480,135 550,148 610,155"
    fill="none" stroke="#534AB7" stroke-width="2"/>
  <text x="615" y="151" font-family="sans-serif" font-size="11" fill="#534AB7">γ=0.9</text>
  <!-- γ=0.5 curve (fast decay) -->
  <polyline points="60,30 130,65 200,97 270,121 340,138 410,149 480,155 550,158 610,159"
    fill="none" stroke="#D85A30" stroke-width="2"/>
  <text x="615" y="138" font-family="sans-serif" font-size="11" fill="#D85A30">γ=0.5</text>
  <!-- t labels -->
  <text x="60" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">0</text>
  <text x="130" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">1</text>
  <text x="200" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">2</text>
  <text x="270" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">3</text>
  <text x="340" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">4</text>
  <text x="410" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">5</text>
  <text x="480" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">6</text>
  <text x="550" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">7</text>
</svg>

- **γ = 0:** Nur der sofortige Reward zählt (rein gierig)
- **γ → 1:** Langzeitorientiert (alle zukünftigen Rewards gleich gewichtet)
- **Typische Werte:** 0.95 – 0.999 je nach Problemhorizont

---

## M3 · Bellman-Gleichungen *(4 Stunden)*

### State-Value-Funktion V(s)

Die **State-Value-Funktion** $V^\pi(s)$ gibt den erwarteten kumulativen Reward an, wenn der Agent in Zustand $s$ startet und Policy $\pi$ folgt:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t = s\right]$$

### Action-Value-Funktion Q(s,a)

Die **Action-Value-Funktion** $Q^\pi(s,a)$ gibt den erwarteten Reward an, wenn Aktion $a$ in Zustand $s$ gewählt wird und dann $\pi$ gefolgt wird:

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t = s, a_t = a\right]$$

### Bellman-Gleichung (rekursive Form)

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]$$

<!-- ILLUSTRATION: Bellman Backup Tree -->
<svg width="100%" viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Root state s -->
  <circle cx="340" cy="40" r="28" fill="#EEEDFE" stroke="#534AB7" stroke-width="2"/>
  <text x="340" y="35" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#3C3489">s</text>
  <text x="340" y="52" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">V(s) = ?</text>
  <!-- Action nodes -->
  <circle cx="200" cy="140" r="22" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="200" y="136" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#085041">a₁</text>
  <text x="200" y="151" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1D9E75">π(a₁|s)</text>
  <circle cx="480" cy="140" r="22" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="480" y="136" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#085041">a₂</text>
  <text x="480" y="151" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1D9E75">π(a₂|s)</text>
  <!-- Lines to action nodes -->
  <line x1="320" y1="64" x2="218" y2="120" stroke="#888780" stroke-width="1.2" marker-end="url(#arr4)"/>
  <line x1="360" y1="64" x2="462" y2="120" stroke="#888780" stroke-width="1.2" marker-end="url(#arr4)"/>
  <!-- Successor states from a1 -->
  <circle cx="110" cy="230" r="24" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="110" y="225" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">s₁'</text>
  <text x="110" y="241" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">V(s₁')</text>
  <circle cx="230" cy="230" r="24" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="230" y="225" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">s₂'</text>
  <text x="230" y="241" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">V(s₂')</text>
  <!-- Successor states from a2 -->
  <circle cx="450" cy="230" r="24" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="450" y="225" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">s₃'</text>
  <text x="450" y="241" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">V(s₃')</text>
  <circle cx="570" cy="230" r="24" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="570" y="225" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">s₄'</text>
  <text x="570" y="241" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">V(s₄')</text>
  <!-- Lines to successor states -->
  <line x1="184" y1="160" x2="128" y2="208" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr4)"/>
  <line x1="200" y1="162" x2="220" y2="208" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr4)"/>
  <line x1="468" y1="160" x2="462" y2="208" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr4)"/>
  <line x1="490" y1="160" x2="554" y2="208" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr4)"/>
  <!-- Reward labels -->
  <text x="146" y="192" font-family="sans-serif" font-size="9" fill="#D85A30">r+γV(s₁')</text>
  <text x="206" y="192" font-family="sans-serif" font-size="9" fill="#D85A30">r+γV(s₂')</text>
  <text x="424" y="192" font-family="sans-serif" font-size="9" fill="#D85A30">r+γV(s₃')</text>
  <text x="498" y="192" font-family="sans-serif" font-size="9" fill="#D85A30">r+γV(s₄')</text>
  <!-- Legend -->
  <text x="340" y="272" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#888780">Bellman Backup: V(s) ist der gewichtete Durchschnitt über Aktionen und Nachfolgezustände</text>
</svg>

### Optimalitätsbedingung (Bellman-Optimalitätsgleichung)

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^*(s')\right]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right]$$

> **Wichtig:** Die optimale Policy ist direkt aus Q* ableitbar: $\pi^*(s) = \arg\max_a Q^*(s,a)$

---

## M4 · Dynamische Programmierung *(4 Stunden)*

### Policy Evaluation

**Ziel:** Für eine gegebene Policy $\pi$ die Value-Funktion $V^\pi$ berechnen.

**Algorithmus:** Iterative Policy Evaluation (Synchrones Update)

```python
def policy_evaluation(pi, P, R, gamma, theta=1e-6):
    V = {s: 0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # Bellman-Update
            V[s] = sum(
                pi[s][a] * sum(
                    P[s][a][s_] * (R[s][a][s_] + gamma * V[s_])
                    for s_ in states
                )
                for a in actions
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V
```

### Policy Iteration

<!-- ILLUSTRATION: Policy Iteration Kreislauf -->
<svg width="100%" viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Start -->
  <rect x="30" y="80" width="110" height="60" rx="10" fill="#F1EFE8" stroke="#888780" stroke-width="1.5"/>
  <text x="85" y="106" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#2C2C2A">Initiale</text>
  <text x="85" y="124" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#2C2C2A">Policy π₀</text>
  <!-- Policy Evaluation -->
  <rect x="200" y="80" width="140" height="60" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="270" y="104" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#085041">Policy</text>
  <text x="270" y="120" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#085041">Evaluation</text>
  <text x="270" y="136" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">→ V^π berechnen</text>
  <!-- Policy Improvement -->
  <rect x="400" y="80" width="140" height="60" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="470" y="104" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#3C3489">Policy</text>
  <text x="470" y="120" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#3C3489">Improvement</text>
  <text x="470" y="136" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">→ π' = greedy(V^π)</text>
  <!-- Convergence -->
  <rect x="574" y="80" width="80" height="60" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="614" y="104" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">π = π'?</text>
  <text x="614" y="124" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">Konvergenz</text>
  <!-- Arrows -->
  <line x1="140" y1="110" x2="198" y2="110" stroke="#888780" stroke-width="1.5" marker-end="url(#arr5)"/>
  <line x1="340" y1="110" x2="398" y2="110" stroke="#888780" stroke-width="1.5" marker-end="url(#arr5)"/>
  <line x1="540" y1="110" x2="572" y2="110" stroke="#888780" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- Feedback loop -->
  <path d="M 614 140 Q 614 185 340 185 Q 200 185 200 142" fill="none" stroke="#D85A30" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arr5)"/>
  <text x="340" y="200" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#D85A30">Wiederhole bis Konvergenz (Policy Improvement Theorem garantiert π* in endl. Schritten)</text>
  <!-- Ja / Nein -->
  <text x="645" y="78" font-family="sans-serif" font-size="11" font-weight="600" fill="#1D9E75">Ja → π*</text>
  <text x="558" y="150" font-family="sans-serif" font-size="11" fill="#D85A30">Nein</text>
</svg>

### Value Iteration

Kombiniert Evaluation und Improvement in einem einzigen Schritt:

$$V_{k+1}(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V_k(s')\right]$$

```python
def value_iteration(P, R, gamma, theta=1e-6):
    V = {s: 0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # Bellman-Optimalitäts-Update (max statt sum)
            V[s] = max(
                sum(P[s][a][s_] * (R[s][a][s_] + gamma * V[s_])
                    for s_ in states)
                for a in actions
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # Policy ableiten
    pi = {s: argmax_a(Q(s,a)) for s in states}
    return V, pi
```

### Vergleich: Policy Iteration vs. Value Iteration

| | Policy Iteration | Value Iteration |
|---|---|---|
| **Pro** | Konvergiert in wenigen Iterationen | Einfachere Implementierung |
| **Con** | Jeder Schritt teuer (Eval-Loop) | Mehr Iterationen nötig |
| **Komplexität** | $O(\|S\|^2 \cdot \|A\|)$ pro Iteration | $O(\|S\|^2 \cdot \|A\|)$ pro Iteration |
| **Anwendung** | Kleine MDPs mit wenigen Aktionen | Große Zustandsräume |

---

## Übungsaufgaben Phase 1

1. **MDP-Formalisierung:** Modelliere das Spiel *Frozen Lake* (4×4 Grid) als MDP. Definiere $\mathcal{S}$, $\mathcal{A}$, $P$, $R$ und $\gamma$.

2. **Bellman-Berechnung (von Hand):** Gegeben ein 3-Zustands-MDP mit bekannten Übergangswahrscheinlichkeiten – berechne $V^\pi$ für die gleichmäßige Policy ($\pi(a|s) = 0.5$).

3. **Policy Iteration implementieren:** Implementiere Policy Iteration für Frozen Lake in Python (mit `gymnasium`) und visualisiere die konvergierte Value-Funktion als Heatmap.

4. **Value Iteration + Vergleich:** Implementiere Value Iteration für dasselbe Problem und vergleiche Konvergenzgeschwindigkeit und Qualität der resultierenden Policy.

5. **Discount-Einfluss:** Untersuche experimentell, wie sich $\gamma \in \{0.5, 0.9, 0.99\}$ auf die optimale Policy und die konvergierten V-Werte auswirkt.

---

## Empfohlene Literatur (Phase 1)

- **Sutton & Barto, Kap. 1–4** – Grundlegendes, MDP, Bellman, DP
- **David Silver, Lecture 1–3** – UCL Course on RL (YouTube)
- **OpenAI Spinning Up:** [spinningup.openai.com](https://spinningup.openai.com) – Einführung in RL-Konzepte
- **Gymnasium Docs:** [gymnasium.farama.org](https://gymnasium.farama.org) – Frozen Lake, Taxi, GridWorld

---

*→ Weiter mit [Phase 2: Klassisches RL](./phase2_klassisches_rl.md)*
