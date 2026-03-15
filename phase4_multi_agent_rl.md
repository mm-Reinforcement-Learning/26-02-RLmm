# Phase 4 · Multi-Agent Reinforcement Learning
**Umfang:** 14 Stunden · Module M15–M18  
**Voraussetzung:** Phase 3 abgeschlossen (Deep RL, PPO, Actor-Critic)

---

## Lernziele dieser Phase

Nach Abschluss von Phase 4 können die Teilnehmer:
- Ein Multi-Agent-Problem als Stochastic Game formalisieren
- Nash-Gleichgewichte berechnen und in RL-Kontexten interpretieren
- QMIX, MADDPG und MAPPO implementieren
- Cooperative vs. Competitive Settings unterscheiden und passende Algorithmen wählen
- Self-Play und Population-Based Training konzeptuell erklären

---

## M15 · MARL Grundlagen *(4 Stunden)*

### Von MDP zu Stochastic Game

Ein **Stochastic Game** (Markov Game) erweitert das MDP auf $n$ Agenten:

$$\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \mathcal{A}^1, \ldots, \mathcal{A}^n, P, R^1, \ldots, R^n, \gamma \rangle$$

| Symbol | Bedeutung |
|--------|-----------|
| $\mathcal{N} = \{1, \ldots, n\}$ | Menge der Agenten |
| $\mathcal{A}^i$ | Aktionsraum von Agent $i$ |
| $\mathbf{a} = (a^1, \ldots, a^n)$ | Joint Action aller Agenten |
| $P(s' \| s, \mathbf{a})$ | Übergang abhängig von allen Aktionen |
| $R^i(s, \mathbf{a}, s')$ | Individuelle Reward-Funktion |

### Spieltypen

<!-- ILLUSTRATION: Cooperative vs Competitive vs Mixed -->
<svg width="100%" viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg">
  <!-- Cooperative -->
  <rect x="20" y="40" width="190" height="140" rx="12" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="115" y="68" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#085041">Cooperative</text>
  <!-- Agents pointing together -->
  <circle cx="75" cy="120" r="20" fill="#1D9E75"/>
  <text x="75" y="125" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">A1</text>
  <circle cx="155" cy="120" r="20" fill="#1D9E75"/>
  <text x="155" y="125" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">A2</text>
  <circle cx="115" cy="165" r="15" fill="#9FE1CB"/>
  <text x="115" y="170" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#085041">Ziel</text>
  <line x1="90" y1="134" x2="105" y2="155" stroke="#085041" stroke-width="1.5"/>
  <line x1="140" y1="134" x2="122" y2="155" stroke="#085041" stroke-width="1.5"/>
  <text x="115" y="195" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">R¹ = R² (gemeinsam)</text>
  <text x="115" y="208" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">z.B. Fußball-Team, Roboter</text>

  <!-- Competitive -->
  <rect x="245" y="40" width="190" height="140" rx="12" fill="#FCEBEB" stroke="#A32D2D" stroke-width="1.5"/>
  <text x="340" y="68" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#501313">Competitive</text>
  <circle cx="290" cy="120" r="20" fill="#E24B4A"/>
  <text x="290" y="125" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">A1</text>
  <circle cx="390" cy="120" r="20" fill="#E24B4A"/>
  <text x="390" y="125" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">A2</text>
  <line x1="310" y1="115" x2="370" y2="115" stroke="#501313" stroke-width="2"/>
  <line x1="350" y1="108" x2="370" y2="115" stroke="#501313" stroke-width="2"/>
  <line x1="350" y1="122" x2="370" y2="115" stroke="#501313" stroke-width="2"/>
  <line x1="330" y1="108" x2="310" y2="115" stroke="#501313" stroke-width="2"/>
  <line x1="330" y1="122" x2="310" y2="115" stroke="#501313" stroke-width="2"/>
  <text x="340" y="155" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#A32D2D">R¹ + R² = 0 (Zero-Sum)</text>
  <text x="340" y="195" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#A32D2D">R¹ = −R²</text>
  <text x="340" y="208" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">z.B. Schach, Poker</text>

  <!-- Mixed -->
  <rect x="470" y="40" width="190" height="140" rx="12" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="565" y="68" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#3C3489">Mixed</text>
  <circle cx="520" cy="110" r="18" fill="#7F77DD"/>
  <text x="520" y="115" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">A1</text>
  <circle cx="610" cy="110" r="18" fill="#7F77DD"/>
  <text x="610" y="115" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">A2</text>
  <circle cx="565" cy="150" r="18" fill="#AFA9EC"/>
  <text x="565" y="155" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#3C3489">A3</text>
  <line x1="535" y1="122" x2="552" y2="137" stroke="#3C3489" stroke-width="1.2"/>
  <line x1="595" y1="122" x2="578" y2="137" stroke="#3C3489" stroke-width="1.2"/>
  <line x1="535" y1="110" x2="592" y2="110" stroke="#A32D2D" stroke-width="1.2" stroke-dasharray="3,2"/>
  <text x="565" y="195" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">Teil-coop., Teil-compet.</text>
  <text x="565" y="208" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">z.B. Capture the Flag</text>
</svg>

### Nash-Gleichgewicht

Ein **Nash-Gleichgewicht** ist eine Strategie-Kombination $(\pi^{1*}, \ldots, \pi^{n*})$, von der kein Agent einseitig abweichen kann, ohne schlechter gestellt zu werden:

$$V^i(\pi^{i*}, \pi^{-i*}) \geq V^i(\pi^i, \pi^{-i*}) \quad \forall \pi^i, \forall i$$

**Herausforderung RL:** In MARL sind alle Agenten gleichzeitig lernend → die Umgebung ist **nicht-stationär** aus der Perspektive jedes einzelnen Agenten.

---

## M16 · Cooperative MARL *(4 Stunden)*

### Centralized Training with Decentralized Execution (CTDE)

Das CTDE-Paradigma ist das Herzstück kooperativer MARL-Algorithmen:

<!-- ILLUSTRATION: CTDE -->
<svg width="100%" viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="ctde1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Training Phase -->
  <rect x="20" y="10" width="300" height="220" rx="12" fill="#E1F5EE" stroke="#1D9E75" stroke-width="2"/>
  <text x="170" y="35" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#085041">Training (Centralized)</text>
  <!-- Centralized Critic -->
  <rect x="55" y="50" width="230" height="55" rx="8" fill="#1D9E75" stroke="#085041" stroke-width="1.5"/>
  <text x="170" y="74" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="white">Zentraler Critic</text>
  <text x="170" y="92" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#E1F5EE">bekommt globalen Zustand + alle Aktionen</text>
  <!-- Agent policies training -->
  <rect x="55" y="130" width="95" height="50" rx="8" fill="#9FE1CB" stroke="#1D9E75" stroke-width="1"/>
  <text x="102" y="153" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#085041">π¹_θ</text>
  <text x="102" y="170" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#085041">Agent 1</text>
  <rect x="185" y="130" width="95" height="50" rx="8" fill="#9FE1CB" stroke="#1D9E75" stroke-width="1"/>
  <text x="232" y="153" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#085041">π²_θ</text>
  <text x="232" y="170" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#085041">Agent 2</text>
  <line x1="170" y1="105" x2="102" y2="128" stroke="#085041" stroke-width="1.2" marker-end="url(#ctde1)"/>
  <line x1="170" y1="105" x2="232" y2="128" stroke="#085041" stroke-width="1.2" marker-end="url(#ctde1)"/>
  <text x="170" y="205" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">Trainiert mit globalem Wissen</text>

  <!-- Execution Phase -->
  <rect x="360" y="10" width="300" height="220" rx="12" fill="#E6F1FB" stroke="#185FA5" stroke-width="2"/>
  <text x="510" y="35" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#0C447C">Ausführung (Decentralized)</text>
  <!-- Agents only see local obs -->
  <rect x="375" y="55" width="120" height="80" rx="8" fill="#B5D4F4" stroke="#185FA5" stroke-width="1.5"/>
  <text x="435" y="82" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#0C447C">Agent 1</text>
  <text x="435" y="98" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">π¹_θ(a|o¹)</text>
  <text x="435" y="114" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">nur lok. Obs. o¹</text>
  <rect x="525" y="55" width="120" height="80" rx="8" fill="#B5D4F4" stroke="#185FA5" stroke-width="1.5"/>
  <text x="585" y="82" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#0C447C">Agent 2</text>
  <text x="585" y="98" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">π²_θ(a|o²)</text>
  <text x="585" y="114" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">nur lok. Obs. o²</text>
  <!-- No communication line during exec -->
  <line x1="495" y1="95" x2="525" y2="95" stroke="#A32D2D" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="510" y="88" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#A32D2D">kein</text>
  <text x="510" y="102" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#A32D2D">Comm.</text>
  <text x="510" y="185" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">Kein zentrales System nötig</text>
  <text x="510" y="200" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">→ dezentral einsetzbar</text>
</svg>

### QMIX

QMIX ermöglicht gemeinsames Lernen mit individuellen Q-Funktionen durch einen **Mixing Network** Ansatz:

$$Q_{tot}(\mathbf{a}, s) = f_s\left(Q^1(o^1, a^1), \ldots, Q^n(o^n, a^n)\right)$$

**Monotonie-Constraint** (IGM): $\frac{\partial Q_{tot}}{\partial Q^i} \geq 0$ — wichtig für dezentrale Ausführung.

### MADDPG

Multi-Agent Deep Deterministic Policy Gradient: Jeder Agent hat einen eigenen **zentralen Critic** der alle Aktionen sieht, und einen **dezentralen Actor**:

```python
# MADDPG Update für Agent i
# Critic Input: (s, a¹, a², ..., aⁿ)  – globaler Zustand + alle Aktionen
Q_i = critic_i(state, all_actions)

# Actor Input: nur lokale Observation oⁱ
a_i = actor_i(obs_i)

# Actor Loss (Gradient über alle anderen Agenten blockiert)
actor_loss = -Q_i(state, [a_1_detach, ..., a_i, ..., a_n_detach]).mean()
```

---

## M17 · Competitive MARL & Self-Play *(3 Stunden)*

### Self-Play

Im Zero-Sum-Setting ist der beste Trainingspartner... der Agent selbst:

<!-- ILLUSTRATION: Self-Play / League Training -->
<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="sp1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Self-Play loop -->
  <rect x="30" y="60" width="130" height="80" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="95" y="90" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#3C3489">Agent v1.0</text>
  <text x="95" y="110" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">aktuelle Policy</text>
  <text x="95" y="126" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">θ_current</text>

  <rect x="270" y="60" width="130" height="80" rx="10" fill="#FAEEDA" stroke="#BA7517" stroke-width="1.5"/>
  <text x="335" y="90" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#633806">Gegner Pool</text>
  <text x="335" y="110" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">θ_old, θ_older...</text>
  <text x="335" y="126" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">ältere Versionen</text>

  <rect x="510" y="60" width="130" height="80" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="2"/>
  <text x="575" y="90" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#085041">Agent v2.0</text>
  <text x="575" y="110" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">verbesserte Policy</text>
  <text x="575" y="126" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">stärker als v1.0</text>

  <!-- Arrows -->
  <line x1="160" y1="100" x2="268" y2="100" stroke="#888780" stroke-width="1.5" marker-end="url(#sp1)"/>
  <text x="214" y="92" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">spielt gegen</text>
  <line x1="400" y1="100" x2="508" y2="100" stroke="#1D9E75" stroke-width="1.5" marker-end="url(#sp1)"/>
  <text x="454" y="92" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">lernt → wird zu</text>
  <path d="M 575 140 Q 575 175 335 175 Q 160 175 95 142" fill="none" stroke="#D85A30" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#sp1)"/>
  <text x="335" y="192" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#D85A30">v2.0 wird zum neuen Gegner für v3.0 Training</text>
</svg>

### AlphaZero Schematik

AlphaZero kombiniert **MCTS (Monte Carlo Tree Search)** mit einem **Deep Neural Network** (Policy + Value Head):

```
1. MCTS-Suche:
   - Starte von aktuellem Zustand s
   - Simuliere N Züge tief (guided by neural network)
   - Berechne Besuchswahrscheinlichkeiten π(a|s)

2. Selbstspiel:
   - Agent A (θ) vs. Agent A (θ) → Trainingspartie
   - Speichere (s, π, z) mit z = Spielergebnis ∈ {-1, 0, +1}

3. Netzwerk-Training:
   - Loss = (z - v)² - π^T log p  (Value + Policy)
   - Update θ auf Trainings-Buffer

4. Evaluation:
   - Neue Version θ_new vs. alte Version θ_old
   - Ersetze wenn Winrate > 55%
```

---

## M18 · Emergentes Verhalten & Skalierung *(3 Stunden)*

### Emergenz in MARL

Emergenz bezeichnet das Auftreten komplexer Verhaltensweisen, die **nicht explizit programmiert** wurden:

<!-- ILLUSTRATION: Emergenz-Beispiele -->
<svg width="100%" viewBox="0 0 680 180" xmlns="http://www.w3.org/2000/svg">
  <!-- Drei Beispielboxen -->
  <rect x="20" y="30" width="190" height="130" rx="10" fill="#FAECE7" stroke="#D85A30" stroke-width="1.5"/>
  <text x="115" y="55" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#712B13">Kommunikation</text>
  <!-- Agents with comm arrows -->
  <circle cx="75" cy="110" r="18" fill="#D85A30"/>
  <text x="75" y="115" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="700" fill="white">A1</text>
  <circle cx="155" cy="110" r="18" fill="#D85A30"/>
  <text x="155" y="115" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="700" fill="white">A2</text>
  <path d="M92 105 Q115 85 138 105" fill="none" stroke="#712B13" stroke-width="1.5"/>
  <text x="115" y="83" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#712B13">emergentes Signal</text>
  <text x="115" y="155" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Agenten erfinden eigene</text>
  <text x="115" y="168" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Kommunikationsprotokoll</text>

  <rect x="245" y="30" width="190" height="130" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="340" y="55" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#3C3489">Arbeitsteilung</text>
  <!-- Different roles -->
  <circle cx="295" cy="105" r="18" fill="#7F77DD"/>
  <text x="295" y="100" text-anchor="middle" font-family="sans-serif" font-size="8" font-weight="700" fill="white">Scout</text>
  <text x="295" y="112" text-anchor="middle" font-family="sans-serif" font-size="8" fill="white">A1</text>
  <circle cx="345" cy="105" r="18" fill="#534AB7"/>
  <text x="345" y="100" text-anchor="middle" font-family="sans-serif" font-size="8" font-weight="700" fill="white">Guard</text>
  <text x="345" y="112" text-anchor="middle" font-family="sans-serif" font-size="8" fill="white">A2</text>
  <circle cx="395" cy="105" r="18" fill="#3C3489"/>
  <text x="395" y="100" text-anchor="middle" font-family="sans-serif" font-size="8" font-weight="700" fill="white">Attacker</text>
  <text x="395" y="112" text-anchor="middle" font-family="sans-serif" font-size="8" fill="white">A3</text>
  <text x="340" y="155" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Spontane Rollenteilung</text>
  <text x="340" y="168" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">ohne explizite Vorgabe</text>

  <rect x="470" y="30" width="190" height="130" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="565" y="55" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#085041">Taktiken</text>
  <!-- Tactics illustration -->
  <circle cx="510" cy="90" r="14" fill="#1D9E75"/>
  <text x="510" y="95" text-anchor="middle" font-family="sans-serif" font-size="9" font-weight="700" fill="white">A1</text>
  <circle cx="560" cy="120" r="14" fill="#1D9E75"/>
  <text x="560" y="125" text-anchor="middle" font-family="sans-serif" font-size="9" font-weight="700" fill="white">A2</text>
  <circle cx="620" cy="85" r="14" fill="#9FE1CB" stroke="#1D9E75" stroke-width="1"/>
  <text x="620" y="90" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#085041">Ziel</text>
  <path d="M524 90 Q560 75 608 82" fill="none" stroke="#085041" stroke-width="1.2" stroke-dasharray="3,2"/>
  <path d="M565 120 Q590 120 608 90" fill="none" stroke="#085041" stroke-width="1.2"/>
  <text x="565" y="155" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Flanking, Decoys,</text>
  <text x="565" y="168" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">koordinierte Angriffe</text>
</svg>

### MAPPO & HAPPO

**MAPPO** (Multi-Agent PPO): Erweitert PPO direkt auf kooperative Multi-Agent-Settings mit zentralem Critic.

**HAPPO** (Heterogeneous-Agent PPO): Ermöglicht PPO-Updates für Agenten mit unterschiedlichen Aktionsräumen — wichtig für heterogene Teams.

---

## Übungsaufgaben Phase 4

1. **Nash-Gleichgewicht berechnen:** Berechne das Nash-Gleichgewicht für das Rock-Paper-Scissors-Spiel und das Prisoners-Dilemma von Hand.

2. **QMIX implementieren:** Implementiere QMIX für `StarCraft Multi-Agent Challenge (SMAC)` mit PettingZoo und vergleiche mit Independent Q-Learning (IQL).

3. **MADDPG in Particle Env:** Trainiere MADDPG für die `cooperative navigation`-Umgebung (MPE). Zeige, dass CTDE bessere Ergebnisse liefert als dezentrales Training.

4. **Self-Play Tic-Tac-Toe:** Implementiere Self-Play mit AlphaZero-Methodik für Tic-Tac-Toe. Plotte Elo-Rating über Trainingsiterationen.

5. **Emergenz beobachten:** Repliziere ein einfaches OpenAI Hide-and-Seek Experiment. Beobachte, ob Agenten emergentes Werkzeugverhalten entwickeln.

---

## Empfohlene Literatur (Phase 4)

- **Lowe et al. (2017):** Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)
- **Rashid et al. (2018):** QMIX: Monotonic Value Function Factorisation for MARL
- **Silver et al. (2017):** Mastering Chess and Shogi by Self-Play with a General RL Algorithm (AlphaZero)
- **OpenAI (2019):** Emergent Tool Use from Multi-Agent Interaction (Hide and Seek)
- **Yu et al. (2022):** The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)

---

*← [Phase 3: Deep RL](./phase3_deep_rl.md) · → [Phase 5: RL in der Praxis](./phase5_praxis.md)*
