# Dynamische Programmierung im Reinforcement Learning

> **Lernziel:** Verstehen, wie klassische DP-Algorithmen (Policy Evaluation, Policy Iteration, Value Iteration) ein vollständig bekanntes MDP lösen.

---

## 1. Grundlagen: Das Markov-Entscheidungsproblem (MDP)

Ein MDP ist definiert durch das Tupel $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

| Symbol | Bedeutung |
|---|---|
| $\mathcal{S}$ | Zustandsraum |
| $\mathcal{A}$ | Aktionsraum |
| $P(s'\|s,a)$ | Übergangswahrscheinlichkeit |
| $R(s,a,s')$ | Belohnungsfunktion |
| $\gamma \in [0,1)$ | Diskontierungsfaktor |

**Ziel:** Finde eine optimale Policy $\pi^*$, die den erwarteten kumulativen Reward maximiert:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

---

## 2. Das Beispiel: Gridworld 4×4

Wir betrachten eine klassische **4×4 Gridworld**:

```
 S  .  .  .
 .  X  .  .
 .  X  .  .
 .  .  .  G
```

- **S** = Startzustand (oben links, Index 0)
- **G** = Ziel / Terminalzustand (unten rechts, Index 15)
- **X** = Hindernisse (Zustände 5 und 10)
- **Aktionen:** ↑ ↓ ← → (bei Wand oder Hindernis: bleibt im selben Zustand)
- **Reward:** $R = -1$ pro Schritt (außer im Terminalzustand)
- **$\gamma = 1.0$** (undiskontiert für Einfachheit)

---

## 3. Bellman-Gleichungen

### Zustandswertfunktion $V^\pi(s)$

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)\Big[R(s,a,s') + \gamma \cdot V^\pi(s')\Big]$$

### Optimale Bellman-Gleichung

$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s'} P(s'|s,a)\Big[R(s,a,s') + \gamma \cdot V^*(s')\Big]$$

---

## 4. Algorithmus 1: Policy Evaluation (Iterative)

Policy Evaluation berechnet $V^\pi$ für eine **gegebene** Policy $\pi$ iterativ.

### Pseudocode

```
Initialisiere V(s) = 0 für alle s ∈ S
Wiederhole:
    Δ ← 0
    Für jedes s ∈ S:
        v ← V(s)
        V(s) ← Σ a π(a|s) · Σ s' P(s'|s,a) · [R(s,a,s') + γ · V(s')]
        Δ ← max(Δ, |v - V(s)|)
Bis Δ < θ  (Konvergenzkriterium)
```

### Python-Implementierung

```python
import numpy as np

def policy_evaluation(policy, env, gamma=1.0, theta=1e-6):
    """
    Iterative Policy Evaluation.
    
    Args:
        policy: np.array [num_states, num_actions] mit Wahrscheinlichkeiten
        env:    Umgebungsobjekt mit .P (Übergangsmodell), .nS, .nA
        gamma:  Diskontierungsfaktor
        theta:  Konvergenzschwelle
    
    Returns:
        V: np.array [num_states] mit Zustandswerten
    """
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, pi_a in enumerate(policy[s]):
                for prob, next_s, reward, done in env.P[s][a]:
                    v += pi_a * prob * (reward + gamma * V[next_s] * (not done))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        
        if delta < theta:
            break
    
    return V
```

### Ergebnis (Random Policy, π = 0.25 für jede Aktion)

Nach Konvergenz erhält man z.B. folgende Wertfunktion:

```
 0.0  -14  -20  -22
-14   ---  -20  -20
-20   ---  -14  -14
-22   -20  -14   0.0
```

> **Interpretation:** Der Wert gibt die erwartete Anzahl von Schritten bis zum Ziel an (negiert). Zustände nahe dem Ziel haben höhere (weniger negative) Werte.

---

## 5. Algorithmus 2: Policy Iteration

Policy Iteration wechselt zwischen **Evaluation** und **Verbesserung**, bis die Policy stabil ist.

### Pseudocode

```
Initialisiere π(s) = beliebige Aktion für alle s
Wiederhole:
    1. Policy Evaluation: V ← V^π
    2. Policy Improvement:
       policy_stable ← True
       Für jedes s ∈ S:
           old_action ← π(s)
           π(s) ← argmax_a Σ_s' P(s'|s,a) [R + γ·V(s')]
           Falls old_action ≠ π(s): policy_stable ← False
Bis policy_stable = True
```

### Python-Implementierung

```python
def policy_iteration(env, gamma=1.0, theta=1e-6):
    """
    Policy Iteration: Findet die optimale Policy π* und V*.
    """
    # Initialisierung: Gleichverteilte Policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        # Schritt 1: Policy Evaluation
        V = policy_evaluation(policy, env, gamma, theta)
        
        # Schritt 2: Policy Improvement
        policy_stable = True
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            
            # Berechne Q(s,a) für alle Aktionen
            Q_s = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_s, reward, done in env.P[s][a]:
                    Q_s[a] += prob * (reward + gamma * V[next_s] * (not done))
            
            # Gierige Policy-Verbesserung
            best_action = np.argmax(Q_s)
            policy[s] = np.eye(env.nA)[best_action]  # deterministische Policy
            
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable:
            return policy, V
```

### Konvergenzverhalten

```
Iteration 1: Policy = Random → V sehr negativ
Iteration 2: Policy verbessert → V steigt
Iteration 3: Policy stabil → Konvergenz ✓
```

---

## 6. Algorithmus 3: Value Iteration

Value Iteration kombiniert Evaluation und Improvement in einem einzigen Sweep – effizienter als Policy Iteration!

### Pseudocode

```
Initialisiere V(s) = 0 für alle s ∈ S
Wiederhole:
    Δ ← 0
    Für jedes s ∈ S:
        v ← V(s)
        V(s) ← max a Σ s' P(s'|s,a) [R(s,a,s') + γ · V(s')]
        Δ ← max(Δ, |v - V(s)|)
Bis Δ < θ

Extrahiere Policy:
    π(s) ← argmax a Σ s' P(s'|s,a) [R(s,a,s') + γ · V(s')]
```

### Python-Implementierung

```python
def value_iteration(env, gamma=1.0, theta=1e-6):
    """
    Value Iteration: Direkte Berechnung von V* ohne explizite Policy.
    """
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        for s in range(env.nS):
            # Bellman-Optimierungsoperator
            Q_s = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_s, reward, done in env.P[s][a]:
                    Q_s[a] += prob * (reward + gamma * V[next_s] * (not done))
            
            v_new = np.max(Q_s)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        
        if delta < theta:
            break
    
    # Optimale Policy extrahieren
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        Q_s = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_s, reward, done in env.P[s][a]:
                Q_s[a] += prob * (reward + gamma * V[next_s] * (not done))
        best_action = np.argmax(Q_s)
        policy[s, best_action] = 1.0
    
    return policy, V
```

### Optimale Wertfunktion (Ergebnis)

```
  0   -1   -2   -3
 -1  ---   -3   -2    ← Werte
 -2  ---   -2   -1
 -3   -2   -1    0
```

### Optimale Policy (Ergebnis)

```
  ·    →    →    ↓
  ↓   ---   ↓    ↓
  ↓   ---   ↓    ↓
  →    →    →    ·
```

---

## 7. Vergleich der Algorithmen

| Eigenschaft | Policy Evaluation | Policy Iteration | Value Iteration |
|---|:---:|:---:|:---:|
| Liefert $V^\pi$ | ✅ | ✅ | ✅ |
| Liefert $\pi^*$ | ❌ | ✅ | ✅ |
| Benötigt Policy | ✅ | ✅ | ❌ |
| Konvergenzgeschwindigkeit | langsam | mittel–schnell | schnell |
| Sweeps bis Konvergenz | viele | wenige | moderat |
| Empfohlen bei | Analyse | kleinem $\|\mathcal{S}\|$ | großem $\|\mathcal{S}\|$ |

---

## 8. Wichtige Konzepte zusammengefasst

### Bellman-Operator $\mathcal{T}$

Der **Bellman-Optimierungsoperator** ist eine Kontraktion mit Faktor $\gamma$:

$$(\mathcal{T}V)(s) = \max_{a} \sum_{s'} P(s'|s,a)\big[R(s,a,s') + \gamma V(s')\big]$$

Durch den **Banach'schen Fixpunktsatz** gilt: wiederholte Anwendung von $\mathcal{T}$ konvergiert immer zu $V^*$.

### Generalized Policy Iteration (GPI)

Alle drei Algorithmen sind Spezialfälle von **GPI**:

```
     Evaluation          Improvement
  V ──────────────► π ──────────────► V
  ▲                                    │
  └────────────────────────────────────┘
         (wiederholen bis Konvergenz)
```

---

## 9. Übungsaufgaben

1. **Grundverständnis:** Warum divergiert Value Iteration bei $\gamma = 1$ ohne Terminalzustand?

2. **Implementierung:** Erweitere die Gridworld um stochastische Übergänge (z.B. mit 20% Wahrscheinlichkeit rutscht der Agent seitwärts). Wie verändert sich $V^*$?

3. **Analyse:** Vergleiche die Anzahl der Bellman-Updates bis zur Konvergenz für Policy Iteration vs. Value Iteration auf einer 8×8 Gridworld.

4. **Theorie:** Zeige, dass der Bellman-Optimierungsoperator $\mathcal{T}$ eine $\gamma$-Kontraktion bezüglich der $\ell_\infty$-Norm ist.

5. **Erweiterung:** Implementiere **Asynchronous Value Iteration**, bei dem Zustände in zufälliger Reihenfolge oder priorisiert (Prioritized Sweeping) aktualisiert werden.

---

## 10. Weiterführende Literatur

- **Sutton & Barto (2018):** *Reinforcement Learning: An Introduction*, Kap. 4 — [frei verfügbar online](http://incompleteideas.net/book/the-book-2nd.html)
- **Puterman (1994):** *Markov Decision Processes* — Theoretische Grundlagen
- **Bertsekas (2012):** *Dynamic Programming and Optimal Control* — Fortgeschrittene Theorie
- **OpenAI Gym:** `FrozenLake-v1` als praxisnahe Gridworld-Implementierung zum Ausprobieren

---

*Erstellt für den Reinforcement-Learning Kurs · Dynamische Programmierung · Stand: 2026*
