# Phase 2 · Klassisches RL
**Umfang:** 20 Stunden · Module M5–M9  
**Voraussetzung:** Phase 1 abgeschlossen (MDP, Bellman, DP)

---

## Lernziele dieser Phase

Nach Abschluss von Phase 2 können die Teilnehmer:
- Monte Carlo-, TD- und Q-Learning-Algorithmen von Grund auf implementieren
- Den Unterschied zwischen On-Policy und Off-Policy klar erklären und begründen
- Eligibility Traces und TD(λ) als Brücke zwischen MC und TD verstehen
- Das Policy Gradient Theorem herleiten und REINFORCE implementieren
- Konvergenzeigenschaften und Instabilitäten (Deadly Triad) benennen

---

## M5 · Monte Carlo Methoden *(4 Stunden)*

### Grundidee

Monte Carlo (MC) Methoden lernen direkt aus **vollständigen Episoden** ohne Modellwissen. Sie schätzen $V(s)$ oder $Q(s,a)$ durch **empirische Durchschnitte** der beobachteten Returns.

<!-- ILLUSTRATION: MC vs TD Idee -->
<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrX" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Episode timeline MC -->
  <text x="40" y="44" font-family="sans-serif" font-size="12" font-weight="700" fill="#3C3489">Monte Carlo</text>
  <!-- States boxes -->
  <rect x="40" y="55" width="46" height="36" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="1"/>
  <text x="63" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#3C3489">s₀</text>
  <line x1="86" y1="73" x2="106" y2="73" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrX)"/>
  <rect x="108" y="55" width="46" height="36" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="1"/>
  <text x="131" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#3C3489">s₁</text>
  <line x1="154" y1="73" x2="174" y2="73" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrX)"/>
  <rect x="176" y="55" width="46" height="36" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="1"/>
  <text x="199" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#3C3489">s₂</text>
  <line x1="222" y1="73" x2="242" y2="73" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrX)"/>
  <text x="260" y="78" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#888780">…</text>
  <line x1="274" y1="73" x2="294" y2="73" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrX)"/>
  <rect x="296" y="55" width="60" height="36" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="326" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#085041">sₜ (End)</text>
  <!-- Big arrow back for MC -->
  <path d="M 326 91 Q 183 140 63 91" fill="none" stroke="#D85A30" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrX)"/>
  <text x="183" y="158" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#D85A30">Gₜ = Σγᵏrₜ₊ₖ₊₁  (erst nach Ende der Episode verfügbar)</text>
  <!-- TD timeline -->
  <text x="380" y="44" font-family="sans-serif" font-size="12" font-weight="700" fill="#0C447C">Temporal Difference</text>
  <rect x="380" y="55" width="46" height="36" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>
  <text x="403" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">sₜ</text>
  <line x1="426" y1="73" x2="458" y2="73" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrX)"/>
  <rect x="460" y="55" width="46" height="36" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>
  <text x="483" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">sₜ₊₁</text>
  <text x="600" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#888780">…  (Ende egal)</text>
  <!-- Short arrow back for TD -->
  <path d="M 483 91 Q 443 125 403 91" fill="none" stroke="#1D9E75" stroke-width="2" marker-end="url(#arrX)"/>
  <text x="443" y="142" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#1D9E75">δ = r + γV(sₜ₊₁) − V(sₜ)  (sofort nach 1 Schritt)</text>
</svg>

### First-Visit vs. Every-Visit MC

- **First-Visit MC:** Nur der erste Besuch eines Zustands $s$ pro Episode zählt für die Schätzung
- **Every-Visit MC:** Jeder Besuch zählt

```python
def mc_first_visit(env, pi, gamma, n_episodes):
    V = defaultdict(float)
    returns = defaultdict(list)
    
    for _ in range(n_episodes):
        episode = generate_episode(env, pi)  # [(s0,a0,r1), (s1,a1,r2), ...]
        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if s not in visited:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)
    return V
```

### Off-Policy MC mit Importance Sampling

Wenn die **Behavior Policy** $b$ sich von der **Target Policy** $\pi$ unterscheidet:

$$V^\pi(s) \approx \frac{\sum_t \rho_t G_t}{\sum_t \rho_t}, \quad \rho_t = \prod_{k=t}^{T-1} \frac{\pi(a_k|s_k)}{b(a_k|s_k)}$$

---

## M6 · Temporal Difference Learning *(4 Stunden)*

### TD(0) – Der einfachste TD-Algorithmus

TD(0) aktualisiert $V(s)$ nach **jedem Schritt** (nicht erst nach Ende der Episode):

$$V(s_t) \leftarrow V(s_t) + \alpha \underbrace{\left[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\right]}_{\text{TD-Fehler } \delta_t}$$

<!-- ILLUSTRATION: TD-Fehler Visualisierung -->
<svg width="100%" viewBox="0 0 680 160" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr6" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- States -->
  <rect x="50" y="55" width="130" height="50" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="115" y="77" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#0C447C">Zustand sₜ</text>
  <text x="115" y="95" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#185FA5">V(sₜ) = 3.2</text>
  <line x1="180" y1="80" x2="270" y2="80" stroke="#888780" stroke-width="1.5" marker-end="url(#arr6)"/>
  <text x="225" y="70" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#D85A30">r=+5</text>
  <rect x="272" y="55" width="140" height="50" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="342" y="77" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#085041">Zustand sₜ₊₁</text>
  <text x="342" y="95" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1D9E75">V(sₜ₊₁) = 6.0</text>
  <!-- TD error calculation -->
  <rect x="460" y="30" width="200" height="100" rx="10" fill="#FAEEDA" stroke="#BA7517" stroke-width="1.5"/>
  <text x="560" y="55" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#633806">TD-Fehler δₜ</text>
  <text x="560" y="75" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#854F0B">= r + γV(sₜ₊₁) − V(sₜ)</text>
  <text x="560" y="93" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#854F0B">= 5 + 0.9×6.0 − 3.2</text>
  <text x="560" y="111" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#412402">= +7.2  (zu niedrig!)</text>
  <line x1="412" y1="80" x2="458" y2="80" stroke="#BA7517" stroke-width="1.5" marker-end="url(#arr6)"/>
  <!-- Update label -->
  <text x="340" y="140" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#534AB7">Update: V(sₜ) ← 3.2 + α × 7.2  (mit Lernrate α = 0.1 → V(sₜ) = 3.92)</text>
</svg>

### TD(λ) und Eligibility Traces

TD(λ) verbindet MC (λ=1) und TD(0) (λ=0) durch eine gewichtete Kombination aller n-Step Returns:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

**Eligibility Trace** $e_t(s)$ merkt sich, welche Zustände kürzlich besucht wurden und für den aktuellen TD-Fehler verantwortlich sind:

$$e_t(s) = \gamma\lambda \cdot e_{t-1}(s) + \mathbf{1}[s_t = s]$$
$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s) \quad \forall s$$

---

## M7 · SARSA & Q-Learning *(4 Stunden)*

### On-Policy: SARSA

SARSA (State-Action-Reward-State-Action) aktualisiert $Q$ mit der **tatsächlich gewählten** nächsten Aktion:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]$$

### Off-Policy: Q-Learning

Q-Learning aktualisiert mit dem **besten möglichen** nächsten Schritt (greedy), unabhängig von der tatsächlichen Aktion:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

<!-- ILLUSTRATION: SARSA vs Q-Learning Backup -->
<svg width="100%" viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr7" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- SARSA column -->
  <text x="160" y="28" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#1D9E75">SARSA (On-Policy)</text>
  <circle cx="160" cy="70" r="24" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="160" y="66" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#085041">sₜ</text>
  <text x="160" y="83" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1D9E75">aₜ gewählt</text>
  <line x1="160" y1="94" x2="160" y2="130" stroke="#888780" stroke-width="1" marker-end="url(#arr7)"/>
  <circle cx="160" cy="152" r="24" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="160" y="148" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#085041">sₜ₊₁</text>
  <text x="160" y="165" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1D9E75">aₜ₊₁ gewählt</text>
  <!-- Action arrows from s_{t+1} SARSA -->
  <line x1="145" y1="174" x2="100" y2="220" stroke="#1D9E75" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arr7)"/>
  <line x1="160" y1="176" x2="160" y2="222" stroke="#1D9E75" stroke-width="2" marker-end="url(#arr7)"/>
  <line x1="175" y1="174" x2="220" y2="220" stroke="#1D9E75" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arr7)"/>
  <circle cx="100" cy="230" r="12" fill="white" stroke="#1D9E75" stroke-width="1"/>
  <circle cx="160" cy="232" r="12" fill="#1D9E75" stroke="#085041" stroke-width="1.5"/>
  <text x="160" y="236" text-anchor="middle" font-family="sans-serif" font-size="9" font-weight="700" fill="white">π</text>
  <circle cx="220" cy="230" r="12" fill="white" stroke="#1D9E75" stroke-width="1"/>
  <text x="160" y="255" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1D9E75">aₜ₊₁ = π(sₜ₊₁)</text>

  <!-- Q-Learning column -->
  <text x="510" y="28" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#534AB7">Q-Learning (Off-Policy)</text>
  <circle cx="510" cy="70" r="24" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="510" y="66" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#3C3489">sₜ</text>
  <text x="510" y="83" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#534AB7">aₜ gewählt</text>
  <line x1="510" y1="94" x2="510" y2="130" stroke="#888780" stroke-width="1" marker-end="url(#arr7)"/>
  <circle cx="510" cy="152" r="24" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="510" y="148" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#3C3489">sₜ₊₁</text>
  <text x="510" y="165" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#534AB7">alle Aktionen</text>
  <!-- Action arrows from s_{t+1} Q-Learning -->
  <line x1="492" y1="174" x2="445" y2="220" stroke="#534AB7" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arr7)"/>
  <line x1="507" y1="176" x2="507" y2="222" stroke="#534AB7" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arr7)"/>
  <line x1="527" y1="174" x2="574" y2="220" stroke="#534AB7" stroke-width="2" marker-end="url(#arr7)"/>
  <circle cx="445" cy="230" r="12" fill="white" stroke="#534AB7" stroke-width="1"/>
  <circle cx="507" cy="232" r="12" fill="white" stroke="#534AB7" stroke-width="1"/>
  <circle cx="574" cy="230" r="12" fill="#534AB7" stroke="#3C3489" stroke-width="1.5"/>
  <text x="574" y="234" text-anchor="middle" font-family="sans-serif" font-size="9" font-weight="700" fill="white">max</text>
  <text x="510" y="255" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#534AB7">max_a' Q(sₜ₊₁, a') – greedy</text>
</svg>

### Deadly Triad

Instabilität entsteht, wenn **alle drei** Komponenten gleichzeitig auftreten:

| Komponente | Bedeutung |
|---|---|
| **Funktionsapproximation** | z.B. neuronales Netz für Q |
| **Bootstrapping** | TD-Update (Q-Wert hängt von Q-Wert ab) |
| **Off-Policy** | Target Policy ≠ Behavior Policy |

> Lösung: DQN (Phase 3) umgeht die Instabilität durch **Experience Replay** und **Target Network**.

---

## M8 · Funktionsapproximation *(4 Stunden)*

### Motivation

Bei großen oder kontinuierlichen Zustandsräumen ist eine Tabelle nicht mehr praktikabel. Wir approximieren:

$$\hat{V}(s, \mathbf{w}) \approx V^\pi(s) \qquad \hat{Q}(s,a, \mathbf{w}) \approx Q^\pi(s,a)$$

### Semi-Gradient TD(0) mit linearer FA

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \underbrace{\left[r + \gamma \hat{V}(s', \mathbf{w}) - \hat{V}(s, \mathbf{w})\right]}_{\delta} \nabla_\mathbf{w} \hat{V}(s, \mathbf{w})$$

**Semi-Gradient:** Der Gradient wird nur auf $\hat{V}(s, \mathbf{w})$ angewendet, nicht auf $\hat{V}(s', \mathbf{w})$ (Target wird als konstant behandelt).

<!-- ILLUSTRATION: Feature-Vektor & Approximation -->
<svg width="100%" viewBox="0 0 680 180" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr8" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- State input -->
  <rect x="30" y="65" width="90" height="50" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="75" y="88" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#0C447C">Zustand s</text>
  <text x="75" y="106" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">(x, y, θ, …)</text>
  <line x1="120" y1="90" x2="158" y2="90" stroke="#888780" stroke-width="1.5" marker-end="url(#arr8)"/>
  <!-- Feature vector -->
  <rect x="160" y="40" width="80" height="100" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="1.5"/>
  <text x="200" y="60" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#633806">Feature</text>
  <text x="200" y="74" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#633806">Vektor</text>
  <text x="200" y="95" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#854F0B">φ(s) ∈ ℝⁿ</text>
  <text x="200" y="113" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#854F0B">Tile Coding /</text>
  <text x="200" y="126" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#854F0B">RBF / Poly</text>
  <line x1="240" y1="90" x2="278" y2="90" stroke="#888780" stroke-width="1.5" marker-end="url(#arr8)"/>
  <!-- Linear combination -->
  <rect x="280" y="40" width="120" height="100" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="340" y="63" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#3C3489">Lineare FA</text>
  <text x="340" y="83" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#534AB7">V̂(s,w)</text>
  <text x="340" y="100" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#534AB7">= wᵀ φ(s)</text>
  <text x="340" y="122" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">w ∈ ℝⁿ lernbar</text>
  <line x1="400" y1="90" x2="438" y2="90" stroke="#888780" stroke-width="1.5" marker-end="url(#arr8)"/>
  <!-- Output -->
  <rect x="440" y="65" width="90" height="50" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="485" y="88" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#085041">V̂(s) ≈ V*(s)</text>
  <text x="485" y="106" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">Skalar</text>
  <!-- Update arrow -->
  <path d="M 485 115 Q 485 155 340 155 Q 200 155 200 142" fill="none" stroke="#D85A30" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr8)"/>
  <text x="340" y="172" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#D85A30">Semi-Gradient-Update: w ← w + α·δ·φ(s)</text>
</svg>

---

## M9 · Policy Gradient Grundlagen *(4 Stunden)*

### Idee: Direkt die Policy optimieren

Statt Q/V zu lernen, parametrisieren wir die Policy direkt:

$$\pi_\theta(a|s) = \text{softmax}\left(\mathbf{w}^\top \phi(s,a)\right)$$

**Ziel:** Maximiere $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[G_0\right]$

### Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[Q^{\pi_\theta}(s,a) \cdot \nabla_\theta \log \pi_\theta(a|s)\right]$$

### REINFORCE Algorithmus

```python
def reinforce(env, pi_theta, optimizer, gamma, n_episodes):
    for episode in range(n_episodes):
        # 1. Episode samplen
        trajectory = []
        state = env.reset()
        done = False
        while not done:
            action, log_prob = pi_theta.act(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((log_prob, reward))
            state = next_state
        
        # 2. Returns berechnen
        G, returns = 0, []
        for _, r in reversed(trajectory):
            G = gamma * G + r
            returns.insert(0, G)
        
        # 3. Policy Gradient Update
        loss = 0
        for (log_prob, _), G_t in zip(trajectory, returns):
            loss -= log_prob * G_t  # Minus wegen Maximierung
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Baseline-Subtraktion zur Varianzreduktion

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\left(G_t - b(s_t)\right) \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

<!-- ILLUSTRATION: Variance Reduction mit Baseline -->
<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Ohne Baseline: hohe Varianz -->
  <text x="170" y="25" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#D85A30">Ohne Baseline – hohe Varianz</text>
  <line x1="40" y1="150" x2="310" y2="150" stroke="#888780" stroke-width="1"/>
  <line x1="40" y1="40" x2="40" y2="155" stroke="#888780" stroke-width="1"/>
  <polyline points="40,145 70,50 100,140 130,60 160,130 190,45 220,135 250,55 280,140 310,70"
    fill="none" stroke="#D85A30" stroke-width="2"/>
  <text x="175" y="170" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Episode</text>
  <text x="22" y="100" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780" transform="rotate(-90,22,100)">∇J</text>

  <!-- Mit Baseline: niedrige Varianz -->
  <text x="510" y="25" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#1D9E75">Mit Baseline V(s) – niedrige Varianz</text>
  <line x1="380" y1="150" x2="650" y2="150" stroke="#888780" stroke-width="1"/>
  <line x1="380" y1="40" x2="380" y2="155" stroke="#888780" stroke-width="1"/>
  <polyline points="380,100 410,92 440,108 470,95 500,105 530,93 560,107 590,96 620,104 650,97"
    fill="none" stroke="#1D9E75" stroke-width="2"/>
  <!-- Baseline reference line -->
  <line x1="380" y1="100" x2="650" y2="100" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="655" y="104" font-family="sans-serif" font-size="10" fill="#534AB7">b(s)</text>
  <text x="515" y="170" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Episode</text>
</svg>

Die Baseline $b(s_t) = V(s_t)$ (erlernte Value-Funktion) reduziert die Varianz drastisch — das Advantage $A_t = G_t - V(s_t)$ gibt an, ob eine Aktion **besser oder schlechter** als erwartet war.

### Actor-Critic

Kombination aus Policy (Actor) und Value-Funktion (Critic):

```
Actor:   π_θ(a|s)  → wählt Aktion
Critic:  V_w(s)    → bewertet Zustand → liefert Baseline
         δ = r + γV_w(s') - V_w(s)  → TD-Fehler als Advantage-Schätzer
```

---

## Übungsaufgaben Phase 2

1. **MC vs. TD:** Implementiere First-Visit MC und TD(0) für `CartPole-v1`. Vergleiche Konvergenzgeschwindigkeit und Stabilität.

2. **SARSA vs. Q-Learning:** Implementiere beide Algorithmen für `CliffWalking-v0`. Erkläre, warum Q-Learning die kürzere aber riskantere Route lernt, SARSA die sichere Route.

3. **Eligibility Traces:** Implementiere TD(λ) für verschiedene λ-Werte (0, 0.5, 0.9, 1.0) und plotte die Lernkurve.

4. **Tile Coding:** Implementiere Tile Coding für den kontinuierlichen Zustandsraum von `MountainCar-v0` und trainiere SARSA damit.

5. **REINFORCE mit Baseline:** Implementiere REINFORCE für `CartPole-v1` (1) ohne Baseline und (2) mit $b = V(s)$ als separates Netz. Vergleiche die Varianz des Gradienten.

---

## Empfohlene Literatur (Phase 2)

- **Sutton & Barto, Kap. 5–7, 9, 13** – MC, TD, FA, Policy Gradient
- **David Silver, Lecture 4–7** – MC, TD, FA, Policy Gradient
- **Spinning Up: Key Papers** – REINFORCE (Williams 1992), Q-Learning (Watkins 1992)
- **CleanRL:** [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) – Referenzimplementierungen

---

*← [Phase 1: Grundlagen & MDP](./phase1_grundlagen_mdp.md) · → [Phase 3: Deep RL](./phase3_deep_rl.md)*
