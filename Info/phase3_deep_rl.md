# Phase 3 · Deep Reinforcement Learning
**Umfang:** 18 Stunden · Module M10–M14  
**Voraussetzung:** Phase 1–2 abgeschlossen; PyTorch-Grundkenntnisse

---

## Lernziele dieser Phase

Nach Abschluss von Phase 3 können die Teilnehmer:
- DQN und alle wichtigen Erweiterungen (Double, Dueling, PER, Rainbow) implementieren
- A2C, A3C und PPO von Grund auf in PyTorch umsetzen
- Den Trust-Region-Ansatz (TRPO) und Clipping-Mechanismus (PPO) mathematisch begründen
- Model-Based RL von Model-Free abgrenzen und Dyna/MuZero konzeptuell beschreiben
- Explorations-Strategien (ε-Greedy bis Curiosity) vergleichen und situationsgerecht wählen

---

## M10 · DQN & Varianten *(4 Stunden)*

### Von Q-Learning zu DQN

Q-Learning funktioniert mit Tabellen. Mit neuronalen Netzen entstehen drei kritische Probleme:

| Problem | Ursache | Lösung in DQN |
|---|---|---|
| **Korrelierte Updates** | Konsekutive Zustände sind ähnlich | Experience Replay Buffer |
| **Instabiles Target** | Q-Target hängt von sich selbst ab | Target Network (periodisch kopiert) |
| **Divergenz** | Deadly Triad mit Off-Policy | Beide oben + Gradient Clipping |

<!-- ILLUSTRATION: DQN Architektur -->
<svg width="100%" viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="dqn1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Environment -->
  <rect x="20" y="110" width="90" height="60" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="65" y="137" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#085041">Env</text>
  <text x="65" y="155" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">s, r</text>

  <!-- Replay Buffer -->
  <rect x="160" y="30" width="120" height="220" rx="10" fill="#FAEEDA" stroke="#BA7517" stroke-width="1.5"/>
  <text x="220" y="55" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#633806">Replay</text>
  <text x="220" y="71" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#633806">Buffer</text>
  <text x="220" y="92" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#854F0B">N = 100k–1M</text>
  <!-- Buffer entries -->
  <rect x="170" y="105" width="100" height="18" rx="3" fill="#EF9F27" opacity="0.5"/>
  <text x="220" y="118" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#412402">(s, a, r, s', done)</text>
  <rect x="170" y="127" width="100" height="18" rx="3" fill="#EF9F27" opacity="0.4"/>
  <text x="220" y="140" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#412402">(s, a, r, s', done)</text>
  <rect x="170" y="149" width="100" height="18" rx="3" fill="#EF9F27" opacity="0.3"/>
  <rect x="170" y="171" width="100" height="18" rx="3" fill="#EF9F27" opacity="0.2"/>
  <text x="220" y="200" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">Random</text>
  <text x="220" y="214" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">Mini-Batch</text>
  <text x="220" y="232" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">Sample ↓</text>

  <!-- Online Q-Network -->
  <rect x="340" y="60" width="130" height="80" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="2"/>
  <text x="405" y="88" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#3C3489">Q-Netz θ</text>
  <text x="405" y="106" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">(Online Network)</text>
  <text x="405" y="124" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">Gradient descent ↓</text>

  <!-- Target Q-Network -->
  <rect x="340" y="170" width="130" height="80" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="405" y="198" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#0C447C">Q-Netz θ⁻</text>
  <text x="405" y="216" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">(Target Network)</text>
  <text x="405" y="234" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">Sync alle C Schritte</text>

  <!-- Loss output -->
  <rect x="530" y="90" width="120" height="60" rx="10" fill="#FCEBEB" stroke="#A32D2D" stroke-width="1.5"/>
  <text x="590" y="116" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#501313">MSE Loss</text>
  <text x="590" y="134" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#A32D2D">(y − Q(s,a;θ))²</text>

  <!-- Arrows -->
  <line x1="110" y1="140" x2="158" y2="140" stroke="#1D9E75" stroke-width="1.5" marker-end="url(#dqn1)"/>
  <line x1="280" y1="140" x2="338" y2="100" stroke="#BA7517" stroke-width="1.5" marker-end="url(#dqn1)"/>
  <line x1="280" y1="140" x2="338" y2="210" stroke="#BA7517" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#dqn1)"/>
  <line x1="470" y1="100" x2="528" y2="110" stroke="#534AB7" stroke-width="1.5" marker-end="url(#dqn1)"/>
  <line x1="470" y1="210" x2="528" y2="130" stroke="#185FA5" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#dqn1)"/>
  <!-- Copy arrow: Online -> Target -->
  <path d="M 405 140 Q 415 155 405 170" fill="none" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="4,2" marker-end="url(#dqn1)"/>
  <text x="425" y="155" font-family="sans-serif" font-size="9" fill="#534AB7">θ⁻ ← θ</text>

  <!-- Action selection arrow -->
  <path d="M 405 60 Q 405 20 65 20 Q 65 55 65 108" fill="none" stroke="#534AB7" stroke-width="1.5" marker-end="url(#dqn1)"/>
  <text x="220" y="15" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">ε-greedy Aktion aus Q(s,a;θ)</text>
</svg>

### DQN-Varianten im Überblick

<!-- ILLUSTRATION: DQN Varianten Stufendiagramm -->
<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Treppendiagramm der Verbesserungen -->
  <rect x="30" y="140" width="80" height="40" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>
  <text x="70" y="158" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#0C447C">DQN</text>
  <text x="70" y="172" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">Basis</text>

  <rect x="130" y="115" width="80" height="65" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1"/>
  <text x="170" y="133" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#085041">Double</text>
  <text x="170" y="147" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#085041">DQN</text>
  <text x="170" y="163" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1D9E75">–Q-Überschätzung</text>

  <rect x="230" y="90" width="80" height="90" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="1"/>
  <text x="270" y="108" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#3C3489">Dueling</text>
  <text x="270" y="122" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#3C3489">DQN</text>
  <text x="270" y="140" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#534AB7">V(s)+A(s,a)</text>
  <text x="270" y="154" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#534AB7">Architektur</text>

  <rect x="330" y="65" width="80" height="115" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="1"/>
  <text x="370" y="83" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#633806">PER</text>
  <text x="370" y="99" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#854F0B">Prioritized</text>
  <text x="370" y="113" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#854F0B">Experience</text>
  <text x="370" y="127" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#854F0B">Replay</text>

  <rect x="430" y="40" width="80" height="140" rx="6" fill="#FAECE7" stroke="#D85A30" stroke-width="1"/>
  <text x="470" y="58" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#712B13">Noisy Net</text>
  <text x="470" y="73" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#D85A30">Stochastische</text>
  <text x="470" y="87" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#D85A30">Gewichte</text>

  <rect x="530" y="20" width="80" height="160" rx="6" fill="#E1F5EE" stroke="#085041" stroke-width="2"/>
  <text x="570" y="40" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#04342C">Rainbow</text>
  <text x="570" y="56" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#085041">Alle 6</text>
  <text x="570" y="70" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#085041">Komponenten</text>
  <text x="570" y="84" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#085041">kombiniert</text>
  <text x="570" y="100" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#085041">SOTA Atari</text>

  <text x="340" y="192" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Jede Erweiterung verbessert Stabilität oder Sample-Effizienz</text>
</svg>

---

## M11 · Policy Gradient (Deep): A2C & A3C *(4 Stunden)*

### Advantage Actor-Critic (A2C)

$$\mathcal{L}(\theta) = -\mathbb{E}\left[A_t \cdot \log \pi_\theta(a_t|s_t)\right] + c_1 \mathcal{L}_V - c_2 H[\pi_\theta]$$

- $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — Advantage (TD-Fehler)
- $\mathcal{L}_V$ — Value-Funktion Loss
- $H[\pi_\theta]$ — Entropie-Bonus (verhindert vorzeitiges Kollabieren der Policy)

### Asynchronous A3C

Mehrere Worker-Threads laufen **parallel** mit eigenen Umgebungskopien und updaten asynchron den globalen Parameter-Server:

<!-- ILLUSTRATION: A3C Architektur -->
<svg width="100%" viewBox="0 0 680 240" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="a3c1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Global Network -->
  <rect x="260" y="80" width="160" height="80" rx="12" fill="#EEEDFE" stroke="#534AB7" stroke-width="2"/>
  <text x="340" y="112" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#3C3489">Global Network</text>
  <text x="340" y="130" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#534AB7">θ (Actor + Critic)</text>
  <text x="340" y="148" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Shared Parameters</text>

  <!-- Workers -->
  <rect x="30" y="30" width="110" height="60" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="85" y="58" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#0C447C">Worker 1</text>
  <text x="85" y="74" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">θ₁ (lokale Kopie)</text>

  <rect x="30" y="110" width="110" height="60" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="85" y="138" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#0C447C">Worker 2</text>
  <text x="85" y="154" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#185FA5">θ₂ (lokale Kopie)</text>

  <rect x="30" y="190" width="110" height="40" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="85" y="214" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#888780">Worker n …</text>

  <!-- Arrows: copy global -->
  <path d="M 260 110 L 142 65" fill="none" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="4,2" marker-end="url(#a3c1)"/>
  <path d="M 260 130 L 142 140" fill="none" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="4,2" marker-end="url(#a3c1)"/>
  <text x="188" y="88" font-family="sans-serif" font-size="9" fill="#534AB7">θ kopieren</text>

  <!-- Arrows: push gradients -->
  <path d="M 142 55 L 258 105" fill="none" stroke="#D85A30" stroke-width="1.5" marker-end="url(#a3c1)"/>
  <path d="M 142 150 L 258 135" fill="none" stroke="#D85A30" stroke-width="1.5" marker-end="url(#a3c1)"/>
  <text x="188" y="132" font-family="sans-serif" font-size="9" fill="#D85A30">∇θ senden</text>

  <!-- Envs on right -->
  <rect x="540" y="30" width="110" height="55" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="595" y="55" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#085041">Env-Kopie 1</text>
  <text x="595" y="70" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1D9E75">unabhängige Simulation</text>
  <rect x="540" y="105" width="110" height="55" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="595" y="130" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#085041">Env-Kopie 2</text>
  <text x="595" y="145" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1D9E75">unabhängige Simulation</text>

  <line x1="140" y1="58" x2="538" y2="58" stroke="#1D9E75" stroke-width="1" marker-end="url(#a3c1)"/>
  <line x1="140" y1="138" x2="538" y2="132" stroke="#1D9E75" stroke-width="1" marker-end="url(#a3c1)"/>
  <text x="340" y="220" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Asynchrone Updates vermeiden korrelierten Replay Buffer (ersetzen ihn)</text>
</svg>

---

## M12 · PPO & TRPO *(4 Stunden)*

### Trust Region Policy Optimization (TRPO)

**Problem:** Zu große Policy-Updates können die Performance irreversibel verschlechtern.

**Lösung:** Update nur innerhalb einer **Trust Region** (begrenzte KL-Divergenz):

$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} A_t\right] \quad \text{s.t. } D_{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \delta$$

### Proximal Policy Optimization (PPO)

PPO vereinfacht TRPO durch **Clipping** des Importance Weights:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t\right)\right]$$

<!-- ILLUSTRATION: PPO Clipping -->
<svg width="100%" viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="ppo1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="60" y1="180" x2="620" y2="180" stroke="#888780" stroke-width="1" marker-end="url(#ppo1)"/>
  <line x1="60" y1="20" x2="60" y2="185" stroke="#888780" stroke-width="1" marker-end="url(#ppo1)"/>
  <text x="630" y="184" font-family="sans-serif" font-size="11" fill="#888780">r(θ)</text>
  <text x="38" y="24" font-family="sans-serif" font-size="11" fill="#888780">L</text>

  <!-- A > 0 case -->
  <!-- Unclipped L = r * A (positive slope line) -->
  <line x1="60" y1="175" x2="340" y2="90" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="5,3"/>
  <!-- Clipped region left (r < 1-eps) -->
  <line x1="60" y1="145" x2="220" y2="145" stroke="#D85A30" stroke-width="3"/>
  <!-- Clipped region right (r > 1+eps) -->
  <line x1="340" y1="90" x2="580" y2="90" stroke="#D85A30" stroke-width="3"/>
  <!-- Middle section (1-eps to 1+eps) -->
  <line x1="220" y1="145" x2="340" y2="90" stroke="#1D9E75" stroke-width="3"/>
  <!-- Boundary lines -->
  <line x1="220" y1="20" x2="220" y2="185" stroke="#BA7517" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="340" y1="20" x2="340" y2="185" stroke="#BA7517" stroke-width="1" stroke-dasharray="3,3"/>
  <!-- Labels -->
  <text x="220" y="196" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#BA7517">1-ε</text>
  <text x="340" y="196" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#BA7517">1+ε</text>
  <text x="280" y="196" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#888780">r=1</text>
  <!-- r = 1 line -->
  <line x1="280" y1="175" x2="280" y2="185" stroke="#888780" stroke-width="1"/>
  <text x="70" y="135" font-family="sans-serif" font-size="10" fill="#D85A30">Clipped</text>
  <text x="245" y="110" font-family="sans-serif" font-size="10" fill="#1D9E75">Linear</text>
  <text x="430" y="82" font-family="sans-serif" font-size="10" fill="#D85A30">Clipped</text>
  <text x="450" y="40" font-family="sans-serif" font-size="11" font-weight="600" fill="#534AB7">A > 0 (gute Aktion)</text>
  <text x="450" y="58" font-family="sans-serif" font-size="10" fill="#888780">Kein Bonus für r &gt; 1+ε</text>
  <text x="450" y="74" font-family="sans-serif" font-size="10" fill="#888780">→ Verhindert overcommit</text>
</svg>

### PPO in der Praxis: Hyperparameter

```python
# Standard PPO Hyperparameter (Schulman et al. 2017)
PPO_CONFIG = {
    "learning_rate":    3e-4,
    "n_steps":          2048,    # Steps pro Update-Runde
    "batch_size":       64,
    "n_epochs":         10,      # Passes über den Mini-Batch
    "gamma":            0.99,
    "gae_lambda":       0.95,    # GAE λ
    "clip_epsilon":     0.2,     # Clipping-Grenze
    "entropy_coeff":    0.0,     # Entropie-Bonus (0.01 für diskret)
    "value_coeff":      0.5,     # V-Funktion Loss Gewicht
    "max_grad_norm":    0.5,     # Gradient Clipping
}
```

---

## M13 · Model-Based RL *(3 Stunden)*

### Model-Free vs. Model-Based

<!-- ILLUSTRATION: Model-Free vs Model-Based -->
<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="mb1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Model-Free (left) -->
  <rect x="20" y="60" width="130" height="80" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="85" y="88" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">Reale</text>
  <text x="85" y="104" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">Umgebung</text>
  <text x="85" y="125" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">teuer: real/simuliert</text>

  <rect x="210" y="60" width="130" height="80" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="275" y="88" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#3C3489">Agent</text>
  <text x="275" y="104" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#3C3489">(Model-Free)</text>
  <text x="275" y="125" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">Q / π direkt</text>

  <line x1="150" y1="100" x2="208" y2="100" stroke="#888780" stroke-width="1.5" marker-end="url(#mb1)"/>
  <text x="175" y="52" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#888780">Model-Free</text>

  <!-- Model-Based (right) -->
  <rect x="390" y="20" width="120" height="60" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="450" y="48" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#085041">Welt-Modell</text>
  <text x="450" y="65" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">P̂(s'|s,a), R̂</text>

  <rect x="390" y="120" width="120" height="60" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="450" y="145" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#0C447C">Agent</text>
  <text x="450" y="162" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">plant in Modell</text>

  <rect x="560" y="60" width="100" height="80" rx="10" fill="#FAEEDA" stroke="#BA7517" stroke-width="1.5"/>
  <text x="610" y="88" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#633806">Reale</text>
  <text x="610" y="104" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#633806">Env</text>
  <text x="610" y="120" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">selten nötig</text>

  <line x1="610" y1="60" x2="520" y2="48" stroke="#888780" stroke-width="1.2" stroke-dasharray="3,2" marker-end="url(#mb1)"/>
  <line x1="510" y1="80" x2="510" y2="120" stroke="#888780" stroke-width="1.5" marker-end="url(#mb1)"/>
  <line x1="510" y1="80" x2="562" y2="85" stroke="#888780" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#mb1)"/>
  <text x="480" y="10" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#1D9E75">Model-Based (sample-effizienter)</text>
</svg>

### Dyna-Architektur

Dyna kombiniert Model-Free Updates mit **Planung** im gelernten Modell:

1. **Real-Step:** Interagiere mit echter Umgebung → speichere $(s, a, r, s')$
2. **Model-Learn:** Update $\hat{P}$ und $\hat{R}$ aus realen Daten
3. **Planung:** Simuliere $n$ Schritte im Modell → Q-Learning Updates

### MuZero (DeepMind, 2019)

MuZero lernt ein **latentes** Weltmodell und kombiniert MCTS-Planung mit Deep RL — ohne direkten Zugriff auf die echten Übergangsregeln. Erreicht SOTA in Atari, Chess, Go und Shogi.

---

## M14 · Explorationsstrategien *(3 Stunden)*

<!-- ILLUSTRATION: Explorations-Spektrum -->
<svg width="100%" viewBox="0 0 680 170" xmlns="http://www.w3.org/2000/svg">
  <!-- Achse -->
  <line x1="40" y1="90" x2="640" y2="90" stroke="#888780" stroke-width="2"/>
  <text x="40" y="115" font-family="sans-serif" font-size="11" fill="#888780">Einfach</text>
  <text x="560" y="115" font-family="sans-serif" font-size="11" fill="#888780">Komplex</text>
  <!-- Strategies -->
  <circle cx="90" cy="90" r="28" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="90" y="87" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#0C447C">ε-</text>
  <text x="90" y="100" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#0C447C">Greedy</text>
  <text x="90" y="135" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">Zufällig</text>

  <circle cx="210" cy="90" r="28" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="210" y="87" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#3C3489">UCB</text>
  <text x="210" y="100" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#3C3489">Bonus</text>
  <text x="210" y="135" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">Count-based</text>

  <circle cx="340" cy="90" r="28" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="340" y="87" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#085041">Noisy</text>
  <text x="340" y="100" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#085041">Nets</text>
  <text x="340" y="135" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">Stochastisch</text>

  <circle cx="470" cy="90" r="28" fill="#FAEEDA" stroke="#BA7517" stroke-width="1.5"/>
  <text x="470" y="87" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#633806">ICM</text>
  <text x="470" y="100" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#633806">Curiosity</text>
  <text x="470" y="135" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">Intrinsisch</text>

  <circle cx="590" cy="90" r="28" fill="#FAECE7" stroke="#D85A30" stroke-width="1.5"/>
  <text x="590" y="87" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="600" fill="#712B13">RND</text>
  <text x="590" y="100" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#D85A30">Random Netz</text>
  <text x="590" y="135" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">Distillation</text>

  <text x="340" y="160" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Bonus r_int = η · ||f(s_{t+1}) − ĝ(s_{t+1})||²  (RND)</text>
</svg>

### Intrinsic Curiosity Module (ICM)

Der Agent erhält einen **intrinsischen Bonus** für unvorhergesehene Zustandsübergänge:

- **Forward Model:** Sagt $\hat{s}_{t+1}$ aus $(s_t, a_t)$ voraus
- **Inverse Model:** Sagt $\hat{a}_t$ aus $(s_t, s_{t+1})$ voraus (Feature Learning)
- **Bonus:** $r^i_t = \frac{\eta}{2} \|\hat{s}_{t+1} - \phi(s_{t+1})\|^2$

---

## Übungsaufgaben Phase 3

1. **DQN für Atari:** Implementiere DQN für `Breakout-v5` mit Experience Replay und Target Network. Plotte die Lernkurve über 1M Schritte.

2. **Double DQN Ablation:** Zeige experimentell, dass Double DQN Q-Überschätzung reduziert. Vergleiche Q-Werte und finale Performance.

3. **PPO für MuJoCo:** Trainiere PPO auf `HalfCheetah-v4`. Variiere `clip_epsilon` ∈ {0.1, 0.2, 0.3} und analysiere den Einfluss auf Stabilität.

4. **Curiosity in SparseMontezuma:** Implementiere RND für `MontezumaRevenge-v5` und zeige, dass intrinsische Belohnungen bei sparse rewards essenziell sind.

5. **Model-Based Mini-Projekt:** Implementiere einen einfachen Dyna-Q-Algorithmus für `Maze-v0` und vergleiche Sample-Effizienz mit reinem Q-Learning.

---

## Empfohlene Literatur (Phase 3)

- **Mnih et al. (2015):** Playing Atari with Deep Reinforcement Learning (DQN)
- **Schulman et al. (2017):** Proximal Policy Optimization Algorithms (PPO)
- **Hessel et al. (2018):** Rainbow: Combining Improvements in Deep RL
- **Pathak et al. (2017):** Curiosity-driven Exploration by Self-Supervised Prediction (ICM)
- **Schrittwieser et al. (2020):** Mastering Atari, Go, Chess and Shogi with MuZero

---

*← [Phase 2: Klassisches RL](./phase2_klassisches_rl.md) · → [Phase 4: Multi-Agent RL](./phase4_multi_agent_rl.md)*
