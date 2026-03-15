# Phase 5 · RL in der Praxis
**Umfang:** 14 Stunden · Module M19–M22  
**Voraussetzung:** Alle Phasen 1–4 abgeschlossen

---

## Lernziele dieser Phase

Nach Abschluss von Phase 5 können die Teilnehmer:
- RL-Agenten für Games und Simulationsumgebungen trainieren und evaluieren
- Sim-to-Real-Transfer-Methoden für Robotik-Anwendungen beschreiben und anwenden
- RLHF-Pipelines für LLMs verstehen und DPO als Alternative zu PPO implementieren
- RL-Projekte professionell debuggen, evaluieren und optimieren
- Ein eigenständiges RL-Projekt von der Problemformulierung bis zum Ergebnis durchführen

---

## M19 · RL für Games & Simulationen *(3 Stunden)*

### Gymnasium: Die Standard-Testumgebung

Gymnasium (ehemals OpenAI Gym) bietet eine einheitliche API für hunderte RL-Umgebungen:

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()         # Zufällige Aktion
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Umgebungskategorien

<!-- ILLUSTRATION: Gymnasium Umgebungen Übersicht -->
<svg width="100%" viewBox="0 0 680 230" xmlns="http://www.w3.org/2000/svg">
  <!-- Classic Control -->
  <rect x="20" y="20" width="140" height="190" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
  <text x="90" y="44" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#0C447C">Classic Control</text>
  <text x="90" y="64" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">CartPole-v1</text>
  <text x="90" y="80" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">MountainCar-v0</text>
  <text x="90" y="96" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">Acrobot-v1</text>
  <text x="90" y="112" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">Pendulum-v1</text>
  <!-- CartPole schematic -->
  <rect x="65" y="145" width="50" height="8" rx="2" fill="#185FA5"/>
  <line x1="90" y1="145" x2="75" y2="118" stroke="#0C447C" stroke-width="2"/>
  <circle cx="75" cy="116" r="4" fill="#378ADD"/>
  <text x="90" y="210" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">diskret / kont.</text>

  <!-- Box2D -->
  <rect x="178" y="20" width="140" height="190" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="248" y="44" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#085041">Box2D</text>
  <text x="248" y="64" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">LunarLander-v2</text>
  <text x="248" y="80" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">BipedalWalker-v3</text>
  <text x="248" y="96" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">CarRacing-v2</text>
  <!-- Lunar lander schematic -->
  <polygon points="248,140 238,170 258,170" fill="none" stroke="#1D9E75" stroke-width="1.5"/>
  <line x1="238" y1="170" x2="225" y2="185" stroke="#1D9E75" stroke-width="1.5"/>
  <line x1="258" y1="170" x2="271" y2="185" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="248" y="210" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">kontinuierlich</text>

  <!-- Atari -->
  <rect x="336" y="20" width="140" height="190" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="406" y="44" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#3C3489">Atari</text>
  <text x="406" y="64" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">Breakout-v5</text>
  <text x="406" y="80" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">Space Invaders</text>
  <text x="406" y="96" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">Montezuma's Rev.</text>
  <text x="406" y="112" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">57 Spiele</text>
  <!-- Pixel grid schematic -->
  <rect x="376" y="130" width="60" height="55" rx="3" fill="#534AB7"/>
  <rect x="379" y="133" width="12" height="12" rx="1" fill="#AFA9EC"/>
  <rect x="393" y="133" width="12" height="12" rx="1" fill="#AFA9EC"/>
  <rect x="407" y="133" width="12" height="12" rx="1" fill="#7F77DD"/>
  <rect x="421" y="133" width="12" height="12" rx="1" fill="#AFA9EC"/>
  <text x="406" y="210" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">Pixel-Input (84×84)</text>

  <!-- MuJoCo -->
  <rect x="494" y="20" width="166" height="190" rx="10" fill="#FAEEDA" stroke="#BA7517" stroke-width="1.5"/>
  <text x="577" y="44" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#633806">MuJoCo</text>
  <text x="577" y="64" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">HalfCheetah-v4</text>
  <text x="577" y="80" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">Ant-v4</text>
  <text x="577" y="96" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">Humanoid-v4</text>
  <text x="577" y="112" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">Hopper-v4</text>
  <!-- Robot schematic -->
  <line x1="577" y1="135" x2="577" y2="160" stroke="#BA7517" stroke-width="2"/>
  <line x1="577" y1="148" x2="557" y2="165" stroke="#BA7517" stroke-width="2"/>
  <line x1="577" y1="148" x2="597" y2="165" stroke="#BA7517" stroke-width="2"/>
  <line x1="577" y1="160" x2="560" y2="185" stroke="#BA7517" stroke-width="2"/>
  <line x1="577" y1="160" x2="594" y2="185" stroke="#BA7517" stroke-width="2"/>
  <circle cx="577" cy="130" r="8" fill="#EF9F27"/>
  <text x="577" y="210" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">kontinuierlich, physikalisch</text>
</svg>

### Reward Shaping & Curriculum Learning

**Reward Shaping:** Ergänze den spärlichen Originalreward durch dichte Zwischenbelohnungen:

$$R'(s,a,s') = R(s,a,s') + F(s,a,s')$$

**Curriculum Learning:** Starte mit einfachen Aufgaben, erhöhe die Schwierigkeit graduell:

```
Leicht: kurze Distanz, kein Wind, langsam
  ↓
Mittel: mittlere Distanz, leichter Wind
  ↓  
Schwer: beliebige Startposition, starker Wind, Hinternisse
```

---

## M20 · RL für Robotik *(3 Stunden)*

### Das Sim-to-Real Problem

Agenten, die in der Simulation trainiert wurden, versagen oft in der realen Welt — die sogenannte **Reality Gap**:

<!-- ILLUSTRATION: Sim-to-Real Gap -->
<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="str1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Sim Box -->
  <rect x="20" y="50" width="220" height="120" rx="12" fill="#EEEDFE" stroke="#534AB7" stroke-width="2"/>
  <text x="130" y="80" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#3C3489">Simulation</text>
  <text x="130" y="100" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">perfekte Physik</text>
  <text x="130" y="116" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">keine Messrauschen</text>
  <text x="130" y="132" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">kein Verschleiß</text>
  <text x="130" y="150" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#534AB7">schnell, billig</text>

  <!-- Gap -->
  <rect x="268" y="80" width="140" height="60" rx="8" fill="#FCEBEB" stroke="#A32D2D" stroke-width="1.5"/>
  <text x="338" y="104" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#501313">Reality Gap</text>
  <text x="338" y="124" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#A32D2D">Modell-Fehler</text>

  <!-- Real Box -->
  <rect x="436" y="50" width="220" height="120" rx="12" fill="#E1F5EE" stroke="#1D9E75" stroke-width="2"/>
  <text x="546" y="80" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#085041">Realität</text>
  <text x="546" y="100" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">Reibung, Latenz</text>
  <text x="546" y="116" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">Messrauschen</text>
  <text x="546" y="132" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">Verschleiß der Motoren</text>
  <text x="546" y="150" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">unbekannte Dynamik</text>

  <line x1="240" y1="110" x2="266" y2="110" stroke="#A32D2D" stroke-width="1.5" marker-end="url(#str1)"/>
  <line x1="408" y1="110" x2="434" y2="110" stroke="#1D9E75" stroke-width="1.5" marker-end="url(#str1)"/>

  <!-- Solution label -->
  <text x="340" y="178" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#888780">Lösungen: Domain Randomization · Fine-tuning · Adaptive RL</text>
</svg>

### Domain Randomization

Beim Training werden **Simulationsparameter zufällig variiert**, sodass der Agent robuster wird:

```python
# Domain Randomization Beispiel
def randomize_domain(env):
    # Physikalische Parameter randomisieren
    env.model.geom_friction[:] = np.random.uniform(0.5, 2.0)
    env.model.body_mass[1:] *= np.random.uniform(0.8, 1.2)
    env.model.actuator_gear[:] *= np.random.uniform(0.9, 1.1)
    
    # Sensorausfälle simulieren
    obs_noise = np.random.normal(0, 0.01, size=env.obs_dim)
    
    return env, obs_noise
```

### Safe RL & Constrained MDPs

In realen Robotikanwendungen gibt es **Sicherheitsconstraints**:

$$\max_\pi \mathbb{E}\left[\sum_t r_t\right] \quad \text{s.t. } \mathbb{E}\left[\sum_t c_t\right] \leq d$$

- $c_t$ — Kosten (z.B. Gelenk-Drehmoment überschritten)
- $d$ — Constraint-Grenze (z.B. max. kumulierte Kosten)

**Algorithmen:** CPO (Constrained Policy Optimization), FOCOPS, Safety Gymnasium

---

## M21 · RLHF & Large Language Models *(4 Stunden)*

### Warum RLHF?

Sprachmodelle, die nur auf Next-Token-Prediction trainiert sind, sind **nicht alignment-optimiert** — sie können:
- Falsche Fakten überzeugend formulieren (Halluzinationen)
- Schädliche oder unangemessene Inhalte generieren
- Die eigentliche Intent des Nutzers missverstehen

RLHF (Reinforcement Learning from Human Feedback) fine-tuned das Modell auf menschliche Präferenzen.

### Die RLHF-Pipeline

<!-- ILLUSTRATION: RLHF Pipeline -->
<svg width="100%" viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="rlhf1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Step 1: SFT -->
  <rect x="20" y="20" width="180" height="80" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="2"/>
  <text x="27" y="42" font-family="sans-serif" font-size="10" font-weight="700" fill="#185FA5">Schritt 1</text>
  <text x="110" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#0C447C">SFT</text>
  <text x="110" y="78" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">Supervised Fine-Tuning</text>
  <text x="110" y="93" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#185FA5">auf kuratierte Demos</text>

  <!-- Step 2: Reward Model -->
  <rect x="250" y="20" width="180" height="80" rx="10" fill="#FAEEDA" stroke="#BA7517" stroke-width="2"/>
  <text x="257" y="42" font-family="sans-serif" font-size="10" font-weight="700" fill="#BA7517">Schritt 2</text>
  <text x="340" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#633806">Reward Modell</text>
  <text x="340" y="78" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">Mensch bewertet Paare</text>
  <text x="340" y="93" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#BA7517">y_A vs. y_B → RM lernt</text>

  <!-- Step 3: PPO -->
  <rect x="480" y="20" width="180" height="80" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="2"/>
  <text x="487" y="42" font-family="sans-serif" font-size="10" font-weight="700" fill="#1D9E75">Schritt 3</text>
  <text x="570" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#085041">PPO Fine-tuning</text>
  <text x="570" y="78" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">LLM = Policy</text>
  <text x="570" y="93" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1D9E75">RM = Reward-Signal</text>

  <!-- Arrows between steps -->
  <line x1="200" y1="60" x2="248" y2="60" stroke="#888780" stroke-width="1.5" marker-end="url(#rlhf1)"/>
  <line x1="430" y1="60" x2="478" y2="60" stroke="#888780" stroke-width="1.5" marker-end="url(#rlhf1)"/>

  <!-- Detail PPO RLHF -->
  <rect x="60" y="145" width="560" height="100" rx="10" fill="#F1EFE8" stroke="#888780" stroke-width="1"/>
  <text x="340" y="168" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#2C2C2A">PPO-RLHF Reward-Formel</text>

  <text x="100" y="195" font-family="sans-serif" font-size="11" fill="#2C2C2A">r(s,a) = RM(prompt, response)</text>
  <text x="100" y="215" font-family="sans-serif" font-size="11" fill="#D85A30">         − β · KL[π_θ(·|s) || π_ref(·|s)]</text>
  <text x="400" y="195" font-family="sans-serif" font-size="10" fill="#888780">← Reward vom gelernten Modell</text>
  <text x="400" y="215" font-family="sans-serif" font-size="10" fill="#D85A30">← KL-Penalty: verhindert Overoptimierung</text>
  <text x="400" y="232" font-family="sans-serif" font-size="10" fill="#534AB7">   (Abstand von SFT-Baseline π_ref)</text>
</svg>

### Direct Preference Optimization (DPO)

DPO umgeht das explizite Reward-Modell und optimiert die Policy **direkt auf Präferenzpaaren**:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

- $y_w$ — bevorzugte Antwort (winner)
- $y_l$ — abgelehnte Antwort (loser)
- $\beta$ — KL-Penalty-Stärke
- $\pi_{ref}$ — SFT-Referenzmodell

**Vorteil:** Einfacher, stabiler, kein separates RM-Training nötig.

### RLHF vs. DPO vs. Constitutional AI

| | RLHF (PPO) | DPO | Constitutional AI |
|---|---|---|---|
| **Reward Model** | Explizit | Implizit | Regelbasiert |
| **Komplexität** | Hoch | Mittel | Mittel |
| **Stabilität** | Schwierig | Stabil | Stabil |
| **Eingesetzt von** | ChatGPT (OpenAI) | LLaMA 3 (Meta) | Claude (Anthropic) |
| **Menschl. Labels** | Viele nötig | Weniger | Minimal |

---

## M22 · Abschlussprojekt & Best Practices *(4 Stunden)*

### Debugging von RL-Agenten

<!-- ILLUSTRATION: RL Debugging Checkliste -->
<svg width="100%" viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg">
  <!-- Header -->
  <rect x="20" y="10" width="640" height="40" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="340" y="35" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#3C3489">RL Debugging Checkliste</text>

  <!-- Column 1: Sanity Checks -->
  <rect x="20" y="65" width="200" height="185" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>
  <text x="120" y="88" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#0C447C">Sanity Checks</text>
  <text x="30" y="110" font-family="sans-serif" font-size="10" fill="#185FA5">✓ Reward-Range prüfen</text>
  <text x="30" y="128" font-family="sans-serif" font-size="10" fill="#185FA5">✓ Obs-Normalisierung</text>
  <text x="30" y="146" font-family="sans-serif" font-size="10" fill="#185FA5">✓ Zufällige Policy testen</text>
  <text x="30" y="164" font-family="sans-serif" font-size="10" fill="#185FA5">✓ Overfitting auf 1 Env</text>
  <text x="30" y="182" font-family="sans-serif" font-size="10" fill="#185FA5">✓ Gradient-Normen loggen</text>
  <text x="30" y="200" font-family="sans-serif" font-size="10" fill="#185FA5">✓ Entropy überwachen</text>
  <text x="30" y="218" font-family="sans-serif" font-size="10" fill="#185FA5">✓ Value-Loss < Policy-Loss?</text>
  <text x="30" y="236" font-family="sans-serif" font-size="10" fill="#185FA5">✓ Replay Buffer gefüllt?</text>

  <!-- Column 2: Hyperparameter -->
  <rect x="240" y="65" width="200" height="185" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="1"/>
  <text x="340" y="88" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#085041">Hyperparameter</text>
  <text x="250" y="110" font-family="sans-serif" font-size="10" fill="#1D9E75">lr: 3e-4 (Adam default)</text>
  <text x="250" y="128" font-family="sans-serif" font-size="10" fill="#1D9E75">γ: 0.99 (start hier)</text>
  <text x="250" y="146" font-family="sans-serif" font-size="10" fill="#1D9E75">batch_size: 64 / 256</text>
  <text x="250" y="164" font-family="sans-serif" font-size="10" fill="#1D9E75">n_envs: 8–32 parallel</text>
  <text x="250" y="182" font-family="sans-serif" font-size="10" fill="#1D9E75">max_grad_norm: 0.5</text>
  <text x="250" y="200" font-family="sans-serif" font-size="10" fill="#1D9E75">entropy_coef: 0.0–0.01</text>
  <text x="250" y="218" font-family="sans-serif" font-size="10" fill="#1D9E75">n_steps (PPO): 2048</text>
  <text x="250" y="236" font-family="sans-serif" font-size="10" fill="#1D9E75">clip_eps (PPO): 0.2</text>

  <!-- Column 3: Monitoring -->
  <rect x="460" y="65" width="200" height="185" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="1"/>
  <text x="560" y="88" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#633806">Monitoring (W&B)</text>
  <text x="470" y="110" font-family="sans-serif" font-size="10" fill="#BA7517">episode_reward (mean)</text>
  <text x="470" y="128" font-family="sans-serif" font-size="10" fill="#BA7517">episode_length</text>
  <text x="470" y="146" font-family="sans-serif" font-size="10" fill="#BA7517">policy_entropy</text>
  <text x="470" y="164" font-family="sans-serif" font-size="10" fill="#BA7517">value_loss</text>
  <text x="470" y="182" font-family="sans-serif" font-size="10" fill="#BA7517">policy_loss</text>
  <text x="470" y="200" font-family="sans-serif" font-size="10" fill="#BA7517">approx_kl</text>
  <text x="470" y="218" font-family="sans-serif" font-size="10" fill="#BA7517">clip_fraction (PPO)</text>
  <text x="470" y="236" font-family="sans-serif" font-size="10" fill="#BA7517">grad_norm</text>
</svg>

### Evaluation & Benchmarking

```python
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

# Standard Evaluierung: 10 Episoden, kein Render
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# Mehrere Seeds für statistische Signifikanz
seeds = [0, 1, 2, 3, 4]
all_rewards = []
for seed in seeds:
    env.reset(seed=seed)
    r, _ = evaluate_policy(model, env, n_eval_episodes=5)
    all_rewards.append(r)

print(f"Über {len(seeds)} Seeds: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
```

### Empfehlungen für das Abschlussprojekt

**Mögliche Themen:**

| Thema | Empfohlener Algorithmus | Umgebung |
|---|---|---|
| Spieleagent | DQN / Rainbow | Atari, Retro |
| Roboter-Laufen | PPO / SAC | MuJoCo, Isaac Gym |
| Ressourcenoptimierung | PPO / REINFORCE | Custom Gym Env |
| Multi-Agent Spiel | QMIX / MAPPO | SMAC, MPE |
| LLM-Alignment | DPO | HuggingFace TRL |
| Autonomes Fahren | SAC / TD3 | CARLA, Highway-Env |

**Projektstruktur (empfohlen):**

```
rl_project/
├── README.md          # Motivation, Methode, Ergebnisse
├── config/
│   └── config.yaml   # Alle Hyperparameter
├── envs/
│   └── custom_env.py # Eigene Gymnasium-Umgebung
├── agents/
│   └── ppo_agent.py  # Algorithmus-Implementierung
├── train.py          # Training-Script
├── eval.py           # Evaluation + Plots
├── notebooks/
│   └── analysis.ipynb
└── results/
    ├── models/       # gespeicherte Checkpoints
    └── plots/        # Lernkurven, Videos
```

### Wichtige Bibliotheken

```bash
# Installation
pip install gymnasium[all]          # Umgebungen
pip install stable-baselines3       # Referenzimpl. (DQN, PPO, SAC, ...)
pip install sb3-contrib             # Extras (QRDQN, TQC, ...)
pip install pettingzoo              # Multi-Agent Umgebungen
pip install wandb                   # Experiment-Tracking
pip install optuna                  # Hyperparameter-Optimierung

# Für RLHF / LLMs
pip install trl                     # Transformer RL (HuggingFace)
pip install datasets transformers   # LLM-Basis
```

---

## Kurs-Abschluss: RL Algorithmen Übersicht

<!-- ILLUSTRATION: Taxonomie RL-Algorithmen -->
<svg width="100%" viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="tax1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Root -->
  <rect x="250" y="10" width="180" height="44" rx="10" fill="#2C2C2A" stroke="#444441" stroke-width="1.5"/>
  <text x="340" y="37" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="white">RL Algorithmen</text>

  <!-- Model-Free / Model-Based -->
  <rect x="60" y="90" width="150" height="40" rx="8" fill="#3C3489" stroke="#534AB7" stroke-width="1.5"/>
  <text x="135" y="115" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">Model-Free</text>
  <rect x="470" y="90" width="150" height="40" rx="8" fill="#085041" stroke="#1D9E75" stroke-width="1.5"/>
  <text x="545" y="115" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">Model-Based</text>
  <line x1="340" y1="54" x2="135" y2="88" stroke="#888780" stroke-width="1.5" marker-end="url(#tax1)"/>
  <line x1="340" y1="54" x2="545" y2="88" stroke="#888780" stroke-width="1.5" marker-end="url(#tax1)"/>

  <!-- Value-Based / Policy Gradient -->
  <rect x="20" y="170" width="130" height="40" rx="8" fill="#0C447C" stroke="#185FA5" stroke-width="1.5"/>
  <text x="85" y="195" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="700" fill="white">Value-Based</text>
  <rect x="170" y="170" width="130" height="40" rx="8" fill="#633806" stroke="#BA7517" stroke-width="1.5"/>
  <text x="235" y="195" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="700" fill="white">Policy Gradient</text>
  <line x1="135" y1="130" x2="85" y2="168" stroke="#888780" stroke-width="1.2" marker-end="url(#tax1)"/>
  <line x1="135" y1="130" x2="235" y2="168" stroke="#888780" stroke-width="1.2" marker-end="url(#tax1)"/>

  <!-- Value-Based Algorithms -->
  <rect x="10" y="245" width="100" height="35" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>
  <text x="60" y="268" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#0C447C">Q-Learning</text>
  <rect x="120" y="245" width="80" height="35" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>
  <text x="160" y="261" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#0C447C">DQN</text>
  <text x="160" y="275" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">Rainbow</text>
  <line x1="85" y1="210" x2="60" y2="243" stroke="#185FA5" stroke-width="1" marker-end="url(#tax1)"/>
  <line x1="85" y1="210" x2="160" y2="243" stroke="#185FA5" stroke-width="1" marker-end="url(#tax1)"/>

  <!-- Policy Gradient Algorithms -->
  <rect x="158" y="245" width="70" height="35" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="1"/>
  <text x="193" y="268" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#633806">REINFORCE</text>
  <rect x="238" y="245" width="70" height="35" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="1"/>
  <text x="273" y="261" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#633806">A2C</text>
  <text x="273" y="275" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">A3C</text>
  <rect x="318" y="245" width="60" height="35" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="1"/>
  <text x="348" y="261" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#633806">PPO</text>
  <text x="348" y="275" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#888780">TRPO</text>
  <line x1="235" y1="210" x2="193" y2="243" stroke="#BA7517" stroke-width="1" marker-end="url(#tax1)"/>
  <line x1="235" y1="210" x2="273" y2="243" stroke="#BA7517" stroke-width="1" marker-end="url(#tax1)"/>
  <line x1="235" y1="210" x2="348" y2="243" stroke="#BA7517" stroke-width="1" marker-end="url(#tax1)"/>

  <!-- Model-Based Algorithms -->
  <rect x="400" y="170" width="100" height="40" rx="8" fill="#085041" stroke="#1D9E75" stroke-width="1"/>
  <text x="450" y="188" text-anchor="middle" font-family="sans-serif" font-size="10" fill="white">Planning</text>
  <text x="450" y="204" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9FE1CB">MCTS, Dyna</text>
  <rect x="520" y="170" width="130" height="40" rx="8" fill="#085041" stroke="#1D9E75" stroke-width="1"/>
  <text x="585" y="188" text-anchor="middle" font-family="sans-serif" font-size="10" fill="white">World Models</text>
  <text x="585" y="204" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9FE1CB">MuZero, Dreamer</text>
  <line x1="545" y1="130" x2="450" y2="168" stroke="#888780" stroke-width="1.2" marker-end="url(#tax1)"/>
  <line x1="545" y1="130" x2="585" y2="168" stroke="#888780" stroke-width="1.2" marker-end="url(#tax1)"/>

  <text x="340" y="310" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#888780">Alle Algorithmen aus Phasen 1–5 eingeordnet</text>
</svg>

---

## Finales Abschlussprojekt

### Aufgabenstellung

Jeder Teilnehmer wählt ein Projekt aus und präsentiert es am letzten Kurstag (20 min Vortrag + 10 min Demo):

**Mindestanforderungen:**
- Eigenständige Gymnasium-Umgebung oder Standard-Env
- Implementierung von mindestens einem Algorithmus aus Phase 3–5
- Wandb-Lernkurve über mindestens 500k Schritte
- Ablations-Experiment (mindestens ein Hyperparameter variiert)
- 5-minütiges Demo-Video

**Bewertungskriterien:**

| Kriterium | Gewichtung |
|---|---|
| Korrekte Implementierung | 30% |
| Experimentelles Design & Ablation | 25% |
| Ergebnisqualität & Benchmarking | 20% |
| Präsentation & Verständlichkeit | 15% |
| Code-Qualität & Reproduzierbarkeit | 10% |

---

## Empfohlene Literatur (Phase 5)

- **Brockman et al. (2016):** OpenAI Gym (Gymnasium-Ursprungspaper)
- **Ouyang et al. (2022):** Training Language Models to Follow Instructions with Human Feedback (InstructGPT/RLHF)
- **Rafailov et al. (2023):** Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)
- **Christiano et al. (2017):** Deep Reinforcement Learning from Human Preferences
- **Tobin et al. (2017):** Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
- **Stable-Baselines3 Docs:** [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)

---

## Herzlichen Glückwunsch zum Kursabschluss!

Du hast alle 5 Phasen des Reinforcement-Learning-Kurses abgeschlossen:

| Phase | Thema | Stunden |
|---|---|---|
| ✅ Phase 1 | Grundlagen & MDP-Framework | 14 h |
| ✅ Phase 2 | Klassisches RL | 20 h |
| ✅ Phase 3 | Deep Reinforcement Learning | 18 h |
| ✅ Phase 4 | Multi-Agent RL | 14 h |
| ✅ Phase 5 | RL in der Praxis | 14 h |
| **Gesamt** | | **80 h** |

*← [Phase 4: Multi-Agent RL](./phase4_multi_agent_rl.md)*
