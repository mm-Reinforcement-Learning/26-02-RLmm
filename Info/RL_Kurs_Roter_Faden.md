**ROTER FADEN**

**Reinforcement Learning – 80-Stunden-Kurs**

Zielgruppe: Fortgeschrittene mit Deep-Learning-Kenntnissen | Umfang: 80 Stunden | Module: 22

# **Kursübersicht**

| **Phase** | **Inhalt** | **Stunden** | **Module** |
| --- | --- | --- | --- |
| **1** | Grundlagen &amp; MDP-Framework | **14 h** | M1–M4 |
| **2** | Klassisches RL (Q-Learning, SARSA, PG) | **20 h** | M5–M9 |
| **3** | Deep Reinforcement Learning | **18 h** | M10–M14 |
| **4** | Multi-Agent RL | **14 h** | M15–M18 |
| **5** | RL in der Praxis (Robotik, Games, LLMs) | **14 h** | M19–M22 |

# **Lernziele**

Nach Abschluss des Kurses sind die Teilnehmer in der Lage:

-   Markov-Entscheidungsprozesse mathematisch zu formulieren und analytisch zu lösen.
-   Klassische RL-Algorithmen (Q-Learning, SARSA, Monte Carlo, TD) korrekt anzuwenden und zu implementieren.
-   State-of-the-Art Deep-RL-Algorithmen (DQN, PPO, A3C) zu implementieren und zu debuggen.
-   Multi-Agent-Systeme zu modellieren und kooperative sowie kompetitive Algorithmen einzusetzen.
-   RL-Methoden auf reale Problemstellungen in Robotik, Games und Large Language Models anzuwenden.

# **Detaillierter Kursplan**

| **Phase 1 Grundlagen & MDP-Framework** | **14 Stunden** |
| --- | --- |

| **M1 · Einführung & Motivation**   _RL vs. Supervised/Unsupervised Learning, Anwendungsfelder (Games, Robotik, LLMs), historischer Überblick, Taxonomie von RL-Algorithmen_ | **2 h** |
| --- | --- |

| **M2 · Markov-Entscheidungsprozesse (MDP)**   _Zustandsraum, Aktionsraum, Übergangsmodell P(s'\|s,a), Reward-Funktion R(s,a,s'), Zeithorizont (endlich / unendlich), Discount-Faktor γ_ | **4 h** |
| --- | --- |

| **M3 · Bellman-Gleichungen**   _Optimalitätsprinzip von Bellman, State-Value V\*(s) und Action-Value Q\*(s,a), Bellman-Backup-Operator, Kontraktionseigenschaft_ | **4 h** |
| --- | --- |

| **M4 · Dynamische Programmierung**   _Policy Evaluation, Policy Iteration, Value Iteration, Komplexitäts- und Konvergenzanalyse, Curse of Dimensionality_ | **4 h** |
| --- | --- |

| **Phase 2 Klassisches RL** | **20 Stunden** |
| --- | --- |

| **M5 · Monte Carlo Methoden**   _First-visit / every-visit MC-Schätzung, MC-Kontrolle (ε-greedy), Off-policy MC, Importance Sampling und seine Varianz_ | **4 h** |
| --- | --- |

| **M6 · Temporal Difference Learning**   _TD(0)-Algorithmus, TD-Fehler δ, Eligibility Traces, TD(λ), Vorwärts- vs. Rückwärtsansicht_ | **4 h** |
| --- | --- |

| **M7 · SARSA & Q-Learning**   _On-policy SARSA vs. Off-policy Q-Learning, Konvergenzbedingungen, tabellarisch vs. FA, Deadly Triad (DL+RL+Off-policy)_ | **4 h** |
| --- | --- |

| **M8 · Funktionsapproximation**   _Lineare Funktionsapproximation, Tile Coding, Radiale Basisfunktionen, Semi-Gradient-Methoden, Stabilitätsprobleme_ | **4 h** |
| --- | --- |

| **M9 · Policy Gradient Grundlagen**   _REINFORCE-Algorithmus, Policy Gradient Theorem, Baseline-Reduktion, einfaches Actor-Critic, Varianzreduktion_ | **4 h** |
| --- | --- |

| **Phase 3 Deep Reinforcement Learning** | **18 Stunden** |
| --- | --- |

| **M10 · DQN & Varianten**   _Experience Replay Buffer, Target Network, Double DQN, Dueling DQN, Prioritized Experience Replay (PER), Rainbow_ | **4 h** |
| --- | --- |

| **M11 · Policy Gradient (Deep)**   _Advantage Actor-Critic (A2C), Asynchronous A3C, asynchrones Training, Advantage-Funktion, Generalized Advantage Estimation (GAE)_ | **4 h** |
| --- | --- |

| **M12 · PPO & TRPO**   _Trust Region Policy Optimization, KL-Constraint, PPO-Clip-Mechanismus, Entropy-Regularisierung, Hyperparameter-Tuning_ | **4 h** |
| --- | --- |

| **M13 · Model-Based RL**   _World Models, Dyna-Architektur, MuZero, Dreamer V1/V2/V3, MBPO, Planungsverfahren (MCTS, CEM)_ | **3 h** |
| --- | --- |

| **M14 · Explorationsstrategien**   _ε-Greedy und Annihilierung, UCB, Thompson Sampling, Curiosity (ICM), Random Network Distillation (RND), Count-based Exploration_ | **3 h** |
| --- | --- |

| **Phase 4 Multi-Agent Reinforcement Learning** | **14 Stunden** |
| --- | --- |

| **M15 · MARL Grundlagen**   _Stochastic Games / Markov Games, Nash-Gleichgewicht, Cooperative vs. Competitive vs. Mixed Settings, Nicht-Stationarität, Skalierbarkeit_ | **4 h** |
| --- | --- |

| **M16 · Cooperative MARL**   _Centralized Training with Decentralized Execution (CTDE), QMIX, MADDPG, Kommunikationsarchitekturen, Credit-Assignment-Problem_ | **4 h** |
| --- | --- |

| **M17 · Competitive MARL & Self-Play**   _AlphaGo / AlphaZero / MuZero, OpenAI Five, Population-Based Training (PBT), League-Training, Exploitability_ | **3 h** |
| --- | --- |

| **M18 · Emergentes Verhalten & Skalierung**   _Emergente Kommunikationsprotokolle, emergente Koordination, skalierbare MARL-Algorithmen, MAPPO, HAPPO_ | **3 h** |
| --- | --- |

| **Phase 5 RL in der Praxis** | **14 Stunden** |
| --- | --- |

| **M19 · RL für Games & Simulationen**   _Atari Benchmark, MuJoCo-Umgebungen, OpenAI Gymnasium, Reward Shaping, Curriculum Learning, Sparse Rewards_ | **3 h** |
| --- | --- |

| **M20 · RL für Robotik**   _Sim-to-Real Transfer, Domain Randomization, Safe RL, Constrained MDPs (CMDPs), Cost-Constraint-Algorithmen (CPO, FOCOPS)_ | **3 h** |
| --- | --- |

| **M21 · RLHF & Large Language Models**   _Reward Modelling aus menschlichem Feedback, PPO für LLMs, Direct Preference Optimization (DPO), Constitutional AI, AI Alignment_ | **4 h** |
| --- | --- |

| **M22 · Abschlussprojekt & Best Practices**   _Debugging und Evaluation von RL-Agenten, Hyperparameter-Tuning (Optuna, W&B), Benchmarking, Projektpräsentationen_ | **4 h** |
| --- | --- |

# **Empfohlene Ressourcen**

## **Lehrbücher**

-   Sutton & Barto: Reinforcement Learning: An Introduction (2nd ed., 2018) – Das Standardwerk
-   Bertsekas: Dynamic Programming and Optimal Control – Mathematisch fundiert
-   Szepesvári: Algorithms for Reinforcement Learning – Kompakter Überblick

## **Online-Kurse & Vorlesungen**

-   David Silver (DeepMind): RL Course – UCL/YouTube (Phase 1–2)
-   Sergey Levine (UC Berkeley): CS285 Deep RL – YouTube (Phase 3–4)
-   Hugging Face: Deep RL Course – Hands-on mit Gymnasium (Phase 5)

## **Bibliotheken & Frameworks**

| **Bibliothek** | **Beschreibung** | **Phase** |
| --- | --- | --- |
| **OpenAI Gymnasium** | Standard-RL-Umgebungen (Atari, MuJoCo, etc.) | Alle Phasen |
| **Stable-Baselines3** | Referenzimplementierungen (DQN, PPO, SAC) | Phase 3–5 |
| **RLlib (Ray)** | Skalierbare MARL &amp; Distributed RL | Phase 3–4 |
| **CleanRL** | Einfache, lesbare Deep-RL-Implementierungen | Phase 3 |
| **PettingZoo** | Multi-Agent-Umgebungen | Phase 4 |

# **Didaktische Hinweise**

-   Jede Phase beginnt mit einer Wiederholung der Vorkenntnisse (ca. 15 min) und endet mit einem kurzen Quiz.
-   Jedes Modul enthält mindestens ein Programmierbeispiel in Python (PyTorch). Empfohlen wird Google Colab oder eine lokale Conda-Umgebung.
-   Die Phasen 3–5 setzen voraus, dass Phase 1–2 vollständig verstanden wurde. Bei heterogenen Gruppen Phase 1 ausweiten.
-   Phase 5 (Praxis) sollte möglichst projektbasiert unterrichtet werden. Teilnehmer wählen ein eigenes RL-Projekt aus.
-   Für die RLHF-Einheit (M21) empfiehlt sich ergänzendes Material zu Transformer-Architekturen, falls nicht alle Teilnehmer LLM-Erfahrung mitbringen.