# 1. Spatiotemporale Daten für SNNs
# Requirement: pip install snntorch torch numpy matplotlib
# Wir simulieren eine Event-Kamera: Jedes Pixel „feuert", wenn sich dort etwas bewegt.

import torch
import numpy as np
import matplotlib.pyplot as plt

# Gittergröße und Zeitschritte
GRID_SIZE = 8       # 8x8 Pixel
TIME_STEPS = 20     # 20 Zeitschritte

def generate_movement(direction: str, noise: float = 0.1) -> torch.Tensor:
    """
    Erzeugt einen Spike-Train für ein sich bewegendes Objekt.
    Gibt einen Tensor der Form (TIME_STEPS, GRID_SIZE * GRID_SIZE) zurück.
    
    Richtungen: 'right', 'down', 'diagonal'
    """
    spikes = torch.zeros(TIME_STEPS, GRID_SIZE, GRID_SIZE)

    for t in range(TIME_STEPS):
        # Position des Objekts zum Zeitpunkt t
        if direction == 'right':
            x, y = t % GRID_SIZE, GRID_SIZE // 2
        elif direction == 'down':
            x, y = GRID_SIZE // 2, t % GRID_SIZE
        elif direction == 'diagonal':
            x, y = t % GRID_SIZE, t % GRID_SIZE
        
        spikes[t, y, x] = 1.0

    # Rauschen hinzufügen (wie echte Event-Kamera)
    noise_mask = torch.rand_like(spikes) < noise
    spikes = torch.clamp(spikes + noise_mask.float(), 0, 1)

    return spikes.view(TIME_STEPS, -1)  # (T, 64)


def create_dataset(n_samples: int = 300):
    """Erstellt einen Datensatz mit 3 Klassen."""
    directions = ['right', 'down', 'diagonal']
    X, y = [], []

    for label, direction in enumerate(directions):
        for _ in range(n_samples // 3):
            X.append(generate_movement(direction))
            y.append(label)

    X = torch.stack(X)     # (N, T, 64)
    y = torch.tensor(y)    # (N,)
    
    # In (T, N, Features) umformen – so erwartet snnTorch es
    return X.permute(1, 0, 2), y

X, y = create_dataset(n_samples=300)
print(f"Daten-Shape: {X.shape}")  # (20, 300, 64)
print(f"Label-Shape: {y.shape}")  # (300,)