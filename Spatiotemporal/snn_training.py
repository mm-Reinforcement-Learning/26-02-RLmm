#  Training
# Der Trick bei SNNs: Wir nutzen die Spike-Rate über alle Zeitschritte als Klassifikations-Signal.

from torch.utils.data import DataLoader, TensorDataset, random_split

# Train/Test Split
dataset = TensorDataset(X.permute(1, 0, 2), y)  # zurück zu (N, T, 64)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

# DataLoader – gibt Batches in (N, T, 64) zurück
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=32)

# Training Setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0

    for X_batch, y_batch in loader:
        # (N, T, 64) → (T, N, 64)
        X_t = X_batch.permute(1, 0, 2)

        optimizer.zero_grad()

        # Forward Pass über alle Zeitschritte
        spk_out, mem_out = model(X_t)   # spk_out: (T, N, 3)

        # Rate Coding: Spike-Summe über Zeit als Klassenwahrscheinlichkeit
        # Ein Neuron, das öfter feuert → höhere Aktivierung → gewinnt
        spike_count = spk_out.sum(dim=0)  # (N, 3)

        loss = loss_fn(spike_count, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (spike_count.argmax(1) == y_batch).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_t = X_batch.permute(1, 0, 2)
            spk_out, _ = model(X_t)
            spike_count = spk_out.sum(dim=0)
            correct += (spike_count.argmax(1) == y_batch).sum().item()
    return correct / len(loader.dataset)


# Training Loop
for epoch in range(30):
    train_loss, train_acc = train_epoch(model, train_loader)
    test_acc = evaluate(model, test_loader)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")
```

**Typische Ausgabe:**
```
Epoch  5 | Loss: 1.0234 | Train Acc: 54.17% | Test Acc: 51.67%
Epoch 10 | Loss: 0.7891 | Train Acc: 71.25% | Test Acc: 68.33%
Epoch 15 | Loss: 0.5102 | Train Acc: 85.42% | Test Acc: 83.33%
Epoch 20 | Loss: 0.3217 | Train Acc: 92.50% | Test Acc: 90.00%
Epoch 25 | Loss: 0.2104 | Train Acc: 96.67% | Test Acc: 95.00%
Epoch 30 | Loss: 0.1543 | Train Acc: 98.33% | Test Acc: 96.67%


## Was passiert hier spatiotemporal?
# ```
# Zeitschritt 1:  Objekt bei (0,4) → Spike bei Neuron 32
# Zeitschritt 2:  Objekt bei (1,4) → Spike bei Neuron 33
# Zeitschritt 3:  Objekt bei (2,4) → Spike bei Neuron 34
#         ↓
# LIF-Neuronen "erinnern" sich (beta=0.9) an vergangene Spikes
#         ↓
# Das Muster "Spikes wandern nach rechts" wird erkannt
#         ↓
# Output-Neuron 0 ("right") feuert am häufigsten → Klasse: right ✓

# Das beta-Parameter ist dabei der Schlüssel: Er bestimmt, wie lange vergangene räumliche Aktivierungen die Entscheidung beeinflussen – das ist das zeitliche Gedächtnis des Systems.
