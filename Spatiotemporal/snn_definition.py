# Spiking Neural Network definieren
# Wir bauen ein LIF-Netzwerk (Leaky Integrate-and-Fire) – das Standardneuron in SNNs.

import snntorch as snn
from snntorch import surrogate
import torch.nn as nn

class SpatiotemporalSNN(nn.Module):
    """
    Ein 3-schichtiges SNN für spatiotemporale Klassifikation.
    
    Architektur:
      Input (64) → LIF → Hidden (128) → LIF → Hidden (64) → LIF → Output (3)
    
    LIF = Leaky Integrate-and-Fire Neuron:
      - Integriert eingehende Spikes
      - Feuert, wenn Membranpotenzial Schwellwert überschreitet
      - "Leckt" (vergisst) über Zeit – das erzeugt das zeitliche Gedächtnis
    """
    def __init__(self, beta: float = 0.9):
        super().__init__()

        # beta = Leak-Rate (0 = kein Gedächtnis, 1 = kein Vergessen)
        # surrogate gradient: ermöglicht Backprop durch nicht-differenzierbare Spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Synaptische Verbindungen (normale Linear-Layer)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

        # LIF-Neuronen – beta steuert das "Gedächtnis" über Zeit
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor):
        """
        x: (TIME_STEPS, BATCH, 64)
        Gibt Spike-Ausgaben und Membranpotenziale über alle Zeitschritte zurück.
        """
        # Membranpotenziale initialisieren
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Sammelt Ausgabe-Spikes und Membranpotenziale über Zeit
        spk3_rec = []
        mem3_rec = []

        for t in range(x.shape[0]):         # Über Zeitschritte iterieren
            cur1 = self.fc1(x[t])           # Synaptischer Strom, Schicht 1
            spk1, mem1 = self.lif1(cur1, mem1)   # LIF feuert oder nicht

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        # Stack: (T, Batch, 3)
        return torch.stack(spk3_rec), torch.stack(mem3_rec)


model = SpatiotemporalSNN(beta=0.9)
print(model)