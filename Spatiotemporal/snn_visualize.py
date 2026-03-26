# Spike-Aktivität visualisieren

def visualize_spikes(model, sample_idx=0):
    """Zeigt, wie das Netz auf eine Bewegung über Zeit reagiert."""
    model.eval()
    with torch.no_grad():
        x_single = X[:, sample_idx:sample_idx+1, :]  # (T, 1, 64)
        spk_out, mem_out = model(x_single)
        
        # spk_out: (T, 1, 3) → (T, 3)
        spikes = spk_out.squeeze(1).numpy()
        membrane = mem_out.squeeze(1).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    directions = ['→ right', '↓ down', '↘ diagonal']

    # Spike-Raster
    axes[0].set_title(f"Output-Spikes über Zeit (Sample: {directions[y[sample_idx]]})")
    for neuron in range(3):
        spike_times = np.where(spikes[:, neuron] > 0.5)[0]
        axes[0].scatter(spike_times, 
                        np.full_like(spike_times, neuron),
                        marker='|', s=200, 
                        label=directions[neuron])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(directions)
    axes[0].set_xlabel("Zeitschritt")
    axes[0].legend()

    # Membranpotenzial
    axes[1].set_title("Membranpotenzial der Output-Neuronen")
    for neuron in range(3):
        axes[1].plot(membrane[:, neuron], label=directions[neuron])
    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Schwellwert')
    axes[1].set_xlabel("Zeitschritt")
    axes[1].set_ylabel("Membranpotenzial")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("spike_visualization.png", dpi=150)
    plt.show()

visualize_spikes(model, sample_idx=0)