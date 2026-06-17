"""
Qwen2.5-14B Latent Space Riemannian Curvature Experiment
- Extract embeddings via llama-cpp-python (mean pooling)
- Fit nonlinear MLP decoder
- Measure curvature via lrw PullbackMetric
- Save results as JSON + PNG
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
from pathlib import Path
from datetime import datetime

MODEL_PATH = "/home/mun/.lmstudio/models/viniciusianni/Qwen2.5-Coder-14B-Q4_K_M-GGUF/qwen2.5-coder-14b-q4_k_m.gguf"
OUTPUT_DIR = Path("results/curvature_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Embedding extraction ────────────────────────────────────
print("=" * 60)
print("Step 1: Qwen2.5-14B Embedding Extraction")
print("=" * 60)

from llama_cpp import Llama

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_gpu_layers=-1,
    embedding=True,
    pooling_type=1,
    verbose=False,
)

PROMPTS = [
    "Riemannian geometry studies the intrinsic structure of curved surfaces",
    "Differential equations describe rates of change as mathematical tools",
    "Quantum mechanics describes particles using probability amplitudes",
    "The second law of thermodynamics states that entropy always increases",
    "Korean SAT reading measures logical comprehension ability",
    "Binary opposition analyzes the structure of two contrasting concepts",
    "The narrator is the voice that conveys the story",
    "Metaphor explains unfamiliar things through familiar ones",
    "Python is an interpreted language with dynamic typing support",
    "Neural networks extract features layer by layer",
    "Gradient descent minimizes loss through iterative optimization",
    "Transformers process sequences using attention mechanisms",
    "The weather is clear today and it feels great",
    "A cup of coffee makes the morning more energetic",
    "Reading is a great way to accumulate indirect experience",
    "Music conveys emotion without words",
]

LABELS = [
    "Riemannian", "DiffEq", "Quantum", "Thermodynamics",
    "SAT-Korean", "BinaryOpp", "Narrator", "Metaphor",
    "Python", "NeuralNet", "GradDescent", "Transformer",
    "Weather", "Coffee", "Reading", "Music",
]

CATEGORIES = ["Math/Science"] * 4 + ["Humanities"] * 4 + ["Coding/AI"] * 4 + ["Everyday"] * 4
COLORS = ["#60a8f0"] * 4 + ["#f07ca0"] * 4 + ["#8b7cf8"] * 4 + ["#5ec49a"] * 4

print(f"Extracting embeddings for {len(PROMPTS)} prompts...")
embeddings = []
for i, prompt in enumerate(PROMPTS):
    emb = llm.embed(prompt)
    emb_np = np.array(emb, dtype=np.float32)
    if emb_np.ndim == 2:
        emb_np = emb_np.mean(axis=0)
    embeddings.append(emb_np)
    print(f"  [{i+1:2d}/{len(PROMPTS)}] {LABELS[i]}: shape={emb_np.shape}")

embeddings = np.array(embeddings, dtype=np.float32)
print(f"\nEmbedding matrix shape: {embeddings.shape}")

# ── 2. MLP decoder + PullbackMetric ───────────────────────────
print("\n" + "=" * 60)
print("Step 2: MLP Decoder Training + PullbackMetric Curvature")
print("=" * 60)

from lrw.metric.pullback import PullbackMetric

emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
emb_dim = emb_tensor.shape[1]
print(f"Embedding dim: {emb_dim}")

LATENT_DIM = 16
emb_mean = emb_tensor.mean(0)
U, S, Vt = torch.linalg.svd(emb_tensor - emb_mean, full_matrices=False)
emb_reduced = (U[:, :LATENT_DIM] * S[:LATENT_DIM]).detach()
print(f"PCA latent shape: {emb_reduced.shape}")

OUTPUT_DIM = 128
emb_target = ((emb_tensor - emb_mean) @ Vt[:LATENT_DIM].T)[:, :OUTPUT_DIM]
if emb_target.shape[1] < OUTPUT_DIM:
    pad = torch.zeros(emb_target.shape[0], OUTPUT_DIM - emb_target.shape[1])
    emb_target = torch.cat([emb_target, pad], dim=1)

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, z):
        return self.net(z)

mlp = MLPDecoder(LATENT_DIM, 64, OUTPUT_DIM)
optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-3)

print("Training MLP decoder (800 epochs)...")
for epoch in range(800):
    optimizer.zero_grad()
    pred = mlp(emb_reduced)
    loss = nn.functional.mse_loss(pred, emb_target)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f"  epoch {epoch+1}/800, loss={loss.item():.6f}")

mlp.eval()
print("MLP training complete")

metric = PullbackMetric(decoder=mlp, regularization=1e-4)

print("Computing metric tensors...")
metric_tensors = metric.metric_tensor(emb_reduced)
print(f"Metric tensor shape: {metric_tensors.shape}")

volume_elements = metric.local_volume_element(emb_reduced)
volume_elements_np = volume_elements.detach().numpy()

eigenvalues = torch.linalg.eigvalsh(metric_tensors)
min_eig = eigenvalues[:, 0].detach().numpy()
max_eig = eigenvalues[:, -1].detach().numpy()
condition_numbers = max_eig / (np.abs(min_eig) + 1e-10)

print(f"\nMeasurement complete!")
print(f"  Volume element range: {volume_elements_np.min():.4e} ~ {volume_elements_np.max():.4e}")
print(f"  Condition number range: {condition_numbers.min():.2f} ~ {condition_numbers.max():.2f}")

# ── 3. Riemannian distance matrix ─────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Riemannian Distance Matrix")
print("=" * 60)

n = len(PROMPTS)
dist_matrix = np.zeros((n, n))
mt_np = metric_tensors.detach().numpy()
emb_np = emb_reduced.detach().numpy()

for i in range(n):
    for j in range(n):
        if i != j:
            diff = emb_np[i] - emb_np[j]
            G_avg = (mt_np[i] + mt_np[j]) / 2
            dist_sq = diff @ G_avg @ diff
            dist_matrix[i, j] = max(dist_sq, 0) ** 0.5

print("Distance matrix computed")

# ── 4. Save JSON ───────────────────────────────────────────────
results = {
    "experiment": "qwen2.5_14b_latent_curvature_v2",
    "model": "Qwen2.5-Coder-14B-Q4_K_M",
    "decoder": "MLP(16->64->128->64->128)",
    "timestamp": datetime.now().isoformat(),
    "embedding_dim": int(emb_dim),
    "latent_dim": LATENT_DIM,
    "n_prompts": n,
    "labels": LABELS,
    "categories": CATEGORIES,
    "volume_elements": volume_elements_np.tolist(),
    "condition_numbers": condition_numbers.tolist(),
    "min_eigenvalues": min_eig.tolist(),
    "max_eigenvalues": max_eig.tolist(),
    "distance_matrix": dist_matrix.tolist(),
    "summary": {
        "volume_element_mean": float(volume_elements_np.mean()),
        "volume_element_std": float(volume_elements_np.std()),
        "condition_number_mean": float(condition_numbers.mean()),
        "condition_number_max": float(condition_numbers.max()),
        "most_curved_label": LABELS[int(condition_numbers.argmax())],
        "least_curved_label": LABELS[int(condition_numbers.argmin())],
    }
}

json_path = OUTPUT_DIR / "qwen_curvature_002.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"JSON saved: {json_path}")

# ── 5. Plot ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Generating Figures")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0e0e10')
for ax in axes:
    ax.set_facecolor('#16161a')

# Panel 1: Volume Element
ax1 = axes[0]
ax1.bar(range(n), volume_elements_np, color=COLORS, alpha=0.85, edgecolor='#2e2e38')
ax1.set_xticks(range(n))
ax1.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=8, color='#9090a8')
ax1.set_title('Local Volume Element\n(Curvature Magnitude)', color='#e8e8f0', fontsize=12, pad=10)
ax1.set_ylabel('Volume Element', color='#9090a8')
ax1.tick_params(colors='#9090a8')
ax1.spines[:].set_color('#2e2e38')
ax1.axhline(volume_elements_np.mean(), color='#f07ca0', linestyle='--', alpha=0.7,
            label=f'mean={volume_elements_np.mean():.2e}')
ax1.legend(fontsize=8, labelcolor='#e8e8f0', facecolor='#16161a', edgecolor='#2e2e38')

# Panel 2: Condition Number
ax2 = axes[1]
ax2.bar(range(n), condition_numbers, color=COLORS, alpha=0.85, edgecolor='#2e2e38')
ax2.set_xticks(range(n))
ax2.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=8, color='#9090a8')
ax2.set_title('Condition Number\n(Curvature Anisotropy)', color='#e8e8f0', fontsize=12, pad=10)
ax2.set_ylabel('Condition Number', color='#9090a8')
ax2.tick_params(colors='#9090a8')
ax2.spines[:].set_color('#2e2e38')
ax2.axhline(condition_numbers.mean(), color='#8b7cf8', linestyle='--', alpha=0.7,
            label=f'mean={condition_numbers.mean():.1f}')
ax2.legend(fontsize=8, labelcolor='#e8e8f0', facecolor='#16161a', edgecolor='#2e2e38')

# Panel 3: Distance Heatmap
ax3 = axes[2]
im = ax3.imshow(dist_matrix, cmap='viridis', aspect='auto')
ax3.set_xticks(range(n))
ax3.set_yticks(range(n))
ax3.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=7, color='#9090a8')
ax3.set_yticklabels(LABELS, fontsize=7, color='#9090a8')
ax3.set_title('Riemannian Distance Matrix\n(Semantic Distance in Latent Space)', color='#e8e8f0', fontsize=12, pad=10)
cbar = fig.colorbar(im, ax=ax3)
cbar.ax.tick_params(colors='#9090a8')
ax3.spines[:].set_color('#2e2e38')

plt.suptitle('Qwen2.5-Coder 14B Q4_K_M — Latent Space Riemannian Curvature\n(Nonlinear MLP Decoder + lrw PullbackMetric)',
             color='#e8e8f0', fontsize=13, y=1.02)
plt.tight_layout()

graph_path = OUTPUT_DIR / "qwen_curvature_002.png"
plt.savefig(graph_path, dpi=150, bbox_inches='tight', facecolor='#0e0e10')
plt.close()
print(f"Figure saved: {graph_path}")

print("\n" + "=" * 60)
print("Experiment Complete — Summary")
print("=" * 60)
print(f"  Most curved: {results['summary']['most_curved_label']} (cond={condition_numbers.max():.1f})")
print(f"  Least curved: {results['summary']['least_curved_label']} (cond={condition_numbers.min():.2f})")
print(f"  Mean condition number: {results['summary']['condition_number_mean']:.2f}")
print(f"\n  Output files:")
print(f"    JSON : {json_path}")
print(f"    Figure: {graph_path}")
