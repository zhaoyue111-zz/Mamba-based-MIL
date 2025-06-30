import numpy as np
import matplotlib.pyplot as plt

w = np.linspace(-5, 5, 1000)
loss_no_reg = 0.5 * (w - 3)**2
l1_penalty = lambda l: l * np.abs(w)

plt.figure(figsize=(10, 6))

for l in [0.0, 0.5, 1.0, 2.0]:
    total_loss = loss_no_reg + l1_penalty(l)
    plt.plot(w, total_loss, label=f'λ = {l}')

plt.title("L1 正则化对 Loss 的影响")
plt.xlabel("w")
plt.ylabel("Total Loss")
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True)
plt.show()
