from HCAE import HCAE
from simulate_data import simulate_data, simulate_labels

subjects = 77
view = -1
labels = simulate_labels(38, 77)
samples = []
for i in range(4):
    samples.append(simulate_data(35, subjects))

HCAE(samples, labels, view)
