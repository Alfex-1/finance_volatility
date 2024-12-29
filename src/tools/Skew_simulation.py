import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Paramètres pour les graphiques
n_samples = 1000000
skew_steps = [-0.6, -0.3, 0, 0.3, 0.6]

# Fonction pour générer une distribution avec skewness
# skewnorm génère une distribution asymétrique

def generate_skewed_data(skewness, size):
    return skewnorm.rvs(a=skewness, size=size)

# Génération des données pour skewness
data_skew = {}
for skew in skew_steps:
    data_skew[f"Skewness={skew}"] = generate_skewed_data(skew, n_samples)
df_skew = pd.DataFrame(data_skew)


# Visualisation des distributions avec skewness
plt.figure(figsize=(13, 7))
for column in df_skew.columns:
    density, bins = np.histogram(df_skew[column], bins=50, density=True)
    center = (bins[:-1] + bins[1:]) / 2
    plt.plot(center, density, label=column)

plt.xlabel("Valeurs")
plt.ylabel("Densité")
plt.legend()
plt.show()