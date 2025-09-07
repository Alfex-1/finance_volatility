# Visualisation des prix des actions des entreprises du S&P 500 et prédiction de la volatilité des rendements quotidiens

Dans un environnement économique volatile, prédire la volatilité des entreprises devient essentiel pour les investisseurs et analystes. Ce projet propose une application intuitive permettant d'analyser et de prévoir la volatilité des prix et rendements des 500 plus grandes entreprises cotées aux États-Unis. L'application offre des estimations précises de la volatilité à venir pour un horizon de 2 à 15 jours, en utilisant le modèle GARCH(p,q). L'approche automatique sélectionne objectivement les modèles les plus adaptés, minimisant ainsi les biais humains. Développée en Python et déployée sur Streamlit, cette solution permet une exploration approfondie des données financières de manière simple et ituitive.

Veuillez trouver l'application ici : https://financevolatility.streamlit.app/

## Prérequis
Les conditions préalables pour exploiter efficacement ce projet varient selon l'utilisation que vous comptez en faire. Voici les recommandations spécifiques :

### Utilisation de l'application :

1. **Disposer d'un environnement Windows**
2. **Connexion Internet :** Une connexion internet est indispensable pour accéder à l'application.

### Utilisation des codes utilisés pour tester l'application (optionnel) :

**Installation de Python :** Pour utiliser notre algorithme, veuillez installer Python dans sa version 3.12.6.  Vous pouvez la télécharger  sur [python.org](https://www.python.org/).
   
## Structure du dépôt 

- __docs__ : La documentation et les images utilisées dans cette documentation.   
- __src__ :
    - **`\app`** : Code Python permet la cnstruction de l'application.
    - **`\tools`** : On y retrouve tous les codes Python pour la construction de l'application et la tester sans la lancer. Aussi, le code pour la comparaison de distributions symétrie et non-symétriques.
- __README.md__ : Le présent message que vous lisez actuellement.         
- __requirements.txt__ : Fichier contenant la liste de tous les modules nécessaires à l'éxecution des codes Python du projet.        

## Installation (optionnel)

1. **Clonez le dépôt GitHub sur votre machine locale:** 
```bash
git clone https://github.com/Alfex-1/finance_volatility.git
```

2. **Installez les dépendances requises:**
```bash
pip install -r requirements.txt
```

## Utilisation (optionnel)

Pour tester d'autres configurations de modèles, vous pouvez d'abord exécuter le script `Fontions.py`, qui contient toutes les importations de modules ainsi que les fonctions nécessaires au bon fonctionnement des autres scripts. Pour l'analyse visuelle, il faut se référer au script `Analyse.py`, où vous pouvez modifier les entreprises étudiées ainsi que la période. Le script `Modélisation.py` décrit le processus de sélection et d'optimisation des modèles, ainsi que la visualisation de leurs résultats. Enfin, le dernier script, skew_simulation.py, permet de créer des séries symétriques et non symétriques (ce qui est utile uniquement pour la documentation).

## L'application 

Pour utiliser l'application, vous devez d'abord choisir entre une analyse visuelle ou des prédictions, puis sélectionner les entreprises (maximum 4). 

Pour l'analyse visuelle, vous devez choisir la période souhaitée (date de début et de fin). Attention : il est probable que les dernières dates ne soient pas disponibles, notamment si vous incluez la date d'aujourd'hui, car il n'y a pas encore de données pour le jour même, mais elles seront disponibles plus tard. Une fois la période définie, il vous suffit de cliquer sur "Lancer".

Pour les prédictions, vous n'avez pas besoin de sélectionner une date de début, car celle-ci sera automatiquement fixée à un an et demi avant la date de fin choisie. Vous pouvez également choisir de visualiser les performances des modèles (attention, cela peut prendre quelques instants). Ensuite, pour les prédictions, vous pourrez spécifier l'horizon temporel, c'est-à-dire le nombre de jours sur lesquels vous souhaitez faire des prédictions, ainsi que le niveau de certitude associé. À noter : plus le niveau de certitude se rapproche de 100%, plus les intervalles de confiance seront visuellement éloignés des valeurs prédites. Enfin, il ne vous reste plus qu'à cliquer sur "Lancer".
