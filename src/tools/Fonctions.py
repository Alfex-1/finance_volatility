import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import mplfinance as mpf
from datetime import datetime, timedelta
from itertools import combinations
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import skew, jarque_bera, shapiro, ttest_1samp, norm
from arch import arch_model
import math
from sklearn.model_selection import ParameterGrid

def import_data(index, start_date, end_date):
    """
    Importe les données historiques d'un indice ou d'un ticker spécifique sur une plage de dates donnée.

    Args:
        index (str or list): Le symbole de l'indice ou du ticker pour lequel récupérer les données (par exemple, "AAPL" pour Apple). Peut être une liste d'indices, comme ["AAPL", "MSFT"].
        start_date (str): La date de début de la période de récupération des données au format 'YYYY-MM-DD'.
        end_date (str): La date de fin de la période de récupération des données au format 'YYYY-MM-DD'.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les données boursières, avec les colonnes Date, Open, High, Low, Close, Volume et Ticker.
        Le DataFrame est indexé par la colonne Date.
    """
    if isinstance(index, str):
        index = [index]  # Si un seul indice est fourni, le convertir en liste pour un traitement uniforme
    
    valid_indexes = []  # Liste des indices valides avec des données disponibles
    df_list = []  # Liste pour stocker les DataFrames des indices valides

    for ticker in index:
        # Téléchargement des données pour chaque ticker
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:  # Vérification si le DataFrame est vide (aucune donnée disponible)
            print(f"Aucune donnée disponible pour {ticker} entre {start_date} et {end_date}. Il sera retiré de l'analyse.")
        else:
            df = df.stack(level=1, future_stack=True).reset_index()
            df.rename(columns={"level_1": "Ticker"}, inplace=True)

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            valid_indexes.append(ticker)  # Ajouter l'indice à la liste des indices valides
            df_list.append(df)  # Ajouter le DataFrame à la liste des DataFrames valides
    
    if df_list:
        # Concatenation des DataFrames valides en un seul DataFrame
        final_df = pd.concat(df_list, axis=0)
        return final_df
    else:
        return None

def interpolate(df, start_date, end_date):
    """
    Interpole les dates manquantes pour chaque ticker dans le DataFrame, en ajoutant toutes les dates manquantes
    et en interpolant les valeurs correspondantes.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données boursières, avec des colonnes pour 'Date', 'Ticker', etc.
        start_date (str): La date de début pour l'interpolation (format 'YYYY-MM-DD').
        end_date (str): La date de fin pour l'interpolation (format 'YYYY-MM-DD').

    Returns:
        pandas.DataFrame: Un DataFrame avec les dates manquantes interpolées pour chaque ticker.
    """
    # Interpoler les dates manquantes séparemment
    df_list = []

    # Diviser le DataFrame en sous-DataFrames par Ticker et interpoler les dates manquantes
    for ticker in df['Ticker'].unique():
        # Filtrer le DataFrame pour le Ticker actuel
        df_ticker = df[df['Ticker'] == ticker].copy()
        del df_ticker['Ticker']
        
        # Réindexer pour ajouter toutes les dates manquantes (fréquence journalière)
        new_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        df_ticker = df_ticker.reindex(new_dates)
        
        # Interpoler les valeurs manquantes pour ce Ticker
        df_ticker = df_ticker.interpolate(method='quadratic', limit_direction='both')
        df_ticker['Ticker'] = ticker
        
        # Ajouter le DataFrame interpolé à la liste
        df_list.append(df_ticker)

    # Rassembler tous les DataFrames en un seul DataFrame
    df = pd.concat(df_list)

    # Si nécessaire, réinitialiser l'index ou ajuster l'index (par exemple, pour le 'Ticker' et 'Date')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    return df

def fit_model(data, p, q, o, lags, mean, dist, vol):
    """Ajuste un modèle ARCH/GARCH à une série temporelle avec des paramètres donnés.

    Cette fonction crée un modèle ARCH ou GARCH (ou ses variantes comme AR ou HAR), ajuste le modèle sur les données
    passées en argument, et retourne les valeurs de AIC et BIC pour évaluer la qualité de l'ajustement.

    Args:
        data (pd.Series): Série temporelle des données financières à ajuster.
        p (int): Ordre de l'élément ARCH du modèle (nombre de retards dans la variance conditionnelle).
        q (int): Ordre de l'élément GARCH du modèle (nombre de retards dans l'erreur au carré).
        o (int): Ordre de l'élément GJRGARCH ou autre forme (optionnel, 0 par défaut).
        lags (list or None): Liste des lags à utiliser pour le modèle AR ou HAR, ou None si non applicable.
        mean (str): Type de moyenne à utiliser dans le modèle. Peut être 'Constant', 'AR' (autoregressive), ou 'HAR' (Heterogeneous Autoregressive).
        dist (str): Distribution des résidus du modèle. Par défaut, 'normal' mais peut être 't' ou autre.
        vol (str): Type de modèle de volatilité. Par défaut, 'GARCH' mais peut aussi être 'EGARCH', 'FIGARCH', etc.

    Returns:
        tuple: Tuple contenant les paramètres p, q, lags, ainsi que les critères d'ajustement AIC et BIC.
            - p (int): Ordre de l'élément ARCH utilisé dans le modèle.
            - q (int): Ordre de l'élément GARCH utilisé dans le modèle.
            - lags (list or None): Liste des lags utilisés dans le modèle AR ou HAR.
            - AIC (float): Critère d'information d'Akaike (Akaike Information Criterion).
            - BIC (float): Critère d'information bayésien (Bayesian Information Criterion).
    """
    try:
        # Création du modèle en fonction des paramètres
        if mean == 'AR':
            model = arch_model(data.dropna(), mean=mean, dist=dist, vol=vol, p=p, q=q, o=o, lags=lags)
        elif mean == 'HAR':
            lags = [1, 5, 22]  # HAR fixe les lags
            model = arch_model(data.dropna(), mean=mean, dist=dist, vol=vol, p=p, q=q, o=o, lags=lags)
        else:
            model = arch_model(data.dropna(), mean=mean, dist=dist, vol=vol, p=p, q=q, o=o)

        # Ajuster le modèle
        model_fit = model.fit(disp='off', options={'maxiter': 750})

        return p, q, lags, model_fit.aic, model_fit.bic

    except Exception as e:
        print(f"Error in fitting model for p={p}, q={q}, lags={lags}: {e}")
        return p, q, lags, None, None  # Retourner des valeurs par défaut en cas d'erreur

def ARCH_search(data, p_max, q_max, o=0, lags_max=7, vol='GARCH', mean='Constant', dist='normal', criterion='aic'):
    """Effectue une recherche exhaustive des meilleurs paramètres pour un modèle ARCH/GARCH sur les données.

    Cette fonction crée une grille de recherche pour les paramètres du modèle ARCH/GARCH (ordre p, q, et lags),
    ajuste plusieurs modèles avec différentes combinaisons de ces paramètres, puis sélectionne le modèle avec 
    le meilleur critère d'information (AIC ou BIC).

    Args:
        data (pd.Series): Série temporelle des données à ajuster avec le modèle ARCH/GARCH.
        p_max (int): Valeur maximale de l'ordre p pour le modèle ARCH.
        q_max (int): Valeur maximale de l'ordre q pour le modèle GARCH.
        o (int, optional): Ordre de l'élément GJRGARCH ou autre forme (par défaut, 0). 
        lags_max (int, optional): Valeur maximale pour l'ordre des lags dans le modèle AR ou HAR (par défaut, 7).
        vol (str, optional): Type de modèle de volatilité, 'GARCH' par défaut mais peut aussi être 'ARCH', 'FIGARCH', etc.
        mean (str, optional): Type de moyenne à utiliser dans le modèle. Par défaut, 'Constant' mais peut être 'AR' ou 'HAR'.
        dist (str, optional): Type de distribution des résidus, 'normal' par défaut mais peut être 't' ou d'autres.
        criterion (str, optional): Critère de sélection du modèle. Par défaut, 'aic' mais peut aussi être 'bic'.

    Returns:
        tuple: Tuple contenant les meilleurs paramètres trouvés pour le modèle.
            - p (int): Ordre de l'élément ARCH (p).
            - q (int): Ordre de l'élément GARCH (q).
            - lags (list or None): Liste des lags utilisés dans le modèle AR ou HAR.
    """
    p_range = range(1, p_max + 1) if vol != 'FIGARCH' else [0, 1]
    q_range = range(1, q_max + 1) if vol != 'FIGARCH' else [0, 1]
    lags_range = range(1, lags_max + 1)

    # Définir la grille de paramètres
    param_grid = {'p': p_range, 'q': q_range, 'lags': lags_range}
    grid = ParameterGrid(param_grid)

    # Utilisation de joblib pour paralléliser le calcul
    results = Parallel(n_jobs=-1)(  # -1 utilise tous les cœurs disponibles
        delayed(fit_model)(data, params['p'], params['q'], o, params['lags'] if mean in ['AR', 'HAR'] else None, mean, dist, vol)
        for params in grid
    )

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results, columns=['p', 'q', 'lags', 'AIC', 'BIC'])

    # Trier les résultats par le critère spécifié
    results_df = results_df.sort_values(by=criterion.upper()).reset_index(drop=True)

    # Extraire les meilleurs paramètres
    best_params = results_df.iloc[0]
    p = int(best_params['p'])
    q = int(best_params['q'])
    lags = best_params['lags']

    return p, q, lags

def model_validation(model):
    """
    Valide les hypothèses relatives à un modèle GARCH sur les résidus d'une série temporelle.

    Args:
        model (arch.__future__.arch_model.ARCHModel): Le modèle GARCH ajusté à la série temporelle. 
               Ce modèle doit avoir des résidus et des paramètres accessibles après ajustement.

    Returns:
        pd.DataFrame: Une DataFrame contenant les résultats des tests de validation des hypothèses, y compris les p-values et un indicateur de respect des hypothèses. 
        Les hypothèses testées incluent la normalité des résidus, l'autocorrélation des résidus, l'autocorrélation des résidus au carré, l'effet ARCH et la stationnarité conditionnelle.
    
    Notes:
        - La stationnarité conditionnelle est vérifiée en s'assurant que la somme des coefficients alpha et beta du modèle GARCH est inférieure à 1. Aucune P-Value n'y est donc associée.
    """
    # Création d'un dictionnaire pour stocker les résultats
    results = {
        'Hypothèse': [],
        'Respect': [],
        'P-Value':[]
    }
    
    # Résidus et paramètres
    resid = model.resid
    resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
    params = pd.DataFrame(model.params)
    params = params[params.index.str.contains('alpha|beta')]
    
    # 1. Normalité des résidus (p-value > 0.05)
    _, p_shapiro = shapiro(resid)
    results['Hypothèse'].append('Normalité des résidus')
    results['Respect'].append(1 if p_shapiro >= 0.05 else 0)
    results['P-Value'].append(p_shapiro)
    
    # 2. Autocorrélation des résidus (p-value >= 0.05 pour toutes les lags)
    lb_resid = acorr_ljungbox(resid, lags=[i for i in range(1, 8)], return_df=True)
    autocorr_resid_pvalues = lb_resid['lb_pvalue']
    results['Hypothèse'].append('Autocorrélation des résidus')
    results['Respect'].append(1 if all(p >= 0.05 for p in autocorr_resid_pvalues) else 0)
    results['P-Value'].append(autocorr_resid_pvalues.tolist())
    
    # 3. Autocorrélation des résidus au carré (p-value >= 0.05 pour toutes les lags)
    lb_resid_sq = acorr_ljungbox(resid**2, lags=[i for i in range(1, 8)], return_df=True)
    autocorr_resid_sq_pvalues = lb_resid_sq['lb_pvalue']
    results['Hypothèse'].append('Autocorrélation des résidus au carré')
    results['Respect'].append(1 if all(p >= 0.05 for p in autocorr_resid_sq_pvalues) else 0)
    results['P-Value'].append(autocorr_resid_sq_pvalues.tolist())
    
    # 4. Hétéroscédasticité conditionnelle (effet ARCH), p-value >= 0.05
    lm_test = het_arch(resid)
    results['Hypothèse'].append('Effet ARCH')
    results['Respect'].append(1 if lm_test[1] >= 0.05 else 0)
    results['P-Value'].append(lm_test[1])
    
    # 5. Stationnarité conditionnelle
    results['Hypothèse'].append('Stationnarité conditionnelle')
    results['Respect'].append(1 if float(np.sum(params, axis=0).iloc[0]) < 1 else 0)
    results['P-Value'].append("None")
    
    # Création d'une DataFrame avec les résultats
    df_results = pd.DataFrame(results)
    return df_results

def distribution(resid):
    """
    Calcule la kurtosis et la skewness (asymétrie) d'une série de résidus pour évaluer la forme de la distribution.

    Args:
        resid (pd.Series ou np.ndarray): Série de résidus pour laquelle la kurtosis et la skewness doivent être calculées.

    Returns:
        float: La kurtosis de la série des résidus, indiquant l'aplatissement ou l'acuité de la distribution.
        float: La skewness (asymétrie) de la série des résidus, indiquant si la distribution est asymétrique vers la gauche ou la droite.
    """
    resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
    kurt = resid.kurtosis()
    skewness = resid.skew()
    
    return kurt, skewness

def forecast_volatility(i, real_values, test_size, vol, p, q, mean, dist, lag):
    """
    Prédit la volatilité pour une période donnée à l'aide d'un modèle GARCH.

    Cette fonction ajuste un modèle GARCH sur les données d'entraînement jusqu'à l'indice spécifié par `i`,
    et prédit la volatilité pour la période suivante en utilisant le modèle ajusté.

    Args:
        i (int): L'indice de l'instant actuel pour lequel la prévision est effectuée.
        real_values (array-like): Les données historiques de séries temporelles utilisées pour ajuster le modèle.
        test_size (int): Le nombre de points de données réservés pour les tests.
        vol (str): Le modèle de volatilité à utiliser (par exemple, 'Garch' ou 'EGarch').
        p (int): L'ordre du modèle GARCH pour le retard des rendements carrés passés.
        q (int): L'ordre du modèle GARCH pour le retard de la volatilité conditionnelle passée.
        mean (str): Le modèle de moyenne à utiliser (par exemple, 'Constant', 'Zero', etc.).
        dist (str): La distribution des erreurs du modèle (par exemple, 'Normal', 't', etc.).
        lag (int): Le nombre de décalages (lags) à inclure dans le modèle GARCH pour prendre en compte l'historique des valeurs.

    Returns:
        float: La volatilité prévisionnelle pour la période suivante, sous forme de racine carrée de la variance prédit.
    """
    current_train = real_values[:-(test_size - i)]
    model = arch_model(current_train, vol=vol, p=p, q=q, mean=mean, dist=dist, lags=lag)
    model_fit = model.fit(disp='off', options={'maxiter': 750})
    pred = model_fit.forecast(horizon=1)
    return np.sqrt(pred.variance.values[-1, :][0])

def rolling_pred(real_values, test_size, vol, p, q, mean, dist, lag, col):
    """
    Effectue des prévisions glissantes de la volatilité pour une série temporelle donnée 
    à l'aide d'un modèle ARCH/GARCH et affiche les résultats.

    Args:
        real_values (pd.Series): Série temporelle complète contenant les valeurs réelles.
        test_size (int): Taille de la période de test pour les prévisions glissantes.
        vol (str): Modèle de volatilité à utiliser ('GARCH', 'ARCH', etc.).
        p (int): Ordre du processus ARCH.
        q (int): Ordre du processus GARCH.
        mean (str): Modèle de la moyenne à utiliser ('Constant', 'Zero', etc.).
        dist (str): Distribution à utiliser pour les résidus ('normal', 't', 'skewt', etc.).
        lag (int): Nombre de décalages (lags) à utiliser pour le modèle, nécessaire si le modèle de moyenne est 'AR' ou 'HAR'.
        col (str): Nom de la colonne associée à la série temporelle.
    
    Displays:
        Un graphique des valeurs réelles de la série et des prévisions glissantes.
    """
    rolling_predictions = []
    rolling_predictions = Parallel(n_jobs=-1, verbose=0)(  # Prédictions parallèles
        delayed(forecast_volatility)(i, real_values, test_size, vol, p, q, mean, dist, lag) for i in range(test_size)
    )

    rolling_predictions_df = pd.Series(rolling_predictions, index=real_values[-test_size:].index)
    true = real_values[-test_size:]
    true = true.round(3)
    preds = rolling_predictions_df
    preds.index = true.index
    preds = preds.round(3)

    # Création du graphique interactif avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=true.index, y=true, mode='lines', name=f'Rendement réel', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=preds.index, y=preds, mode='lines', name='Volatilité prédite', line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=f'Prévisions glissantes de la volatilité des actions {col}',
        xaxis_title=None,
        yaxis_title='Volatilité et rendements (en %)',
        xaxis=dict(
            tickformat='%d-%m-%Y',
            tickangle=45
        ),
        template="seaborn",
        legend=dict(
            font=dict(size=12),
            x=0.01, y=0.01, 
            traceorder='normal', 
            orientation='h', 
            xanchor='left', 
            yanchor='bottom'
        ),
        title_font=dict(size=17),
        title_x=0.1,
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=80))
    fig.show()

def forecasting_volatility(data, model, vol, p, q, mean, dist, lag, col, horizon, conf_level=0.95):
    """
    Prédit la volatilité future d'un actif avec un intervalle de confiance dynamique.

    Args:
        data (pd.Series): Séries temporelles des rendements ou des prix historiques de l'actif.
        model (str): Type de modèle ARCH/GARCH à utiliser pour la prévision (par exemple, 'GARCH', 'EGARCH').
        vol (str): Modèle de volatilité à utiliser (par exemple, 'GARCH', 'EGARCH').
        p (int): Ordre de l'auto-régression (p) dans le modèle GARCH.
        q (int): Ordre de la moyenne mobile (q) dans le modèle GARCH.
        mean (str): Modèle de moyenne (par exemple, 'Constant', 'AR').
        dist (str): Distribution des résidus dans le modèle (par exemple, 'normal', 't').
        lag (int): Le nombre de décalages à utiliser si le modèle de moyenne est 'AR'.
        col (str): Le nom de l'actif ou de la colonne dans les données.
        horizon (int): L'horizon de prévision, en nombre de jours pour lesquels la volatilité doit être prédite.
        conf_level (float, optional): Niveau de confiance pour l'intervalle (par défaut 0.95).

    Affiche :
        Un graphique représentant la volatilité prédite avec l'intervalle de confiance pour l'horizon spécifié.
    """
    # Modélisation ARCH/GARCH
    model = arch_model(data, vol=vol, p=p, q=q, mean=mean, dist=dist, lags=lag)
    model_fit = model.fit(disp='off', options={'maxiter': 750})

    # Prévisions de la volatilité pour l'horizon donné
    pred = model_fit.forecast(horizon=horizon)
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
    predicted_volatility = np.sqrt(pred.variance.values[-1, :]).round(3)
    
    # Remplacement des valeurs négatives dans la variance par 0
    variance_values = np.clip(pred.variance.values[-1, :], 0, None)

    # Calcul du seuil de l'intervalle de confiance
    z_score = round(norm.ppf((1 + conf_level) / 2),3)
    conf_int_lower = np.sqrt(np.maximum(variance_values - z_score * np.sqrt(variance_values), 0)).round(3)
    conf_int_upper = np.sqrt(pred.variance.values[-1, :] + z_score * np.sqrt(pred.variance.values[-1, :])).round(3)

    # Création du graphique interactif avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_dates, y=conf_int_upper, mode='lines', name=f'Limite supérieure ({int(conf_level*100)}%)', line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=predicted_volatility, mode='lines', name=f'Volatilité prédite', line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=conf_int_lower, mode='lines', name=f'Limite inférieure ({int(conf_level*100)}%)', line=dict(color='yellow', dash='dash')
    ))
    fig.update_layout(
        legend=dict(traceorder='normal'),
        title=f'Prédiction de volatilité des actions {col} pour les {horizon} prochains jours',
        xaxis_title=None,
        yaxis_title='Volatilité prédite (en %)',
        xaxis=dict(
            tickformat='%d-%m-%Y', 
            tickangle=45
        ),
        yaxis=dict(range=[conf_int_lower.min(), conf_int_upper.max()],
                   autorange=False),
        template="seaborn",
        title_font=dict(size=17),
        title_x=0.1,
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=80))
    fig.show()

def mean_dist(hyp_df, data, kurtosis, skewness):
    """
    Détermine la spécification de la moyenne et de la distribution d'un modèle basé sur les hypothèses
    statistiques et les caractéristiques de la série temporelle d'entraînement.

    Args:
        hyp_df (pd.DataFrame): DataFrame contenant les résultats des tests d'hypothèses pour les résidus,
                                notamment la vérification de l'autocorrélation des résidus, leur carré,
                                et la normalité des résidus. Ce DataFrame doit avoir les colonnes suivantes :
                                - 'Hypothèse' : nom des hypothèses (ex. 'Autocorrélation des résidus')
                                - 'Respect' : résultats des tests (1 pour respecté, 0 pour non respecté)
                                - 'P-Value' : p-value des tests (si applicable).
        data (array-like): Série temporelle d'entraînement utilisée pour le test de la moyenne. 
                            Un test de moyenne nulle (t-test) est effectué pour déterminer si la moyenne 
                            est significativement différente de zéro.
        kurtosis (float): La kurtose des résidus de la série temporelle, mesurant l'aplatissement de la distribution.
        skewness (float): L'asymétrie des résidus de la série temporelle, mesurant la déviation de la distribution par rapport à la symétrie.

    Returns:
        mean (str): La spécification de la moyenne choisie pour le modèle. Peut être l'une des options suivantes :
            - 'Zero' si la moyenne est insignifiquement différente de zéro.
            - 'AR' si une autocorrélation des résidus est observée.
            - 'HAR' si une autocorrélation des résidus au carré est détectée en plus de l'autocorrélation des résidus.
            - 'Constant' si aucune des conditions précédentes n'est remplie.
        dist (str): La distribution des résidus du modèle choisie en fonction de la kurtose et de l'asymétrie :
            - 'ged' si la kurtose est proche de 0 et l'asymétrie est faible.
            - 't' si la kurtose est significativement différente de 0 et l'asymétrie est faible.
            - 'skewt' l'asymétrie est significative, peu importe la kurtose.
            - 'normal' si la p-value du test est inférieure à 10%.
    """
    # Détermination de la moyenne
    _, p_value_ttest = ttest_1samp(data, popmean=0)
    autocorr_resid = hyp_df.loc[hyp_df['Hypothèse'] == 'Autocorrélation des résidus', 'Respect'].values[0]
    autocorr_resid_squared = hyp_df.loc[hyp_df['Hypothèse'] == 'Autocorrélation des résidus au carré', 'Respect'].values[0]
    pvalue_normal = hyp_df.loc[hyp_df['Hypothèse'] == 'Normalité des résidus', 'P-Value'].values[0]
    kurtosis, skewness = round(kurtosis,3), round(skewness,3)
    
    if autocorr_resid == 0:  # Autocorrélation des résidus présente
        if autocorr_resid_squared == 1:
            mean = 'AR'
        elif autocorr_resid_squared == 0:
            mean = 'HAR'
    else:  # Pas d'autocorrélation des résidus
        if p_value_ttest >= 0.05:  # Test de moyenne nulle
            mean = 'Zero'
        else:
            mean = 'Constant'

    # Détermination de la distribution
    if pvalue_normal > 0.01:
        dist='normal'
    else:
        if (kurtosis >= 1.1 or kurtosis <= -0.6) and abs(skewness) >= 0.3:
            dist='skewt'
        elif (kurtosis >= 1.1 or kurtosis <= -0.6) and abs(skewness) < 0.3:
            dist = 't'
        elif (kurtosis <= 1.1 and kurtosis >= -0.6) and abs(skewness) >= 0.3:
            dist = 'skewt'
        else:
            dist='ged'

    return str(mean), str(dist)