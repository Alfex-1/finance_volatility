from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_1samp, norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sklearn.model_selection import ParameterGrid
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import requests
from joblib import Parallel, delayed
from io import StringIO

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
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, repair=True, ignore_tz=True, rounding=False, session=None, auto_adjust=True)
        
        if df.empty:  # Vérification si le DataFrame est vide (aucune donnée disponible)
            st.warning(f"Aucune donnée disponible pour {ticker} entre {start_date} et {end_date}. Il sera retiré de l'analyse.")
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
        df_ticker = df_ticker.drop(columns=['Ticker', 'Repaired?'])
        
        # Réindexer pour ajouter toutes les dates manquantes (fréquence journalière)
        new_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        df_ticker = df_ticker.reindex(new_dates)
        
        # Interpoler les valeurs manquantes pour ce Ticker
        df_ticker = df_ticker.interpolate(method='time')
        df_ticker['Ticker'] = ticker
        
        # Ajouter le DataFrame interpolé à la liste
        df_list.append(df_ticker)

    # Rassembler tous les DataFrames en un seul DataFrame
    df = pd.concat(df_list)

    # Si nécessaire, réinitialiser l'index ou ajuster l'index (par exemple, pour le 'Ticker' et 'Date')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    return df

def visualize_correlation(df, object_viz="prix de clôture"):
    tickers = df.columns.tolist()

    if len(tickers) > 1:

        n = len(tickers)

        # ========================================================
        # 1. MATRICE DE CORRÉLATION PEARSON
        # ========================================================

        correlation_matrix = df.corr(
            method="pearson"
        ) * 100

        # Masque : on cache la diagonale + triangle supérieur
        mask_upper = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

        fig_heatmap, ax_heatmap = plt.subplots(
            figsize=(10, 8)
        )

        sns.heatmap(
            correlation_matrix,
            mask=mask_upper,
            annot=True,
            fmt=".2f",
            cmap="flare",
            cbar=False,
            ax=ax_heatmap,
            square=True
        )

        ax_heatmap.set_title(
            f"Corrélations entre les {object_viz} "
            f"des entreprises (en %)",
            fontsize=16
        )

        ax_heatmap.set_xlabel(None)
        ax_heatmap.set_ylabel(None)

        st.pyplot(fig_heatmap)


        # ========================================================
        # 2. MATRICE DES REGPLOTS
        # ========================================================

        fig_reg, axes = plt.subplots(
            nrows=n,
            ncols=n,
            figsize=(4 * n, 4 * n)
        )

        # Cas particulier : un seul ticker
        if n == 1:
            axes = np.array([[axes]])

        for i in range(n):
            for j in range(n):

                ax = axes[i, j]

                # ------------------------------------------------
                # Triangle supérieur + diagonale
                # ------------------------------------------------

                if i < j:
                    ax.set_visible(False)
                    continue

                elif i == j:
                    ax.text(
                        0.5,
                        0.5,
                        f"{tickers[i]}\n100 %",
                        ha="center",
                        va="center",
                        fontsize=14,
                        transform=ax.transAxes
                    )

                    ax.set_xticks([])
                    ax.set_yticks([])

                    continue

                # ------------------------------------------------
                # Triangle inférieur
                # ------------------------------------------------

                ticker_x = tickers[j]
                ticker_y = tickers[i]

                x = df[ticker_x]
                y = df[ticker_y]

                # Supprimer les NaN pour cette paire
                pair_data = pd.concat(
                    [x, y],
                    axis=1
                ).dropna()

                x_clean = pair_data.iloc[:, 0]
                y_clean = pair_data.iloc[:, 1]

                sns.regplot(
                    x=x_clean,
                    y=y_clean,
                    ax=ax,
                    scatter_kws={"s": 15},
                    line_kws={"color": "red"},
                    order=1,
                    robust=True,
                    ci=None
                )

                ax.set_xlabel(
                    ticker_x,
                    fontsize=11
                )

                ax.set_ylabel(
                    ticker_y,
                    fontsize=11
                )

                ax.tick_params(
                    axis="both",
                    labelsize=8
                )

        fig_reg.suptitle(
            f"Relation entre les {object_viz} "
            "des entreprises",
            fontsize=18
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.97]
        )

        st.pyplot(fig_reg)

def downside_deviation(returns, threshold=0):
    """Volatilité calculée uniquement sur les rendements négatifs (sous le seuil)."""
    downside_returns = returns[returns < threshold]
    return downside_returns.std() if len(downside_returns) > 0 else 0

def upside_deviation(returns, threshold=0):
    """Volatilité calculée uniquement sur les rendements positifs (au-dessus du seuil)."""
    upside_returns = returns[returns > threshold]
    return upside_returns.std() if len(upside_returns) > 0 else 0

def fit_garch_model(data, p, q, o, lags, mean, dist, vol):
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
            model = arch_model(data.dropna(), mean=mean, dist=dist, vol=vol, p=p, q=q, o=o, lags=[1, 5, 22])
        else:
            model = arch_model(data.dropna(), mean=mean, dist=dist, vol=vol, p=p, q=q, o=o)
        
        # Ajuster le modèle
        model_fit = model.fit(disp='off', options={'maxiter': 2000})
        params = model_fit.params
        alpha_beta = params[params.index.str.contains('alpha|beta')]
        sum_params = np.round(float(alpha_beta.sum()), 4)

        return model_fit, sum_params, p, q, lags, model_fit.aic, model_fit.bic

    except Exception as e:
        print(f"Error in fitting model for p={p}, q={q}, lags={lags}: {e}")
        return None, None, p, q, lags, None, None  # Retourner des valeurs par défaut en cas d'erreur

def ARCH_search(data, p_max, q_max, lags, o=0, vol='GARCH', mean='Constant', dist='normal', criterion='aic'):
    """Effectue une recherche exhaustive des meilleurs paramètres pour un modèle ARCH/GARCH sur les données.

    Cette fonction crée une grille de recherche pour les paramètres du modèle ARCH/GARCH (ordre p, q, et lags),
    ajuste plusieurs modèles avec différentes combinaisons de ces paramètres, puis sélectionne le modèle avec 
    le meilleur critère d'information (AIC ou BIC).

    Args:
        data (pd.Series): Série temporelle des données à ajuster avec le modèle ARCH/GARCH.
        p_max (int): Valeur maximale de l'ordre p pour le modèle ARCH.
        q_max (int): Valeur maximale de l'ordre q pour le modèle GARCH.
        lags (int or list): Lags du modèle AR ou HAR.
        o (int, optional): Ordre de l'élément GJRGARCH ou autre forme (par défaut, 0).
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

    # Définir la grille de paramètres
    param_grid = {'p': p_range, 'q': q_range}
    grid = ParameterGrid(param_grid)

    # Utilisation de joblib pour paralléliser le calcul
    results = Parallel(n_jobs=-1)(
        delayed(fit_garch_model)(data, params['p'], params['q'], o, lags if mean in ['AR', 'HAR'] else None, mean, dist, vol)
        for params in grid
    )
       
    # Enlever le model fit des résultats pour ne garder que les paramètres et les critères
    results = [result[1:] for result in results]
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results, columns=['sum_params', 'p', 'q', 'lags', 'AIC', 'BIC'])

    # Écarter les ajustements ayant échoué (sum_params, AIC ou BIC manquants)
    results_df = results_df.dropna(subset=['sum_params', 'AIC', 'BIC'])

    # Garder uniquement les modèles stationnaires (somme des paramètres < 1), sauf pour FIGARCH
    if vol != 'FIGARCH':
        stationary_df = results_df[results_df['sum_params'] < 1]

        if not stationary_df.empty:
            results_df = stationary_df
        else :
            st.warning("Impossibile de trouver un modèle pertinent")

    # Trier les résultats par le critère spécifié
    results_df = results_df.sort_values(by=criterion.upper())

    # Extraire les meilleurs paramètres
    best_params = results_df.iloc[0]
    p = int(best_params['p'])
    q = int(best_params['q'])

    return p, q

def model_validation(model):
    """
    Valide les hypothèses relatives à un modèle GARCH sur les résidus d'une série temporelle.

    Args:
        model (arch.__future__.arch_model.ARCHModel): Le modèle GARCH ajusté à la série temporelle. 
               Ce modèle doit avoir des résidus et des paramètres accessibles après ajustement.

    Returns:
        pd.DataFrame: Une DataFrame contenant les résultats des tests de validation des hypothèses, y compris les p-values et un indicateur de respect des hypothèses. 
        Les hypothèses testées incluent la normalité des résidus, l'autocorrélation des résidus, l'autocorrélation des résidus au carré, l'effet ARCH et la stationnarité conditionnelle.
        Un attribut suggested_ar_lags est également retourné, indiquant les lags AR nécessaires
        pour absorber l'autocorrélation des résidus (utile si l'hypothèse 'Autocorrélation des
        résidus' est violée).
    
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
    sum_params = float(np.sum(params, axis=0).iloc[0])
    
    # 1. Normalité des résidus (p-value > 0.05)
    _, p_shapiro = shapiro(resid)
    results['Hypothèse'].append('Normalité des résidus')
    results['Respect'].append(1 if p_shapiro >= 0.05 else 0)
    results['P-Value'].append(p_shapiro)
    
    # 2. Autocorrélation des résidus (p-value >= 0.05 pour toutes les lags)
    lb_resid = acorr_ljungbox(resid, lags=[i for i in range(1, 13)], return_df=True)
    autocorr_resid_pvalues = lb_resid['lb_pvalue']
    results['Hypothèse'].append('Autocorrélation des résidus')
    results['Respect'].append(1 if all(p >= 0.05 for p in autocorr_resid_pvalues) else 0)
    results['P-Value'].append(autocorr_resid_pvalues.tolist())

    # Lags AR suggérés en cas d'autocorrélation détectée
    # On prend tous les lags où le test Ljung-Box est significatif
    # (p < 0.05), jusqu'au dernier lag concerné
    significant_lags = [
        lag for lag, p in zip(range(1, 13), autocorr_resid_pvalues)
        if p < 0.05
    ]
    suggested_ar_lags = significant_lags if significant_lags else [1]
    
    # 3. Autocorrélation des résidus au carré (p-value >= 0.05 pour toutes les lags)
    lb_resid_sq = acorr_ljungbox(resid**2, lags=[i for i in range(1, 13)], return_df=True)
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
    persistence = np.round(sum_params, 4)
    results['Hypothèse'].append('Stationnarité conditionnelle')
    results['Respect'].append(1 if persistence < 1 else 0)
    results['P-Value'].append(persistence)
    
    # Création d'une DataFrame avec les résultats
    df_results = pd.DataFrame(results)
    df_results.attrs['suggested_ar_lags'] = suggested_ar_lags

    return df_results

def hypothesis_status(df_validation, hypothesis):
    """
    Retourne Oui / Non en fonction du résultat
    du test correspondant.
    """
    values = df_validation.loc[df_validation["Hypothèse"] == hypothesis, "Respect"]

    if len(values) == 0:
        return "N/A"

    return "Oui" if values.iloc[0] == 1 else "Non"

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
    model_fit = model.fit(disp='off', options={'maxiter': 2000})
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
    st.plotly_chart(fig)

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
    model_fit = model.fit(disp='off', options={'maxiter': 2000})

    # Prévisions de la volatilité pour l'horizon donné
    pred = model_fit.forecast(horizon=horizon)
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
    predicted_volatility = np.sqrt(pred.variance.values[-1, :]).round(3)

    # Remplacement des valeurs négatives dans la variance par 0
    variance_values = np.clip(pred.variance.values[-1, :], 0, None)

    # Calcul du seuil de l'intervalle de confiance
    z_score = round(norm.ppf((1 + conf_level) / 2),3)
    conf_int_lower = np.sqrt(np.maximum(variance_values - z_score * np.sqrt(variance_values), 0)).round(3)
    conf_int_upper = np.sqrt(variance_values + z_score * np.sqrt(variance_values)).round(3)

    # Création du graphique interactif avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_volatility, mode='lines', name='Volatilité Prédite', line=dict(color='red')))
    
    # Remplissage pour la région de confiance
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],  # Concatenate future dates with reversed ones
        y=np.concatenate([conf_int_upper, conf_int_lower[::-1]]),  # Upper and lower bounds
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.3)',
        line=dict(color='rgba(0, 0, 0, 0)'),
        name=f'Région de confiance au niveau {int(conf_level*100)}%'))   
    
    # Personnalisation du graphique
    fig.update_layout(
        legend=dict(traceorder='normal'),
        title=f"Prévision de la Volatilité pour {col}",
        xaxis_title=None,
        yaxis_title="Volatilité prédite (en %)",
        yaxis=dict(range=[conf_int_lower.min()+0.1, conf_int_upper.max()+0.1],
                   autorange=False),
        template="seaborn",
        title_font=dict(size=17),
        title_x=0.1,
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=80))
    
    st.plotly_chart(fig, width='stretch')
    
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
    if pvalue_normal > 0.1:
        dist='normal'
    else:
        if abs(skewness) >= 0.3:
            dist='skewt'
        elif (kurtosis >= 1.1 or kurtosis <= -0.6) and abs(skewness) < 0.3:
            dist = 't'
        else:
            dist='ged'

    return str(mean), str(dist)

st.title("Analyse des prix et des rendements des actions de plusieurs entreprises et prédiction des risques associés")
st.write(
    ("Bienvenue sur l'application ! Vous pouvez y visualiser le prix des actions des entreprises du S&P 500 et du CAC40 ainsi que leurs rendements quotidiens. "
     "Vous avez également la possibilité de consulter les prédictions des risques (volatilité) associés aux investissements dans les actions de ces entreprises, à court terme.")
)

# Lien pour voir la documentation
st.link_button("Voir la documentation", "https://github.com/Alfex-1/finance_volatility/blob/main/docs/Documentation.pdf")

# Case à cocher pour "Analyse" et "Prédiction"
st.sidebar.title("Paramètres")
option = st.sidebar.radio(
    "Choisissez le type d'étude que vous voulez mener",
    ["Analyse", "Prédiction"]
)

# Entreprises
url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
url_cac40 = "https://en.wikipedia.org/wiki/CAC_40"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/139.0.0.0 Safari/537.36"
}

try:
    response = requests.get(url_sp500, headers=headers, timeout=60)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    sp500_df = tables[0]
    tickers_sp500 = sp500_df[['Symbol', 'Security']]
except (requests.RequestException, ValueError, IndexError) as e:
    st.error("Impossible de récupérer la liste des entreprises du S&P 500. Veuillez réessayer plus tard.")
    st.stop()
    
try:
    response = requests.get(url_cac40, headers=headers, timeout=60)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    cac40_df = tables[4]
    tickers_cac40 = cac40_df[['Ticker', 'Company']]
    tickers_cac40['Ticker'] = tickers_cac40['Ticker'].str.split('.').str[0]
except (requests.RequestException, ValueError, IndexError) as e:
    st.error("Impossible de récupérer la liste des entreprises du CAC 40. Veuillez réessayer plus tard.")
    st.stop()
    
# Rassembler les données
all_tickers = pd.concat([tickers_sp500.rename(columns={'Symbol': 'Ticker', 'Security': 'Company'}), tickers_cac40], ignore_index=True)

ticker_to_name = dict(zip(all_tickers['Ticker'], all_tickers['Company']))
selected_companies = st.sidebar.multiselect("Choisissez les entreprises à analyser", 
                                    all_tickers['Company'].tolist(),
                                    max_selections=4)

start_date = None
end_date = None
visu_perf = None
launch = False

if option == "Analyse" and len(selected_companies) >=1:
    # Importation des données
    today = datetime.today()
    default_end_date = today

    # Définir la date de début par défaut comme étant 1 an avant la date de fin
    default_start_date = default_end_date - timedelta(days=365)  # 1 an avant la date de fin

    # Utiliser Streamlit pour afficher les dates avec les valeurs par défaut
    start_date = st.sidebar.date_input("Sélectionnez la date à partir de laquelle les analyses débuteront", value=default_start_date.date())
    end_date = st.sidebar.date_input("Sélectionnez la date à partir de laquelle les analyses se finiront", value=default_end_date.date())    

    selected_tickers = all_tickers[all_tickers['Company'].isin(selected_companies)]['Ticker'].tolist()
    
    df = import_data(selected_tickers, start_date, end_date)
    
    if df is None:
        st.sidebar.warning("Aucune données disponibles n'ont pu être trouvé pour cette période et pour ces entreprises.")
    else:
        # Interpolation
        df = interpolate(df, start_date=start_date, end_date=end_date).dropna()
        
        # Prévenir des dates manquantes
        missing_days = (pd.to_datetime(end_date) - pd.to_datetime(df['Date'].max())).days
        if missing_days > 1:
            st.sidebar.warning(f"Attention : les données ne sont pas disponibles pour les {missing_days} derniers jours.")
        elif missing_days == 1:
            st.sidebar.warning(f"Attention : les données ne sont pas disponibles pour le dernier jour.")

        # Donner les vrais noms
        df['Ticker'] = df['Ticker'].map(ticker_to_name)
        
        # Lancer l'application
        launch = st.sidebar.button("Lancer")
    
elif option == "Prédiction" and len(selected_companies) >=1:   
    # Importation des données
    end_date = st.sidebar.date_input("Sélectionnez la date à partir de laquelle les prédictions débuteront", value=pd.to_datetime("today"))
    start_date = end_date - pd.Timedelta(days=365 + 31 * 6)
    
    # Choisir de visualiser les performances sur la base de test
    visu_perf = st.sidebar.toggle("Visualisation des performances de chaque modèle par rapport aux données rélles")
    if visu_perf:
        st.sidebar.warning("Attention : l'évaluation de chaque modèle peut prendre du temps")
    
    # Choisir l'intervalle de confiance des prédictions
    conf_int = st.sidebar.slider("Choisissez le degré de certitude des prédictions (en %).", min_value=80, max_value=99, value=95)
    conf_int = conf_int/100
    
    # Choisir l'horizon des prédictions
    horizon = st.sidebar.slider("Choisissez l'horizon des prédictions (en jours)", min_value=2, max_value=15, value=7)    

    selected_tickers = all_tickers[all_tickers['Company'].isin(selected_companies)]['Ticker'].tolist()
    df = import_data(selected_tickers, start_date, end_date)
    
    if df is None:
        st.sidebar.warning("Aucune donnée disponible n'a pu être trouvée pour cette période et pour ces entreprises.")
    else:
        # Interpolation
        df = interpolate(df, start_date=start_date, end_date=end_date).dropna()
        
        # Prévenir des dates manquantes
        missing_days = (pd.to_datetime(end_date) - pd.to_datetime(df['Date'].max())).days
        if missing_days > 1:
            st.sidebar.warning(f"Attention : les données ne sont pas disponibles pour les {missing_days} derniers jours. Ils seront alors compris dans l'horizon temporel que vous sélectionnerez.")
        elif missing_days == 1:
            st.sidebar.warning(f"Attention : les données ne sont pas disponibles pour le dernier jour. Il sera alors compris dans l'horizon temporel que vous sélectionnerez.")
        
        # Donner les vrais noms
        df['Ticker'] = df['Ticker'].map(ticker_to_name)
        
        # Lancer l'application
        launch = st.sidebar.button("Lancer")

elif len(selected_companies) == 0:
    st.sidebar.info("Veuillez sélectionner au moins une entreprise à analyser.")

if option == "Analyse" and len(selected_companies) >= 1 and start_date and end_date and df is not None and launch:
    
    # Visualisation des prix des actions
    fig = go.Figure()

    for ticker in df["Ticker"].unique():
        df_ticker = df[df["Ticker"] == ticker]

        fig.add_trace(
            go.Scatter(
                x=df_ticker["Date"],
                y=df_ticker["Close"],
                mode="lines",
                name=ticker,
                hovertemplate=(
                    "Ticker: %{text}<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Close: %{y:.2f}<extra></extra>"
                ),
                text=[ticker] * len(df_ticker)
            )
        )

    fig.update_layout(
        title="Évolution des prix de clôture par entreprise",
        xaxis_title=None,
        yaxis_title="Prix de clôture (en USD)",
        legend_title="Entreprises",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, width='stretch')

    # Calculer les rendements quotidiens et cumulés
    df_list = []
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker].copy()
        ticker_data["Returns"] = ticker_data["Close"].pct_change(fill_method=None)
        ticker_data["Cumul_returns"] = (1 + ticker_data["Returns"]).cumprod() - 1
        
        df_list.append(ticker_data)

    # Combiner toutes les DataFrames
    df_returns  = pd.concat(df_list, ignore_index=True).dropna()

    # Visualiser l'évolution des rendements quotidiens et cumulés
    df_returns["Cumul_returns"] *= 100
    df_returns["Returns"] *= 100

    fig = go.Figure()

    for ticker in df_returns["Ticker"].unique():
        data = df_returns[df_returns["Ticker"] == ticker]

        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Returns"],
                mode="lines",
                name=ticker,
                hovertemplate=(
                    "Ticker: %{text}<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Return: %{y:.2f}%<extra></extra>"
                ),
                text=[ticker] * len(data)
            )
        )

    fig.update_layout(
        title="Rendements journaliers par entreprise",
        xaxis_title=None,
        yaxis_title="Rendements journaliers (%)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Entreprises"
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width='stretch')


    fig = go.Figure()

    for ticker in df_returns["Ticker"].unique():
        data = df_returns[df_returns["Ticker"] == ticker]

        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Cumul_returns"],
                mode="lines",
                name=ticker,
                hovertemplate=(
                    "Ticker: %{text}<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Cumul: %{y:.2f}%<extra></extra>"
                ),
                text=[ticker] * len(data)
            )
        )

    fig.update_layout(
        title="Évolution des rendements cumulés par entreprise",
        xaxis_title=None,
        yaxis_title="Rendements cumulés (%)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Entreprises"
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width='stretch')

    # Comparaison des rendements moyennes et de leur volatilité (risques)
    mean_list = []
    std_list = []

    for ticker in df_returns['Ticker'].unique():
        ticker_data = df_returns[df_returns['Ticker'] == ticker].copy()
        mean = ticker_data['Returns'].mean()
        std = ticker_data['Returns'].std()
        
        mean_list.append(mean)
        std_list.append(std)

    # Créer la DataFrame des performances et des risques
        tickers = df_returns['Ticker'].unique()
        mean_list = []
        downside_list = []
        upside_list = []

        for ticker in tickers:
            returns = df_returns[df_returns['Ticker'] == ticker]['Returns']
            mean_list.append(returns.mean())
            downside_list.append(downside_deviation(returns))
            upside_list.append(upside_deviation(returns))
    
    df_perf = pd.DataFrame({
        "Ticker": df['Ticker'].unique(),
        "Moyenne": mean_list,
        "Ecart-type": std_list,
        "Risque négatif": downside_list,
        "Risque positif": upside_list
    })

    # Visualiser les risques et les performances
    df_perf = df_perf.sort_values(by="Moyenne", ascending=False)
    ticker_order = df_perf["Ticker"].tolist()
    
    df_perf_melted = pd.melt(
        df_perf,
        id_vars=["Ticker"],
        value_vars=["Moyenne", "Ecart-type", "Risque négatif", "Risque positif"],
        var_name="Mesure",
        value_name="Valeur"
    )
    
    
    df_perf_melted["Mesure"] = df_perf_melted["Mesure"].replace({
        "Moyenne": "Rendements",
        "Ecart-type": "Volatilité",
        "Risque négatif": "Volatilité baissière",
        "Risque positif": "Volatilité haussière"
    })

    couleurs = {
        "Rendements": "#4C78A8",
        "Volatilité": "#B0B0B0",
        "Volatilité baissière": "#E45756",
        "Volatilité haussière": "#54A24B"
    }

    fig = go.Figure()

    for mesure in couleurs.keys():
        df_m = df_perf_melted[df_perf_melted["Mesure"] == mesure].set_index("Ticker").loc[ticker_order].reset_index()

        fig.add_trace(
            go.Bar(
                x=df_m["Ticker"],
                y=df_m["Valeur"],
                name=mesure,
                marker_color=couleurs[mesure],
                text=df_m["Valeur"].round(2),
                textposition="outside"
            )
        )

    fig.update_layout(
        title="Rendements et volatilité moyens des actions par entreprise",
        xaxis_title=None,
        yaxis_title="Mesures en %",
        barmode="group",
        template="plotly_white",
        legend_title="Mesure",
        xaxis=dict(categoryorder="array", categoryarray=ticker_order)
    )

    st.plotly_chart(fig, width='stretch')
        
    # Pour chaque ticker, tracer un graphique en chandelle interactif
    data = df.copy()
    for ticker in df['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].copy()

        # Assurer que la date est bien au bon format
        ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
        
        # Créer un graphique en chandelier avec Plotly
        fig = go.Figure(data=[
            go.Candlestick(
                x=ticker_data['Date'],
                open=ticker_data['Open'],
                high=ticker_data['High'],
                low=ticker_data['Low'],
                close=ticker_data['Close'],
                name=ticker
            )
        ])
        # Ajouter un volume en barre
        fig.add_trace(
            go.Bar(
                x=ticker_data['Date'],
                y=ticker_data['Volume'],
                name="Volume",
                marker_color='rgba(158,202,225,0.8)',
                opacity=0.6,
                yaxis="y2"
            )
        )
        # Mettre en forme l'affichage
        fig.update_layout(
            title=f"Performances journalières des actions {ticker}",
            xaxis_title="Date",
            yaxis_title="Prix",
            yaxis=dict(title="Prix", side="right"),
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="left",
                showgrid=False,
            ),
            xaxis_rangeslider_visible=True,
            template="plotly_dark",
            height=800,
            width=1200
        )
        
        # Afficher le graphique
        st.plotly_chart(fig)

    # Vérification de l'existance de corrélations pour les prix de clôture
    df_pivot_close = df_returns.pivot(index="Date", columns="Ticker", values="Close")
    visualize_correlation(df_pivot_close, object_viz="prix de clôture") 
    
    # Vérification de l'existance de corrélations pour les rendements
    df_pivot_returns = df_returns.pivot(index="Date", columns="Ticker", values="Returns")
    visualize_correlation(df_pivot_returns, object_viz="rendements")

elif (
    option == "Prédiction"
    and len(selected_companies) >= 1
    and end_date
    and df is not None
    and visu_perf is not None
    and launch
):

    with st.spinner(
        "La recherche du modèle optimal pour chaque entreprise peut durer quelques temps. Merci de patienter !"
    ):

        # ============================================================
        # 1. CALCUL DES RENDEMENTS
        # ============================================================

        df_list = []

        for ticker in df["Ticker"].unique():
            ticker_data = df[df["Ticker"] == ticker].copy()

            ticker_data["Returns"] = ticker_data["Close"].pct_change(fill_method=None)* 100

            df_list.append(ticker_data)

        # Combiner toutes les DataFrames
        df = pd.concat(df_list, ignore_index=True).dropna()

        # Mise sous forme matricielle
        df_pivot = df.pivot(index="Date", columns="Ticker", values="Returns")

        # ============================================================
        # 2. INITIALISATION
        # ============================================================

        model_summary = []
        model_val = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(df_pivot.columns) * (5 if visu_perf else 4)
        current_step = 0

        # ============================================================
        # 3. BOUCLE SUR LES ENTREPRISES
        # ============================================================

        for col in df_pivot.columns:

            # --------------------------------------------------------
            # Séparation apprentissage / test
            # --------------------------------------------------------

            train_size = int(len(df_pivot[col]) * (2 / 3))
            test_size = len(df_pivot[col]) - train_size

            train = df_pivot[col].iloc[:train_size]

            # ========================================================
            # ÉTAPE 1 : CHOIX INITIAL DE LA MOYENNE
            # ========================================================

            current_step += 1
            progress_bar.progress(current_step / total_steps)

            status_text.text(f"Recherche du modèle initial pour {col}...")

            # Test t de moyenne nulle
            _, p_value_ttest = ttest_1samp(train, popmean=0)

            if p_value_ttest >= 0.05:
                initial_mean = "Zero"
            else:
                initial_mean = "Constant"

            # ========================================================
            # ÉTAPE 2 : RECHERCHE INITIALE DE p ET q
            # ========================================================

            p, q = ARCH_search(
                train,
                p_max=8,
                q_max=8,
                lags=None,
                vol="GARCH",
                mean=initial_mean,
                dist="normal",
                criterion="aic"
            )

            initial_dist = "normal"
            initial_lag = None

            # ========================================================
            # ÉTAPE 3 : ESTIMATION DU MODÈLE INITIAL
            # ========================================================

            model = fit_garch_model(
                train, p=p, q=q, o=0,
                vol="GARCH", mean=initial_mean,
                dist=initial_dist, lags=None)[0]

            # ========================================================
            # ÉTAPE 4 : DIAGNOSTIC DU MODÈLE INITIAL
            # ========================================================

            current_step += 1
            progress_bar.progress(current_step / total_steps)

            status_text.text(f"Validation du modèle initial pour {col}...")

            # Résidus
            resid = model.resid

            # Tests des hypothèses
            df_val_initial = model_validation(model)

            # Distribution des résidus
            kurt_val, skewness_val = distribution(resid)

            # Vérification globale
            hypotheses_ok = all(df_val_initial["Respect"])

            # ========================================================
            # CAS 1 :
            # LE MODÈLE INITIAL EST SATISFAISANT
            # ========================================================

            if hypotheses_ok:

                final_mean = initial_mean
                final_dist = initial_dist
                final_lag = initial_lag

                df_val_final = df_val_initial

            # ========================================================
            # CAS 2 :
            # AU MOINS UNE HYPOTHÈSE EST VIOLÉE
            # ========================================================

            else:
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                status_text.text(f"Correction de la spécification pour {col}...")

                # ----------------------------------------------------
                # Détermination de la nouvelle moyenne et distribution
                # ----------------------------------------------------

                suggested_mean, suggested_dist = mean_dist(df_val_initial, train, kurt_val, skewness_val)

                # ----------------------------------------------------
                # Modification de la moyenne et/ou de la distribution ?
                # ----------------------------------------------------

                mean_changed = suggested_mean != initial_mean
                dist_changed = suggested_dist != initial_dist

                # ====================================================
                # CAS 2A :
                # UNIQUEMENT LA DISTRIBUTION CHANGE
                # ====================================================

                if dist_changed and not mean_changed:

                    # On conserve p et q. On modifie uniquement la distribution des innovations
                    final_mean = initial_mean
                    final_dist = suggested_dist
                    final_lag = None

                    model = fit_garch_model(train, p=p, q=q, o=0, vol="GARCH", mean=final_mean, dist=final_dist, lags=None)[0]
                    df_val_final = model_validation(model)

                # ====================================================
                # CAS 2B :
                # LA MOYENNE CHANGE (on passe à AR ou HAR)
                # Dans ce cas, on recherche à nouveau p et q.
                # ====================================================

                elif mean_changed:

                    final_mean = suggested_mean
                    final_dist = suggested_dist

                    # -----------------------------------------------
                    # AR
                    # -----------------------------------------------

                    if final_mean == "AR":
                        
                        final_lag = df_val_initial.attrs.get('suggested_ar_lags', [1])
                        
                        p, q = ARCH_search(train, p_max=8, q_max=8,
                            lags=final_lag, vol="GARCH",
                            mean=final_mean, dist=final_dist,
                            criterion="aic")

                    # -----------------------------------------------
                    # HAR
                    # -----------------------------------------------

                    elif final_mean == "HAR":
                        
                        final_lag = [1, 5, 22]

                        p, q = ARCH_search(train, p_max=8, q_max=8,
                            lags=final_lag, vol="GARCH",
                            mean=final_mean, dist=final_dist,
                            criterion="aic")

                    # -----------------------------------------------
                    # Constant / Zero
                    # -----------------------------------------------

                    else:

                        final_lag = None

                        p, q = ARCH_search(train,
                            p_max=8, q_max=8,
                            lags=final_lag, vol="GARCH",
                            mean=final_mean, dist=final_dist,
                            criterion="aic")

                    # Estimation du nouveau modèle
                    model = fit_garch_model(
                        train, p=p, q=q, o=0,
                        vol="GARCH", mean=final_mean,
                        dist=final_dist, lags=final_lag)[0]

                    # Validation finale
                    df_val_final = model_validation(model)

                # ====================================================
                # CAS 2C :
                # UNE AUTRE HYPOTHÈSE EST VIOLÉE
                # ====================================================

                else:
                    # On conserve la structure obtenue
                    final_mean = suggested_mean
                    final_dist = suggested_dist
                    final_lag = None

                    model = fit_garch_model(
                        train, p=p, q=q, o=0,
                        vol="GARCH", mean=final_mean,
                        dist=final_dist, lags=final_lag)[0]

                    df_val_final = model_validation(model)

            # ========================================================
            # 5. PRÉVISIONS GLISSANTES
            # ========================================================

            if visu_perf:

                current_step += 1
                progress_bar.progress(current_step / total_steps)
                status_text.text(f"Prévisions glissantes pour {col}...")

                rolling_pred(
                    real_values=df_pivot[col],
                    test_size=test_size,
                    vol="GARCH", p=p, q=q,
                    mean=final_mean, dist=final_dist,
                    col=col, lag=final_lag
                )

            # ========================================================
            # 6. PRÉDICTIONS FINALES
            # ========================================================

            current_step += 1
            progress_bar.progress(current_step / total_steps)

            status_text.text(f"Prédictions pour {col}...")

            forecasting_volatility(
                data=df_pivot[col], col=col,
                model=model, vol="GARCH",
                p=p, q=q,
                mean=final_mean, dist=final_dist, lag=final_lag,
                horizon=horizon, conf_level=conf_int
            )

            # ========================================================
            # 7. RÉSUMÉ DU MODÈLE FINAL
            # ========================================================

            model_summary.append({
                "Entreprise": col,
                "Ordre p": p,
                "Ordre q": q,
                "Moyenne": final_mean,
                "Distribution d'erreur": final_dist,
                "Retard": (
                    str(final_lag)
                    if final_lag is not None
                    else "Aucun"
                )
            })

            # ========================================================
            # 8. RÉSUMÉ DE LA VALIDATION FINALE
            # ========================================================
            
            model_val.append({
                "Entreprise": col,
                "Normalité des résidus": hypothesis_status(df_val_final, "Normalité des résidus"),
                "Indépendance des résidus": hypothesis_status(df_val_final, "Autocorrélation des résidus"),
                "Indépendance des résidus au carré": hypothesis_status(df_val_final, "Autocorrélation des résidus au carré"),
                "Homoscédasticité conditionnelle": hypothesis_status(df_val_final, "Effet ARCH"),
                "Stationnarité conditionnelle": hypothesis_status(df_val_final, "Stationnarité conditionnelle")
            })

        # ============================================================
        # 9. TABLEAU RÉCAPITULATIF DES MODÈLES
        # ============================================================

        model_summary_df = pd.DataFrame(model_summary)

        if not model_summary_df.empty:
            model_summary_df.set_index(
                "Entreprise",
                inplace=True
            )

        # ============================================================
        # 10. TABLEAU RÉCAPITULATIF DES HYPOTHÈSES
        # ============================================================

        model_val_df = pd.DataFrame(model_val)

        if not model_val_df.empty:
            model_val_df.set_index("Entreprise", inplace=True)

        # ============================================================
        # 11. AFFICHAGE
        # ============================================================

        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.write("Veuillez trouver ci-dessous les modèles de volatilité (GARCH) utilisés pour les prédictions de chaque entreprise.")
        st.dataframe(model_summary_df)

        st.write("Veuillez trouver ci-dessous le résumé du respect des hypothèses statistiques associées à chaque modèle final.")
        st.dataframe(model_val_df)

        # ============================================================
        # 12. NETTOYAGE
        # ============================================================

        progress_bar.empty()
        status_text.empty()
    
else:
    st.write('Saisissez les options afin de débuter les analyses puis appuyer sur "Lancer"')
