import numpy as np
import pandas as pd
import streamlit as st
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
from scipy.stats import skew, jarque_bera, shapiro, ttest_1samp
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
            st.warning(f"Aucune donnée disponible pour {ticker} entre {start_date} et {end_date}. Il sera retiré de l'analyse.")
        else:
            df = df.stack(level=1).reset_index()
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
        df_ticker = df_ticker.interpolate(method='polynomial', order=2)
        df_ticker['Ticker'] = ticker
        
        # Ajouter le DataFrame interpolé à la liste
        df_list.append(df_ticker)

    # Rassembler tous les DataFrames en un seul DataFrame
    df = pd.concat(df_list)

    # Si nécessaire, réinitialiser l'index ou ajuster l'index (par exemple, pour le 'Ticker' et 'Date')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    return df

def fit_model(params, data, mean, dist, vol, o, criterion):
    """
    Ajuste un modèle GARCH aux données et retourne le critère d'information spécifié (AIC ou BIC) pour évaluer la qualité de l'ajustement.

    Args:
        params (dict): Dictionnaire contenant les paramètres du modèle GARCH, avec les clés 'p' et 'q' représentant respectivement l'ordre de l'auto-régression (AR) et de la moyenne mobile (MA).
        data (pd.Series): Série temporelle des données sur lesquelles ajuster le modèle GARCH. Les valeurs manquantes seront ignorées.
        mean (str): Spécification du modèle de la moyenne (par exemple, 'Constant', 'AR', 'Zero', etc.).
        dist (str): Distribution des résidus du modèle (par exemple, 'Normal', 'StudentsT', etc.).
        vol (str): Spécification du modèle de volatilité (par exemple, 'GARCH', 'EGARCH', etc.).
        o (int): Ordre de l'innovation (spécifique à certains modèles comme l'EGARCH).
        criterion (str): Le critère d'évaluation à retourner ('aic' pour le critère d'information d'Akaike ou 'bic' pour le critère d'information bayésien).

    Returns:
        dict: Dictionnaire contenant les valeurs des paramètres 'p' et 'q' et le critère spécifié ('AIC' ou 'BIC') du modèle ajusté.
    """
    p, q = params['p'], params['q']
    model = arch_model(data.dropna(), mean=mean, dist=dist, vol=vol, p=p, q=q, o=o)
    model_fit = model.fit(disp='off', options={'maxiter': 750})
    
    # Retourner uniquement le critère spécifié (AIC ou BIC)
    if criterion == 'aic':
        return {'p': p, 'q': q, 'AIC': model_fit.aic}
    elif criterion == 'bic':
        return {'p': p, 'q': q, 'BIC': model_fit.bic}


def ARCH_search(data, p_max, q_max, o=0, vol='GARCH', mean='Constant', dist='normal', criterion='aic'):
    """
    Recherche le meilleur modèle ARCH/GARCH en utilisant une grille de recherche pour les paramètres p et q.
    Le modèle optimal est sélectionné en fonction du critère spécifié (AIC ou BIC).

    Args:
        data (pd.Series): Séries temporelles des données sur lesquelles ajuster le modèle ARCH/GARCH.
        p_max (int): Valeur maximale de l'ordre p pour le modèle GARCH (nombre de termes d'autorégression).
        q_max (int): Valeur maximale de l'ordre q pour le modèle GARCH (nombre de termes de moyenne mobile).
        o (int, optional): Ordre de l'innovation pour les modèles comme FIGARCH. Par défaut, 0.
        vol (str, optional): Modèle de volatilité à utiliser (par exemple, 'GARCH', 'FIGARCH', 'EGARCH'). Par défaut, 'GARCH'.
        mean (str, optional): Type de modèle pour la moyenne (par exemple, 'Constant', 'AR', etc.). Par défaut, 'Constant'.
        dist (str, optional): Distribution des résidus du modèle (par exemple, 'normal', 't', etc.). Par défaut, 'normal'.
        criterion (str, optional): Critère à utiliser pour sélectionner le meilleur modèle ('aic' ou 'bic'). Par défaut, 'aic'.

    Returns:
        tuple: Un tuple contenant les meilleurs ordres p et q du modèle GARCH sélectionné en fonction du critère.
    """
    p_range = range(1, p_max + 1) if vol != 'FIGARCH' else [0, 1]
    q_range = range(0, q_max + 1) if vol != 'FIGARCH' else [0, 1]
    param_grid = {'p': p_range, 'q': q_range}
    grid = ParameterGrid(param_grid)

    # Exécution parallèle en fonction du critère choisi
    results = Parallel(n_jobs=-1)(delayed(fit_model)(params, data, mean, dist, vol, o, criterion) for params in grid)
    
    # Convertir en DataFrame et trier
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=criterion.upper()).reset_index(drop=True)
    
    p = int(results_df.head(1).iloc[0].values[0])
    q = int(results_df.head(1).iloc[0].values[1])
    
    return p, q

def model_validation(resid):
    """
    Valide les hypothèses relatives à un modèle GARCH sur les résidus d'une série temporelle.

    Args:
        resid (array-like): Les résidus de la série temporelle à tester. Il s'agit des erreurs résiduelles du modèle de série temporelle ajusté.

    Returns:
        pd.DataFrame: Une DataFrame contenant les résultats des tests de validation des hypothèses, y compris les p-values.
    """
    # Création d'un dictionnaire pour stocker les résultats
    results = {
        'Hypothèse': [],
        'Respect': [],
        'P-Value':[]
    }
    
    # 1. Normalité des résidus (p-value > 0.05)
    _, p_shapiro = shapiro(resid)
    results['Hypothèse'].append('Normalité des résidus')
    results['Respect'].append(1 if p_shapiro > 0.05 else 0)
    results['P-Value'].append(p_shapiro)
    
    # 2. Autocorrélation des résidus (p-value <= 0.05 pour toutes les lags)
    lb_resid = acorr_ljungbox(resid, lags=[i for i in range(1, 13)], return_df=True)
    autocorr_resid_pvalues = lb_resid['lb_pvalue']
    results['Hypothèse'].append('Autocorrélation des résidus')
    results['Respect'].append(1 if all(p >= 0.05 for p in autocorr_resid_pvalues) else 0)
    results['P-Value'].append(autocorr_resid_pvalues.tolist())
    
    # 3. Autocorrélation des résidus au carré (p-value <= 0.05 pour toutes les lags)
    lb_resid_sq = acorr_ljungbox(resid**2, lags=[i for i in range(1, 13)], return_df=True)
    autocorr_resid_sq_pvalues = lb_resid_sq['lb_pvalue']
    results['Hypothèse'].append('Autocorrélation des résidus au carré')
    results['Respect'].append(1 if all(p >= 0.05 for p in autocorr_resid_sq_pvalues) else 0)
    results['P-Value'].append(autocorr_resid_sq_pvalues.tolist())
    
    # 4. Hétéroscédasticité conditionnelle (effet ARCH), p-value > 0.05
    lm_test = het_arch(resid)
    results['Hypothèse'].append('Effet ARCH')
    results['Respect'].append(1 if lm_test[1] > 0.05 else 0)
    results['P-Value'].append(lm_test[1])
    
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
    kurt = resid.kurtosis()
    skewness = resid.skew()
    
    return kurt, skewness

def forecast_volatility(i, real_values, test_size, vol, p, q, mean, dist):
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

    Returns:
        float: La volatilité prévisionnelle pour la période suivante, sous forme de racine carrée de la variance prédit.
    """
    current_train = real_values[:-(test_size - i)]
    model = arch_model(current_train, vol=vol, p=p, q=q, mean=mean, dist=dist)
    model_fit = model.fit(disp='off', options={'maxiter': 750})
    pred = model_fit.forecast(horizon=1)
    return np.sqrt(pred.variance.values[-1, :][0])

def rolling_pred(real_values, train, test_size, vol, p, q, mean, dist, col):
    """
    Effectue des prévisions glissantes de la volatilité pour une série temporelle donnée 
    à l'aide d'un modèle ARCH/GARCH et affiche les résultats.

    Args:
        real_values (pd.Series): Série temporelle complète contenant les valeurs réelles.
        train (pd.Series): Partie d'entraînement de la série temporelle.
        test_size (int): Taille de la période de test pour les prévisions glissantes.
        vol (str): Modèle de volatilité à utiliser ('GARCH', 'ARCH', etc.).
        p (int): Ordre du processus ARCH.
        q (int): Ordre du processus GARCH.
        mean (str): Modèle de la moyenne à utiliser ('Constant', 'Zero', etc.).
        dist (str): Distribution à utiliser pour les résidus ('normal', 't', 'skewt', etc.).
        col (str, optional): Nom de la colonne associée à la série temporelle. Par défaut `col`.
    
    Returns:
        pd.Series: Série des prévisions glissantes de la volatilité pour la période de test.
    
    Displays:
        Un graphique des valeurs réelles de la série et des prévisions glissantes.
    """
    rolling_predictions = []
    rolling_predictions = Parallel(n_jobs=-1, verbose=0)(
        delayed(forecast_volatility)(i, real_values, test_size, vol, p, q, mean, dist) for i in range(test_size)
    )

    rolling_predictions_df = pd.Series(rolling_predictions, index=real_values[-test_size:].index)
    true = real_values[-test_size:]
    preds = rolling_predictions_df
    preds.index = true.index 
    
    plt.figure(figsize=(9, 5))
    plt.plot(true.index, true, label=f'Rendement réel')
    plt.plot(true.index, preds, label='Volatilité prédite', linestyle='dashed')
    plt.title(f'\nPrévision glissante de la volatilité des actions {col}\n', fontsize=15)
    plt.legend(fontsize=12)
    plt.ylabel('Volatilité et rendements (en %)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def forecasting_volatility(data, model, vol, p, q, mean, dist, col, horizon):
    """
    Prédit la volatilité future d'un actif à l'aide d'un modèle ARCH/GARCH ajusté, en générant des prévisions sur un horizon spécifié.

    Args:
        data (pd.Series): Séries temporelles des rendements ou des prix historiques de l'actif sur lequel calculer la volatilité.
        model (str): Le type de modèle ARCH/GARCH à utiliser pour la prévision (par exemple, 'GARCH', 'EGARCH', etc.).
        vol (str): Modèle de volatilité à utiliser dans le cadre du modèle ARCH/GARCH (par exemple, 'GARCH', 'EGARCH').
        p (int): Ordre de l'auto-régression (p) dans le modèle GARCH.
        q (int): Ordre de la moyenne mobile (q) dans le modèle GARCH.
        mean (str): Modèle pour la moyenne (par exemple, 'Constant', 'AR', etc.).
        dist (str): Distribution des résidus dans le modèle (par exemple, 'normal', 't', etc.).
        col (str): Le nom de l'actif ou de la colonne dans les données, utilisé pour les titres et légendes.
        horizon (int): L'horizon de prévision, c'est-à-dire le nombre de jours pour lesquels la volatilité doit être prédite.

    Returns:
        None: Affiche un graphique représentant la volatilité prédite pour l'horizon spécifié.
    """
    model = arch_model(data, vol=vol, p=p, q=q, mean=mean, dist=dist)
    model_fit = model.fit(disp='off', options={'maxiter': 750})
    pred = model_fit.forecast(horizon=horizon)
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, horizon+1)]
    pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)
    
    plt.figure(figsize=(9, 5))
    plt.plot(pred)
    plt.title(f'\nPrédiction de volatilité des actions {col} pour les {horizon-1} prochains jours\n', fontsize=15)
    plt.ylabel("Volatilité prédite (en %)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def select_volatility_model(df_results, kurt, skewness):
    """
    Sélectionne le modèle de volatilité à utiliser basé sur les résultats des tests de validation
    et les caractéristiques des résidus.

    Args:
        df_results (pd.DataFrame): Résultats des tests de validation du modèle (hypothèses validées ou non).
        kurt (float): Kurtosis des résidus.
        skewness (float): Skewness (asymétrie) des résidus.

    Returns:
        str: Le modèle de volatilité à utiliser ('GARCH', 'EGARCH', 'GJR-GARCH', 'APARCH').
    """
    # Vérification des hypothèses de validation
    if all(df_val['Respect']==1):
        # Si toutes les hypothèses sont respectées (résidus normaux, pas d'autocorrélation ni d'effet ARCH)
        model = 'GARCH'
        
    # Si des autocorrélations sont significatives à long terme
    if df_results.loc[df_results['Hypothèse'] == 'Autocorrélation des résidus au carré', 'Respect'].values[0] == 0 and df_results.loc[df_results['Hypothèse'] == 'Autocorrélation des résidus', 'Respect'].values[0] == 0:
        model = 'FIGARCH'
    
    if df_results.loc[df_results['Hypothèse'] == 'Autocorrélation des résidus au carré', 'Respect'].values[0] == 0 or df_results.loc[df_results['Hypothèse'] == 'Autocorrélation des résidus', 'Respect'].values[0] == 0:
        model = 'HGARCH'
    
    # Si les résidus montrent une asymétrie et une kurtosis élevée (indiquant des queues épaisses)
    diff_kurt = abs(kurt-3)
    if diff_kurt >= 0.4 and (abs(skewness) >= 0.5):
        model = 'APARCH'

    else:
        model = 'EGARCH'
    
    # Autres situations : EGARCH
    return model

def mean_dist(hyp_df, data, kurtosis, skewness):
    """
    Détermine la spécification de la moyenne et de la distribution d'un modèle basé sur les hypothèses
    statistiques et les caractéristiques de la série temporelle d'entraînement.

    Args:
        hyp_df (pd.DataFrame): DataFrame contenant les résultats des tests d'hypothèses pour les résidus,
                                notamment la vérification de l'autocorrélation des résidus et leur carré.
        train (array-like): Série temporelle d'entraînement utilisée pour le test de la moyenne. 
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
            - 'ged' si la kurtose est significativement différente de 3 et l'asymétrie est forte.
            - 't' si la kurtose est significativement différente de 3 et l'asymétrie est faible.
            - 'skewt' si la kurtose est proche de 3 et l'asymétrie est significative.
            - 'normal' si la kurtose est proche de 3 et l'asymétrie est faible.
    """
    # Détermination de la moyenne
    _, p_value_ttest = ttest_1samp(data, popmean=0)
    autocorr_resid = hyp_df.loc[hyp_df['Hypothèse'] == 'Autocorrélation des résidus', 'Respect'].values[0]
    autocorr_resid_squared = hyp_df.loc[hyp_df['Hypothèse'] == 'Autocorrélation des résidus au carré', 'Respect'].values[0]
    
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
    diff_kurt = abs(kurtosis-3)
    if hyp_df.loc[hyp_df['Hypothèse'] == 'Normalité des résidus', 'P-Value'].values[0] > 0.01:
        dist='normal'
    else:
        if diff_kurt >= 0.3 and abs(skewness) >= 0.3:
            dist='ged'
        elif diff_kurt >= 0.3 and abs(skewness) < 0.3:
            dist = 't'
        elif diff_kurt < 0.3 and abs(skewness) >= 0.3:
            dist = 'skewt'
        else:
            dist='t'

    return str(mean), str(dist)

st.title("Analyse des prix et des rendements des actions de plusieurs entreprises et prédiction des risques associés")
st.subheader("Auteur : BRUNET Alexandre")
st.write(
    ("Bienvenue sur l'application dans laquelle vous pouvez visualiser le prix des actions des entreprises ainsi que leur rendements quotidiens. "
     "Vous pouvez égalemet choisir de visualiser les prédictions des risques liés à aux investissements des actions des entreprises, sur le court terme.")
)

# Case à cocher pour "Analyse" et "Prédiction"
option = st.selectbox("Choisissez une option", ["Analyse", "Prédiction"])

# Entreprises
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
sp500_df = tables[0]
tickers = sp500_df[['Symbol', 'Security']]
ticker_to_name = dict(zip(tickers['Symbol'], tickers['Security']))
selected_companies = st.multiselect("Choisissez les entreprises à analyser", 
                                    tickers['Security'].tolist(),
                                    max_selections=4)

start_date = None
end_date = None

if option == "Analyse" and len(selected_companies) >=1:
    # Importation des données
    today = datetime.today()
    default_end_date = today - timedelta(days=6*30)  # 6 mois avant aujourd'hui

    # Définir la date de début par défaut comme étant 1 an avant la date de fin
    default_start_date = default_end_date - timedelta(days=365)  # 1 an avant la date de fin

    # Utiliser Streamlit pour afficher les dates avec les valeurs par défaut
    start_date = st.date_input("Date de début", value=default_start_date.date())
    end_date = st.date_input("Date de fin", value=default_end_date.date())    

    selected_tickers = tickers[tickers['Security'].isin(selected_companies)]['Symbol'].tolist()
    
    df = import_data(selected_tickers, start_date, end_date)
    
    if df is None:
        st.warning("Aucune données disponibles n'ont pu être trouvé pour cette période et pour ces entreprises.")
    else:
        df = interpolate(df, start_date=start_date, end_date=end_date).dropna()
        # Donner les vrais noms
        df['Ticker'] = df['Ticker'].map(ticker_to_name)
        
        # Lancer l'application
        launch = st.button("Lancer")
    
elif option == "Prédiction" and len(selected_companies) >=1:   
    # Importation des données
    end_date = st.date_input("Sélectionner la date de fin", value=pd.to_datetime("today"))
    start_date = end_date - pd.Timedelta(days=365 + 30 * 6)
    
    # Choisir l'horizon des prédictions
    horizon = st.slider("Choisissez l'horizon des prédictions (en jours)", min_value=2, max_value=15, value=7) + 1    

    selected_tickers = tickers[tickers['Security'].isin(selected_companies)]['Symbol'].tolist()
    df = import_data(selected_tickers, start_date, end_date)
    
    if df is None:
        st.warning("Aucune donnée disponible n'a pu être trouvée pour cette période et pour ces entreprises.")
    else:
        df = interpolate(df, start_date=start_date, end_date=end_date).dropna()
        # Donner les vrais noms
        df['Ticker'] = df['Ticker'].map(ticker_to_name)
        
        # Lancer l'application
        launch = st.button("Lancer")
elif option == None:
    st.warning("Veuillez sélectionner une seule option à la fois.")

if option == "Analyse" and len(selected_companies) >= 1 and start_date and end_date and df is not None and launch:
    # Visualisation des prix des actions
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=df, x="Date", y="Adj Close", hue="Ticker")
    plt.title("\nÉvolution des prix de clôture par entreprise\n", fontsize=16)
    plt.xlabel(None)
    plt.ylabel("Prix de clôture (en USD)", fontsize=13)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.grid(True)
    plt.legend(title="Entreprises", fontsize=12.5, title_fontsize=14, loc='best')
    st.pyplot(plt)

    # Calculer les rendements quotidiens et cumulés
    df_list = []
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker].copy()
        ticker_data["Returns"] = ticker_data["Adj Close"].pct_change(fill_method=None)
        ticker_data["Cumul_returns"] = (1 + ticker_data["Returns"]).cumprod() - 1
        
        df_list.append(ticker_data)

    # Combiner toutes les DataFrames
    df_returns  = pd.concat(df_list, ignore_index=True).dropna()

    # Visualiser l'évolution des rendements quotidiens et cumulés
    df_returns["Cumul_returns"] *= 100
    df_returns["Returns"] *= 100

    plt.figure(figsize=(10, 6))
    for ticker in df['Ticker'].unique():
        ticker_data = df_returns[df_returns['Ticker'] == ticker]
        plt.plot(ticker_data['Date'], ticker_data['Returns'], label=ticker)
    plt.title("\nRendements journaliers par entreprise\n",fontsize=15)
    plt.xlabel(None)
    plt.ylabel("Rendements journaliers (en %)",fontsize=13)
    plt.legend(title="Entreprises", fontsize=12.5, title_fontsize=14, loc='best')
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)


    plt.figure(figsize=(10, 6))
    for ticker in df['Ticker'].unique():
        ticker_data = df_returns[df_returns['Ticker'] == ticker]
        plt.plot(ticker_data['Date'], ticker_data['Cumul_returns'], label=ticker)
    plt.title("\nEvolution des rendements journaliers cumulés par entreprise\n",fontsize=15)
    plt.xlabel(None)
    plt.ylabel("Rendements cumulés journaliers (%)",fontsize=13)
    plt.legend(title="Entreprises", fontsize=12.5, title_fontsize=14, loc='best')
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    # Comparaison des rendements moyennes et de leur volatilité (risques)
    mean_list = []
    std_list = []

    for ticker in df_returns['Ticker'].unique():
        ticker_data = df_returns[df_returns['Ticker'] == ticker].copy()
        mean = ticker_data['Returns'].mean()
        std = ticker_data['Returns'].std()
        
        mean_list.append(mean)
        std_list.append(std)

    # Créer la DataFrame des performances
    df_perf = pd.DataFrame({
        "Ticker": df['Ticker'].unique(),
        "Moyenne": mean_list,
        "Ecart-type": std_list})

    # Visualiser les risques et les performances
    df_perf_melted = pd.melt(df_perf, id_vars=["Ticker"], value_vars=["Moyenne", "Ecart-type"], 
                            var_name="Mesure", value_name="Valeur")
    df_perf_melted["Mesure"] = df_perf_melted["Mesure"].replace({
        "Moyenne": "Performance",
        "Ecart-type": "Risque"
    })
    df_perf_melted = df_perf_melted.sort_values(
        by=["Mesure", "Valeur"], 
        ascending=[True, False]
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Ticker", y="Valeur", hue="Mesure", data=df_perf_melted)
    plt.title("\nPerformances et risques des actions par entreprise\n", fontsize=15)
    plt.xlabel(None)
    plt.ylabel("Mesures en %", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(title=None, fontsize=13, loc='best')

    # Ajouter les valeurs des étiquettes sur chaque barre
    for p in ax.patches:
        height = p.get_height()  # Obtenir la hauteur de chaque barre
        if height != 0:  # Ignorer les barres dont la hauteur est zéro
            ax.annotate(f'{height:.2f}',  # Formatage avec 2 décimales
                        xy=(p.get_x() + p.get_width() / 2, height),  # Positionnement au sommet de la barre
                        xytext=(0, 3),  # Décalage vertical pour le texte
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)
    st.pyplot(plt)
        
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
            xaxis_rangeslider_visible=True,  # Activer la barre de zoom interactive
            template="plotly_dark",
            height=800,
            width=1200
        )
        
        # Afficher le graphique
        st.plotly_chart(fig)

    # Vérification de l'existance de corrélations
    df_pivot = df_returns.pivot(index="Date", columns="Ticker", values="Returns")
    tickers = df_pivot.columns

    if len(tickers) > 1:
        ticker_pairs = list(combinations(tickers, 2))

        # Calcul automatique des dimensions du tableau
        num_pairs = len(ticker_pairs)
        ncols = 3  # Fixer un nombre raisonnable de colonnes (modifiable si besoin)
        nrows = math.ceil(num_pairs / ncols)  # Utiliser math.ceil pour avoir assez de lignes

        # Créer la figure et les axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten()  # Aplatir pour une itération facile

        # Traiter chaque paire de tickers
        for idx, (ticker_x, ticker_y) in enumerate(ticker_pairs):
            ax = axes[idx]
            sns.regplot(
                x=df_pivot[ticker_x], 
                y=df_pivot[ticker_y], 
                ax=ax, 
                scatter_kws={'s': 20}, 
                line_kws={'color': 'red'}, 
                ci=None,  # Supprime l'intervalle de confiance
                robust=True  # Robustesse aux outliers
            )
            ax.set_xlabel(ticker_x, fontsize=13)
            ax.set_ylabel(ticker_y, fontsize=13)
            ax.tick_params(axis='both', labelsize=14)

        # Désactiver les axes inutilisés si le nombre de paires est inférieur au nombre d'axes
        for idx in range(num_pairs, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("\nRelation entre les prix de clôture ajustés entre chaque entreprise", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig)  # Passer l'objet 'fig' directement à Streamlit

        # Créer la matrice de corrélation
        correlation_matrix = df_pivot.corr(method='spearman')* 100

        # Affichage de la heatmap des corrélations
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="flare", fmt=".2f")
        plt.title("\nCorrélations entre les prix de clôture de chaque entreprise (en %)\n")
        plt.xlabel(None)
        plt.ylabel(None)
        plt.grid(False)
        st.pyplot(plt) 

elif option == "Prédiction" and len(selected_companies) >= 1 and end_date and df is not None and launch:
    gif_list = [
    "https://tenor.com/fr/view/pussy-financial-pussy-stonks-stocks-stonks-up-gif-24960970.gif",
    "https://tenor.com/fr/view/money-cat-gif-25358470.gif",
    "https://tenor.com/fr/view/financial-advisor-jet760-says-no-financial-advisor-gif-13070262923158806499.gif",
    "https://tenor.com/fr/view/cat-money-hit-cute-gif-21672589.gif",
    "https://tenor.com/fr/view/cat-finance-financial-advisor-cute-cute-cat-gif-304932396169847213.gif",
    "https://tenor.com/fr/view/zoratsaiyt-zoradraws-acctandsec-會計-秘書-gif-25086834.gif",
    "https://tenor.com/fr/view/tenset-10set-cat-clock-gif-11918755868618435360.gif",
    "https://tenor.com/fr/view/lazy-cat-waiting-bored-kitten-gif-18187747.gif",
    "https://tenor.com/fr/view/sure-cat-nail-file-gif-17863326.gif",
    "https://tenor.com/fr/view/stillwaiting-mad-waitingforyou-kitten-waiting-gif-20072807.gif",
    "https://tenor.com/fr/view/cat-waiting-kicking-stare-gif-26949820.gif",
    "https://tenor.com/fr/view/tired-cat-texting-in-bed-me-in-bed-gif-14635242.gif",
    "https://tenor.com/fr/view/cat-cat-meme-meme-cat-clean-screen-gif-15152740950488033203.gif"]
    gif_url = np.random.choice(gif_list)
    gif_placeholder = st.empty()

    with st.spinner("La recherche du modèle optimal pour chaque entreprise peut durer quelques secondes. Merci de patienter !"):
        gif_placeholder.image(gif_url, width=300)
        # Calculer les rendements quotidiens
        df_list = []
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].copy()
            ticker_data["Returns"] = ticker_data["Adj Close"].pct_change(fill_method=None)*100
            df_list.append(ticker_data)

        # Combiner toutes les DataFrames
        df = pd.concat(df_list, ignore_index=True).dropna()
        df_pivot = df.pivot(index="Date", columns="Ticker", values="Returns")

        # Initialisation de la liste pour stocker les résultats
        model_summary = []
        model_val = []

        # Choix et validation des modèles
        for col in df_pivot.columns:
            train = df_pivot[col]
            test_size = int(len(df_pivot[col]) * 0.4)

            # Test t pour moyenne nulle
            _, p_value_ttest = ttest_1samp(train, popmean=0)
            if p_value_ttest >= 0.05:
                mean_t = 'Zero'
            else:
                mean_t = 'Constant'

            # Recherche des meilleurs hyperparamètres
            p, q = ARCH_search(train, p_max=7, q_max=7, vol='GARCH', mean=mean_t, criterion='aic')

            # Construction du meilleur modèle selon le critère d'information
            model = arch_model(train, vol='GARCH', p=p, q=q, mean=mean_t, rescale=False)
            model = model.fit(disp='off', options={'maxiter': 750})

            # Résidus du modèle
            resid = model.resid

            # Validation
            df_val = model_validation(resid)

            # Distribution
            kurt_val, skewness_val = distribution(resid)

            if all(df_val['Respect']) == 1:
                dist = 'normal'
                rolling_pred(real_values=df_pivot[col], train=train, test_size=test_size, vol="GARCH", p=p, q=q, mean=mean_t, dist=dist, col=col)
                forecasting_volatility(data=train, model=model, vol='GARCH', p=p, q=q, mean=mean_t, dist='normal', col=col, horizon=horizon)
                break
            else:
                # Choisir le meilleur modèle selon la violation des hypothèses et la distribution des résidus
                mean, dist = mean_dist(df_val, train, kurt_val, skewness_val)

                # Recherche des meilleurs hyperparamètres
                p, q = ARCH_search(train, p_max=7, q_max=7, vol='GARCH', mean=mean, dist=dist, criterion='aic')

                # Construction du meilleur modèle selon le critère d'information
                model = arch_model(train, vol='GARCH', p=p, q=q, mean=mean, dist=dist, rescale=False)
                model = model.fit(disp='off', options={'maxiter': 750})
                resid = model.resid
                
                # Validation
                df_val = model_validation(resid)

                # Prédictions glissantes
                rolling_pred(real_values=df_pivot[col], train=train, test_size=test_size, vol='GARCH', p=p, q=q, mean=mean, dist=dist, col=col)
                forecasting_volatility(data=train, model=model, vol='GARCH', p=p, q=q, mean=mean, dist=dist, col=col, horizon=horizon)

            # Ajouter les informations du modèle à la liste
            model_summary.append({
                'Entreprise': col,
                'Ordre p': p,
                'Ordre q': q,
                'Moyenne': mean,
                "Distribution d'erreur": dist
            })
            
            # Ajouter les informations sur le respect des hypothèses
            model_val.append({
                'Entreprise': col,
                'Normalité des résidus': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Normalité des résidus', 'Respect'].values[0] == 1 else "Non",
                'Indépendance des résidus': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Autocorrélation des résidus', 'Respect'].values[0] == 1 else "Non",
                'Indépendance des résidus au carré': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Autocorrélation des résidus au carré', 'Respect'].values[0] == 1 else "Non",
                'Homoscédasticité conditionnelle': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Effet ARCH', 'Respect'].values[0] == 1 else "Non",
            })

        # Informations des modèles
        model_summary_df = pd.DataFrame(model_summary)
        model_val_df = pd.DataFrame(model_val)
        
        st.write("Veuillez trouver ci-dessous les modèles de volatilité (GARCH) utilisés pour les prédictions de chaque entreprise.")
        st.dataframe(model_summary_df)
        
        st.write("Veuillez trouver ci-dessous le résumé du respect des hypothèses statistiques associées à chaque modèle.")
        st.dataframe(model_val_df)
        gif_placeholder.empty()
    
else:
    st.write('Saisissez les options afin de débuter les analyses puis appuyer sur "Lancer"')
