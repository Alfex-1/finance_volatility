# Importation
start_date='2023-06-01'
end_date='2024-01-01'
# df = import_data(["AAPL", "MSFT", "GOOG","META"], start_date, end_date)
df = import_data(["MSFT"], start_date, end_date)

# Interpolation des données manquantes
df = interpolate(df, start_date=start_date, end_date=end_date).dropna()

# Remplacement des noms
df['Ticker'] = df['Ticker'].replace({'AAPL': 'Apple',
                                         'MSFT': 'Microsoft',
                                         'AMZN': 'Amazon',
                                         'GOOG': 'Google'})

# Calculer les rendements quotidiens
df_list = []
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data["Returns"] = ticker_data["Close"].pct_change(fill_method=None)*100
    
    df_list.append(ticker_data)

# Combiner toutes les DataFrames
df  = pd.concat(df_list, ignore_index=True).dropna()
df_pivot = df.pivot(index="Date", columns="Ticker", values="Returns")

horizon=7

# Initialisation de la liste pour stocker les résultats
model_summary = []
model_val = []

# Choix et validation des modèles
for col in df_pivot.columns:
    train_size = int(len(df_pivot[col]) * (2/3))
    test_size = len(df_pivot[col]) - train_size

    train = df_pivot[col][:train_size]  # Ensemble d'apprentissage
    
    # Test t pour moyenne nulle
    _, p_value_ttest = ttest_1samp(train, popmean=0)
    if p_value_ttest >= 0.05:
        mean_t = 'Zero'
    else:
        mean_t ='Constant'
    
    # Recherche des meilleurs hyperparamètres
    p, q, _ = ARCH_search(train, p_max=9, q_max=9, vol='GARCH', mean=mean_t, criterion='aic')

    # Construction du meilleur modèle selon le critère d'information
    model = arch_model(train, vol='GARCH', p=p, q=q, mean=mean_t, rescale=False)
    model = model.fit(disp='off', options={'maxiter': 750})
    
    # Résidus du modèles
    resid = model.resid
    
    # Validation
    df_val = model_validation(model)
    
    # Distribution
    kurt_val, skewness_val = distribution(resid)
    
    if all(df_val['Respect']) == 1:
        dist='normal'
        rolling_pred(real_values=df_pivot[col], train=train, test_size=test_size, vol="GARCH", p=p, q=q, mean=mean_t, dist=dist, col=col)
        forecasting_volatility(data=df_pivot[col], model=model,vol='GARCH', p=p, q=q, mean=mean_t, dist='normal', col=col, horizon=horizon, conf_level=0.95)
        break
    else:
        # Choisir le meilleur modèle selon la violation des hypothèses et la distribution des résidus
        mean, dist = mean_dist(df_val, train, kurt_val, skewness_val)
        
        # Recherche des meilleurs hyperparamètres
        if mean == 'AR':
            p, q, lag = ARCH_search(train, p_max=9, q_max=9, vol='GARCH', mean=mean, dist=dist, criterion='aic')
            
            # Construction du meilleur modèle selon le critère d'information
            model = arch_model(train, vol='GARCH', p=p, q=q, mean=mean, dist=dist, lags=lag,  rescale=False)
        
        else:
            lag=None
            p, q, _ = ARCH_search(train, p_max=9, q_max=9, vol='GARCH', mean=mean, dist=dist, criterion='aic')
        
            # Construction du meilleur modèle selon le critère d'information
            model = arch_model(train, vol='GARCH', p=p, q=q, mean=mean, dist=dist, rescale=False)
        
        model = model.fit(disp='off', options={'maxiter': 750})
        
        # Validation
        df_val = model_validation(model)
        
        # Prédictions glissantes
        rolling_pred(real_values=df_pivot[col], test_size=test_size, vol='GARCH', p=p, q=q, mean=mean, dist=dist, col=col, lag=lag)
        forecasting_volatility(data=df_pivot[col], model=model,vol='GARCH', p=p, q=q, mean=mean, dist=dist, col=col, lag=lag, horizon=horizon, conf_level=0.95)
                
        # Ajouter les informations du modèle à la liste
        model_summary.append({
            'Entreprise': col,
            'Ordre p': p,
            'Ordre q': q,
            'Moyenne': mean,
            "Distribution d'erreur": dist,
            'Retard' : "Aucun" if mean != 'AR' else lag
        })
            
        # Ajouter les informations sur le respect des hypothèses
        model_val.append({
            'Entreprise': col,
            'Normalité des résidus': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Normalité des résidus', 'Respect'].values[0] == 1 else "Non",
            'Indépendance des résidus': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Autocorrélation des résidus', 'Respect'].values[0] == 1 else "Non",
            'Indépendance des résidus au carré': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Autocorrélation des résidus au carré', 'Respect'].values[0] == 1 else "Non",
            'Homoscédasticité conditionnelle': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Effet ARCH', 'Respect'].values[0] == 1 else "Non",
            'Stationnarité conditionnelle': "Oui" if df_val.loc[df_val['Hypothèse'] == 'Stationnarité conditionnelle', 'Respect'].values[0] == 1 else "Non"
        })

# Informations des modèles
model_summary_df = pd.DataFrame(model_summary)
model_summary_df.set_index('Entreprise', inplace=True)
model_val_df = pd.DataFrame(model_val)
model_val_df.set_index('Entreprise', inplace=True)

print(model_summary_df)
print(model_val_df)