# Importation
start_date='2023-01-01'
end_date='2024-01-01'
df = import_data(["AAPL", "MSFT", "GOOG","META"], start_date, end_date)

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
    ticker_data["Returns"] = ticker_data["Adj Close"].pct_change(fill_method=None)*100
    
    df_list.append(ticker_data)

# Combiner toutes les DataFrames
df  = pd.concat(df_list, ignore_index=True).dropna()
df_pivot = df.pivot(index="Date", columns="Ticker", values="Returns")

horizon=14

# Initialisation de la liste pour stocker les résultats
model_summary = []

# Choix et validation des modèles
for col in df_pivot.columns:
    # Division en bases d'apprentissage-validation
    train_size = int(0.8 * len(df_pivot[col]))
    train = df_pivot[col].iloc[:train_size]
    test_size = int(len(df_pivot[col])*0.4)
    
    # Test t pour moyenne nulle
    _, p_value_ttest = ttest_1samp(train, popmean=0)
    if p_value_ttest >= 0.05:
        mean_t = 'Zero'
    else:
        mean_t ='Constant'
    
    # Recherche des meilleurs hyperparamètres
    p, q = ARCH_search(train, p_max=8, q_max=8, vol='GARCH', mean=mean_t, criterion='aic')

    # Construction du meilleur modèle selon le critère d'information
    model = arch_model(train, vol='GARCH', p=p, q=q, mean=mean_t, rescale=False)
    model = model.fit(disp='off', options={'maxiter': 750})
    
    # Résidus du modèles
    resid = model.resid
    
    # Validation
    df_val = model_validation(resid)
    
    # Distribution
    kurt_val, skewness_val = distribution(resid)
    
    if all(df_val['Respect']) == 1:
        dist='normal'
        rolling_pred(real_values=df_pivot[col], train=train, test_size=test_size, vol="GARCH", p=p, q=q, mean=mean_t, dist=dist, col=col)
        forecasting_volatility(data=train, model=model,vol='GARCH', p=p, q=q, mean=mean_t, dist='normal', col=col, horizon=horizon)
        break
    else:
        # Choisir le meilleur modèle selon la violation des hypothèses et la distribution des résidus
        mean, dist = mean_dist(df_val, train, kurt_val, skewness_val)
        
        # Recherche des meilleurs hyperparamètres
        p, q = ARCH_search(train, p_max=8, q_max=8, vol='GARCH', mean=mean, dist=dist, criterion='aic')
        
        # Construction du meilleur modèle selon le critère d'information
        model = arch_model(train, vol='GARCH', p=p, q=q, mean=mean, dist=dist, rescale=False)
        model = model.fit(disp='off', options={'maxiter': 750})
        
        # Prédictions glissantes
        # rolling_pred(real_values=df_pivot[col], train=train, test_size=test_size, vol='GARCH', p=p, q=q, mean=mean, dist=dist, col=col)
        forecasting_volatility(data=train, model=model,vol='GARCH', p=p, q=q, mean=mean, dist=dist, col=col, horizon=horizon)
                
    # Ajouter les informations du modèle à la liste
    model_summary.append({
        'Entreprise': col,
        'p': p,
        'q': q,
        'Moyenne': mean,
        'Distribution': dist
    })
    
    end_time = time.time()

# Informations des modèles
model_summary_df = pd.DataFrame(model_summary)
print(model_summary_df)