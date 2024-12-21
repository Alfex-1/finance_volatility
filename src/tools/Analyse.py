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

# Visualisation des prix des actions
plt.figure(figsize=(11, 6))
sns.lineplot(data=df, x="Date", y="Adj Close", hue="Ticker")
plt.title("\nÉvolution des prix de clôture par entreprise\n", fontsize=16)
plt.xlabel(None)
plt.ylabel("Prix de clôture", fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.grid(True)
plt.legend(title="Entreprises", fontsize=12.5, title_fontsize=14, loc='best')
plt.show()

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
plt.title("\nRendements quotidiens par entreprise\n",fontsize=15)
plt.xlabel(None)
plt.ylabel("Rendements journaliers (%)",fontsize=13)
plt.legend(title="Entreprises", fontsize=12.5, title_fontsize=14, loc='best')
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
for ticker in df['Ticker'].unique():
    ticker_data = df_returns[df_returns['Ticker'] == ticker]
    plt.plot(ticker_data['Date'], ticker_data['Cumul_returns'], label=ticker)
plt.title("\nRendements quotidiens cumulés par entreprise\n",fontsize=15)
plt.xlabel(None)
plt.ylabel("Rendements cumulés journaliers (%)",fontsize=13)
plt.legend(title="Entreprises", fontsize=12.5, title_fontsize=14, loc='best')
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()

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
plt.title("\nPerformance et risque des rendements journaliers par entreprise\n", fontsize=15)
plt.xlabel("Entreprises", fontsize=13)
plt.ylabel("Mesures (en %)", fontsize=13)
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
plt.show()
    
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
        title=f"Performance journalière des actions {ticker}",
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
        template="plotly_dark",  # Choix de style (modifiable selon vos goûts)
        height=800,
        width=1200
    )
    
    # Afficher le graphique
    fig.show()

# Vérification de l'existance de corrélations
df_pivot = df_returns.pivot(index="Date", columns="Ticker", values="Returns")
tickers = df_pivot.columns

if len(tickers) > 1:
    ticker_pairs = list(combinations(tickers, 2))

    # Calcul automatique des dimensions du tableau
    num_pairs = len(ticker_pairs)
    ncols = 3  # Fixer un nombre raisonnable de colonnes (modifiable si besoin)
    nrows = int(round(num_pairs / ncols,0))

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
    plt.show()

    # Créer la matrice de corrélation
    correlation_matrix = df_pivot.corr(method='spearman')

    # Affichage de la heatmap des corrélations
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Corrélations entre les prix de clôture de chaque entreprises")
    plt.xlabel("Entreprises")
    plt.ylabel("Entreprises")
    plt.grid(False)
    plt.show()