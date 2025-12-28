#"Weekend Getaway Ranker" PROJECT
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Top_Indian_Places.csv")   


print("Columns in dataset:\n", df.columns)

features = [
    'Google review rating',
    'Number of google review in lakhs',
    'time needed to visit in hrs'
]


df = df.dropna(subset=features)


scaler = MinMaxScaler()

df[['Rating_norm',
    'Popularity_norm',
    'Time_norm']] = scaler.fit_transform(df[features])
df['Final_Score'] = (
    0.45 * df['Rating_norm'] +
    0.35 * df['Popularity_norm'] +
    0.20 * (1 - df['Time_norm'])
)
top_places = df.sort_values(by='Final_Score', ascending=False)
display(
    top_places[['Name', 'City', 'Final_Score']]
    .head(3)
    .style
    .set_properties(**{'font-weight': 'bold'})
)

