import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv('Cleaned_SpotifyFeatures.csv')
df_sample = df.sample(5000, random_state=42)
X = df_sample[['loudness']]
y = df_sample['popularity']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
plt.scatter(X, y, label='Actual Popularity')
plt.plot(X, predictions, color='red', label='Predicted Trend')
plt.title('Predicting Popularity from Loudness')
plt.xlabel('Loudness (dB)')
plt.ylabel('Popularity')
plt.legend()
plt.show()
