import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("kurd_mentions_trend_comparisons.csv")
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["kurd_mentions", "keywords", "total_messages"]) #analysing kurdish nationalist mentions against linguistic homophily, controlling for total msg volume

df.set_index("date", inplace=True)

y = df["kurd_mentions"]
X = df[["keywords", "total_messages"]]

model = SARIMAX(y, exog=X, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)


print("\nMODEL SUMMARY:")
print(results.summary())

print("\nMODEL COEFFICIENTS:")
print(results.params)

df["predicted_kurds"] = results.predict(start=1, end=len(y)-1, exog=X.iloc[1:])

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["kurd_mentions"], label="Actual", linewidth=2)
plt.plot(df.index, df["predicted_kurds"], label="ARIMA Forecast", linestyle="--")
plt.title("ARIMA Forecast of Kurdish Nationalist Expression with Exogenous Regressors")
plt.xlabel("Date")
plt.ylabel("Nationalist Expression Incidence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
