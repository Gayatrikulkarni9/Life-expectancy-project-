# Life-expectancy-project-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

try:
    df = pd.read_csv("Life Expectancy Data.csv")
except FileNotFoundError:
    print("File not found. Please upload the 'Life Expectancy Data.csv.")
    uploaded = files.upload()
    if uploaded:
        # Assuming the user uploads the correct file
        for fn in uploaded.keys():
            print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
            df = pd.read_csv(fn)
    else:
        print("No file was uploaded. Please upload the file to proceed.")
        df = None

if df is not None:
    print("Dataset loaded successfully.")


  imputer = SimpleImputer(strategy="mean")

for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = imputer.fit_transform(df[[col]])

numerical_cols = df.select_dtypes(include=['float64','int64']).columns

for col in numerical_cols:
    sns.boxplot(y=df[col])
    plt.show()


for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

  iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

  df[col] = np.where((df[col] < lower) | (df[col] > upper),
                       df[col].mean(),
                       df[col])

plt.figure(figsize=(10,6))
sns.histplot(df["Life expectancy "], kde=True)
plt.title("Distribution of Life Expectancy")
plt.show()


avg_life = df.groupby("Year")["Life expectancy "].mean()

plt.plot(avg_life)
plt.title("Average Life Expectancy Over Years")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.show()


sns.barplot(x="Status", y="Life expectancy ", data=df)
plt.title("Life Expectancy by Country Status")
plt.show()


plt.figure(figsize=(15,10))
numerical_df = df.select_dtypes(include=np.number)
sns.heatmap(numerical_df.corr(), annot=True, cmap="viridis")
plt.title("Correlation Matrix")
plt.show()

le = LabelEncoder()
df["Status"] = le.fit_transform(df["Status"])

X = df.drop(["Life expectancy ", "Country"], axis=1)
y = df["Life expectancy "]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

model = XGBRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))



from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_scaled, y, cv=10, scoring="r2")

print("Mean CV Score:", scores.mean())

































































