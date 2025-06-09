import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

raw_df = pd.read_csv('data/user_behavior_dataset.csv')
raw_df.drop('User ID', axis=1, inplace=True)
encoded_df = pd.get_dummies(raw_df, columns=['Device Model', 'Operating System', 'Gender'], drop_first=True)
columns = encoded_df.drop('User Behavior Class', axis=1).columns.tolist()

X_train, X_test, y_train, y_test = joblib.load('data/preprocessed_data.pkl')

X_combined = np.vstack((X_train, X_test))
y_combined = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)

label_encoder = joblib.load('models/label_encoder.pkl')
y_combined_decoded = label_encoder.inverse_transform(y_combined)

df_encoded = pd.DataFrame(X_combined, columns=columns)
df_encoded['User Behavior Class'] = y_combined_decoded

sns.countplot(x='User Behavior Class', data=df_encoded)
plt.title('Distribusi Kelas (Encoded Dataset)')
plt.xlabel('Kelas')
plt.ylabel('Jumlah')
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=False)
plt.title('Heatmap Korelasi Fitur (Encoded Dataset)')
plt.tight_layout()
plt.show()

