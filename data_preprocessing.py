import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv('data.csv')

# Define preprocessing steps
numeric_features = ['funding_amount', 'team_size', 'revenue', 'profit']
categorical_features = ['market_sector', 'location']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data
X = data.drop('success', axis=1)
y = data['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Save preprocessed data
pd.DataFrame(X_train).to_csv('data/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/y_test.csv', index=False)

