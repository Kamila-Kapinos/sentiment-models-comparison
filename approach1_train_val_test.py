import pandas as pd
import numpy as np
import time
import warnings
import copy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("==================================================")
print("=== PODEJŚCIE 1: Złożone (Train / Val / Test) ===")
print("==================================================\n")

# 1. Wczytanie danych
print("Wczytywanie danych...")
df = pd.read_csv('data/processed_reviews.csv')
X = df['full_text']
y = df['label']

# 2. Podział na 3 zbiory (Test: 20%, Val: 15%, Train: 65%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1875, random_state=42, stratify=y_temp)

print(f"Podział danych zakończony:")
print(f"  Train:      {len(X_train)} próbek")
print(f"  Validation: {len(X_val)} próbek")
print(f"  Test:       {len(X_test)} próbek\n")

# 3. Definicja Pipeline i GridSearchCV
gs_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words=None, min_df=3)),
    ('svm', LinearSVC(max_iter=3000, dual='auto', random_state=42))
])

param_grid = {
    'tfidf__max_features': [15000, 30000, None],
    'tfidf__ngram_range':  [(1, 2), (1, 3)],
    'svm__C':              [0.3, 1, 5],
}

grid_search = GridSearchCV(
    gs_pipe, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1, refit=True
)

# 4. Optymalizacja na zbiorze Train
print("Rozpoczynam GridSearchCV na zbiorze Train...")
start = time.time()
grid_search.fit(X_train, y_train)
print(f"Zakończono w czasie: {time.time() - start:.1f}s")
print(f"Najlepsze hiperparametry: {grid_search.best_params_}")
print(f"CV F1-macro na X_train: {grid_search.best_score_:.4f}\n")

# 5. Sprawdzenie na zbiorze Walidacyjnym (Sanity Check)
print("Ewaluacja modelu na niezależnym zbiorze Validation...")
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"F1-macro (Validation): {val_f1:.4f}\n")

cm_val = confusion_matrix(y_val, y_val_pred)
cm_val_pct = cm_val.astype(float) / cm_val.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7, 5))
sns.heatmap(cm_val_pct, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Neg (1-2)', 'Neu (3)', 'Pos (4-5)'],
            yticklabels=['Neg (1-2)', 'Neu (3)', 'Pos (4-5)'])
plt.ylabel('Etykieta rzeczywista')
plt.xlabel('Predykcja modelu')
plt.title(f'Confusion Matrix - Zbior Walidacyjny (F1={val_f1:.4f})')
plt.tight_layout()
plt.show()

# 6. Refitowanie (dotrenowanie) na połączonym zbiorze Train + Validation
print("Refitowanie modelu na połączeniu Train + Validation przed testem...")
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

final_model = copy.deepcopy(best_model)
final_model.fit(X_trainval, y_trainval)

# 7. Finalny Test
print("\n=== WYNIKI FINALNE NA ZBIORZE TEST ===")
y_test_pred = final_model.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred, average='macro')
print(f"F1-macro (Test): {test_f1:.4f}\n")
print(classification_report(y_test, y_test_pred, target_names=['Negative (1-2)', 'Neutral (3)', 'Positive (4-5)']))

# 8. Confusion Matrix
cm_test = confusion_matrix(y_test, y_test_pred)
cm_test_pct = cm_test.astype(float) / cm_test.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7, 5))
sns.heatmap(cm_test_pct, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['Neg (1-2)', 'Neu (3)', 'Pos (4-5)'],
            yticklabels=['Neg (1-2)', 'Neu (3)', 'Pos (4-5)'])
plt.ylabel('Etykieta rzeczywista')
plt.xlabel('Predykcja modelu')
plt.title(f'Confusion Matrix - Zbior Testowy (F1={test_f1:.4f})')
plt.tight_layout()
plt.show()
