from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42,
    ),
}

scores = {}

for name, model in models.items():
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    scores[name] = cv_scores.mean()

final_model = models[max(scores, key=scores.get)]
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
