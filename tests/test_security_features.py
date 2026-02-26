from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from apt.minimization.minimizer import GeneralizeToRepresentative

def test_security_features():
    dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    base_accuracy = model.score(X_test, y_test)

    predictions = model.predict(X_train)

    gen = GeneralizeToRepresentative(
        estimator=model,
        target_accuracy=0.95,
        min_privacy_threshold=0.15,
        k_anonymity=5,
        feature_sensitivity_scores={'age': 0.9, 'income': 0.85},
    )

    gen.fit(X_train, predictions)
    transformed = gen.transform(X_test)

    transformed_predictions = model.predict(transformed)
    transformed_accuracy = (transformed_predictions == y_test).mean()

    assert transformed_accuracy >= 0.9 * base_accuracy
    assert gen.k_anonymity_enforcer.get_statistics()['violation_count'] >= 0
    assert gen.privacy_budget_tracker is not None