import warnings
# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from apt.minimization.minimizer import GeneralizeToRepresentative

def main():
    print("=" * 80)
    print("SECURITY-ENHANCED DATA MINIMIZATION TEST")
    print("=" * 80)
    print()
    
    # Load and prepare data
    print("[Step 1] Loading dataset...")
    dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print()
    
    # Train base model
    print("[Step 2] Training base model...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    base_accuracy = model.score(X_test, y_test)
    print(f"  Base model accuracy: {base_accuracy:.4f}")
    print()
    
    # Get predictions for training data
    predictions = model.predict(X_train)
    
    # Initialize with all security features
    print("[Step 3] Initializing GeneralizeToRepresentative with security features...")
    print("  - Privacy budget threshold: 0.15")
    print("  - K-anonymity: k=5")
    print("  - Sensitivity weighting: age = 0.9 and income = 0.85")
    print()
    
    gen = GeneralizeToRepresentative(
        estimator=model,
        target_accuracy=0.90,
        min_privacy_threshold=0.5,      # Feature 1: Privacy budget enforcement
        k_anonymity=7,                    # Feature 2: K-anonymity protection
        feature_sensitivity_scores={      # Feature 3: Sensitivity weighting
            'age': 0.9,
            'income': 0.85
        }
    )
    print()
    
    # Fit with security features active
    print("[Step 4] Fitting generalization model with security features...")
    print("-" * 80)
    gen.fit(X_train, predictions)
    print("-" * 80)
    print()
    
    # Display results
    print("[Step 5] Results Summary:")
    print("=" * 80)
    print(f"Final NCP score (fit): {gen.ncp.fit_score:.4f}")
    
    if gen.privacy_budget_tracker:
        margin = gen.privacy_budget_tracker.get_privacy_margin()
        print(f"Privacy margin: {margin:.4f}")
        print(f"Privacy threshold met: {margin >= 0}")
        print(f"Privacy history length: {len(gen.privacy_budget_tracker.get_privacy_history())}")
    
    if gen.k_anonymity_enforcer:
        stats = gen.k_anonymity_enforcer.get_statistics()
        print(f"K-anonymity violations fixed: {stats['violation_count']}")
        print(f"Cells merged: {stats['merge_count']}")
    
    if gen.sensitivity_calculator:
        sensitivity_scores = gen.sensitivity_calculator.get_all_sensitivity_scores()
        print(f"Feature sensitivity scores calculated: {len(sensitivity_scores)}")
        for feature, score in sensitivity_scores.items():
            print(f"  {feature}: {score:.3f}")
    
    print()
    
    # Transform test data
    print("[Step 6] Transforming test data...")
    transformed = gen.transform(X_test)
    print(f"Transformation complete. Shape: {transformed.shape}")
    print(f"Transform NCP score: {gen.ncp.transform_score:.4f}")
    print()
    
    # Verify accuracy
    print("[Step 7] Verifying model accuracy on transformed data...")
    transformed_predictions = model.predict(transformed)
    transformed_accuracy = (transformed_predictions == y_test).mean()
    print(f"Accuracy on transformed data: {transformed_accuracy:.4f}")
    print(f"Accuracy retention: {transformed_accuracy / base_accuracy:.4f}")
    print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
