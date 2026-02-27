# Security-Enhanced Data Minimization for GDPR Compliance

## Overview

The repository builds upon the original Goldsteen et al. (2022), an implementation of data minimization. This version includes three key security enhancements to protect user's privacy, and to provide additional GDPR compliant privacy protections. The security enhancements are designed to fill gaps within the original Goldsteen et al. (2022) implementation including: providing formal privacy assurances, mitigating the threat of homogeneous attacks, and sensitivity aware generalizations.

The diff between this fork and the original IBM `ai-privacy-toolkit` repository is available in `changes.diff` at the repository root.

## Original Paper

**Citation:** Goldsteen, A., Ezov, G., Shmelkin, R. et al. Data minimization for GDPR compliance in machine learning models. AI Ethics (2021). https://doi.org/10.1007/s43681-021-00095-8

The original implementation uses knowledge distillation with decision trees to achieve data minimization by reducing the amount of personal data required for machine learning model predictions without compromising prediction accuracy. The enhanced version provides security mechanisms to ensure that privacy is always maintained at a level that does not fall below a certain threshold of acceptability and also protects against previously identified attack vectors.

## Security Features Implemented

### Feature 1: Privacy Budget Management with Minimum Privacy Threshold Enforcement

**File:** `apt/minimization/privacy_budget.py`

**Security Purpose:**
Provides a mechanism that ensures the minimum privacy level (defined as Normalized Certainty Penalty (NCP) scores) remains above a pre-defined minimum threshold. This provides a mechanism to ensure GDPR compliant data minimization practices maintain a minimum level of privacy protection.

**Technical Implementation:**
- **Class:** `PrivacyBudgetTracker`
- **Key Methods:**
  - `check_privacy_threshold(current_ncp)`: Validates that NCP score meets minimum threshold
  - `record_operation(operation, ncp_before, ncp_after)`: Creates audit trail of privacy-affecting operations
  - `get_privacy_margin()`: Calculates remaining privacy budget before threshold violation

**Integration Points:**
- Initialized in `GeneralizeToRepresentative.__init__()` when `min_privacy_threshold > 0.0`
- Called in `fit()` method:
  - After initial generalization 
  - After each tree pruning iteration 
  - Before and after feature removal operations

**Function Call Sequence:**
```
fit()
  → __init__() [initializes PrivacyBudgetTracker if min_privacy_threshold > 0]
  → _calculate_cells()
  → _modify_cells()
  → calculate_ncp() [initial NCP calculation]
  → privacy_budget_tracker.check_privacy_threshold(initial_ncp) [privacy check]
  → (pruning loop)
    → _calculate_level_cells()
    → calculate_ncp() [NCP after pruning]
    → privacy_budget_tracker.check_privacy_threshold(ncp_after) [enforce threshold]
    → privacy_budget_tracker.record_operation() [audit trail]
  → (feature removal loop)
    → calculate_ncp() [NCP before removal]
    → _remove_feature_from_generalization()
    → calculate_ncp() [NCP after removal]
    → privacy_budget_tracker.check_privacy_threshold() [block if violation]
```

**Security Mechanism:**
A tracker keeps a record of all NCP scores. Operations which may compromise privacy such as pruning or removing features from the tracker are not allowed,  doing so would reduce the privacy to less than the minimum required threshold. An operation can be rolled back in order to prevent privacy being reduced regardless of how much it improves the accuracy of the model.

**Usage Example:**
```python
from apt.minimization.minimizer import GeneralizeToRepresentative

# Initialize with minimum privacy threshold of 0.15 (15% information loss required)
gen = GeneralizeToRepresentative(
    estimator=model,
    target_accuracy=0.95,
    min_privacy_threshold=0.15  # Enforce minimum privacy
)
gen.fit(X_train, predictions)
```

---

### Feature 2: K-Anonymity Protection Against Homogeneity Attacks

**File:** `apt/minimization/k_anonymity.py`

**Security Purpose:**
Protection against homogeneity attacks that utilize small groups of data to determine private information about a user. K-anonymity ensures that every generalized record will be "indistinguishable" from at least k-1 additional records so that even when an attacker determines which group a user belongs to they are unable to isolate the specific user or infer sensitive information based upon group membership analysis.

**Technical Implementation:**
- **Class:** `KAnonymityEnforcer`
- **Key Methods:**
  - `enforce_k_anonymity(cells, samples, cells_by_id)`: Main enforcement method that merges violating cells
  - `_count_records_per_cell()`: Counts records matching each cell's constraints
  - `_identify_violations()`: Finds cells with fewer than k records
  - `_merge_violating_cells()`: Merges small cells with similar neighbors

**Integration Points:**
- Initialized in `GeneralizeToRepresentative.__init__()` when `k_anonymity >= 2`
- Called in `fit()` method:
  - After `_modify_cells()` completes (line ~337)
  - After each tree pruning iteration (line ~362)

**Function Call Sequence:**
```
fit()
  → __init__() [initializes KAnonymityEnforcer if k_anonymity >= 2]
  → _calculate_cells()
  → _modify_cells()
  → k_anonymity_enforcer.enforce_k_anonymity() [first enforcement]
    → _count_records_per_cell()
    → _identify_violations()
    → _merge_violating_cells()
      → _find_merge_target()
      → _merge_two_cells()
  → (pruning loop)
    → _calculate_level_cells()
    → k_anonymity_enforcer.enforce_k_anonymity() [maintain k-anonymity after pruning]
```

**Security Mechanism:**
An enforcer monitors cell size and when a cell violates k-anonymity rules, it merges the violating cells with adjacent cells with similar values (adjacent cells in the decision tree) or cells with overlapping value ranges for features. The enforcer ensures that even after the attacker has identified the cluster membership of a record, it is difficult for the attacker to have a high degree of certainty as to the identity of the user or to infer sensitive information based upon the cluster membership.

**Usage Example:**
```python
# Initialize with k-anonymity protection (k=5 means each record is indistinguishable from 4 others)
gen = GeneralizeToRepresentative(
    estimator=model,
    target_accuracy=0.95,
    k_anonymity=5  # Enforce k-anonymity
)
gen.fit(X_train, predictions)
```

---

### Feature 3: Sensitivity-Weighted Generalization with Privacy-Aware Feature Prioritization

**File:** `apt/minimization/sensitivity_weights.py`

**Security Purpose:**
Sensitivity-weighted NCP scores and decision to remove features during generalization, protects sensitive features with more aggressive privacy measures as opposed to non-sensitive features. The purpose is to provide a mechanism to meet specific domain-based privacy constraints (e.g. medical data, financial data), to improve privacy-utility trade-offs, and to focus generalization efforts on the most valuable (high-sensitivity) targets.

**Technical Implementation:**
- **Class:** `SensitivityWeightCalculator`
- **Key Methods:**
  - `calculate_sensitivity_scores(samples, categorical_features)`: Auto-calculates sensitivity based on entropy and distribution
  - `apply_sensitivity_weight(ncp_score, feature)`: Applies sensitivity weighting to NCP scores
  - `prioritize_features_for_removal(features, feature_ncp_scores)`: Sorts features by removal priority (low sensitivity first)
  - `_calculate_categorical_sensitivity()`: Entropy-based sensitivity for categorical features
  - `_calculate_numerical_sensitivity()`: Distribution-based sensitivity for numerical features

**Integration Points:**
- Initialized in `GeneralizeToRepresentative.__init__()` (always enabled)
- Called in `fit()` method:
  - After `_get_feature_data()` to calculate sensitivity scores (line ~315)
  - In `_calc_ncp_for_generalization()` to weight NCP calculations (line ~543, 548)
  - In `_calculate_ncp_for_feature_from_cells()` to weight feature-level NCP (line ~1093)
  - In `_get_feature_to_remove()` to prioritize feature removal (line ~1071)

**Function Call Sequence:**
```
fit()
  → __init__() [initializes SensitivityWeightCalculator]
  → _get_feature_data()
  → sensitivity_calculator.calculate_sensitivity_scores() [calculate sensitivity]
    → _calculate_categorical_sensitivity() [for categorical features]
    → _calculate_numerical_sensitivity() [for numerical features]
  → _calculate_cells()
  → calculate_ncp()
    → _calc_ncp_for_generalization()
      → sensitivity_calculator.apply_sensitivity_weight() [weight each feature's NCP]
  → (feature removal loop)
    → _get_feature_to_remove()
      → _calculate_ncp_for_feature_from_cells()
        → sensitivity_calculator.apply_sensitivity_weight() [weight feature NCP]
      → sensitivity_calculator.prioritize_features_for_removal() [select low-sensitivity features]
```

**Security Mechanism:**
Entropy is used to calculate the sensitivity of categorical features and distribution properties are used to calculate the sensitivity of numerical features. Features with higher sensitivity have higher weighted NCP scores which increases the likelihood that they will be generalized and decreases the likelihood that they will be removed from the process of generalizing a model. If it becomes necessary to increase accuracy in a model by removing features, the least sensitive features will be removed first thereby protecting the sensitive features from privacy loss.

**Usage Example:**
```python
# Manual sensitivity scores (optional - auto-calculation is default)
sensitivity_scores = {
    'age': 0.9,        # Highly sensitive
    'income': 0.85,    # Highly sensitive
    'occupation': 0.6, # Moderately sensitive
    'city': 0.3       # Low sensitivity
}

gen = GeneralizeToRepresentative(
    estimator=model,
    target_accuracy=0.95,
    feature_sensitivity_scores=sensitivity_scores  # Sensitivity weighting
)
gen.fit(X_train, predictions)
```

---

## Combined Usage Example

All three security features can be used together for maximum protection:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from apt.minimization.minimizer import GeneralizeToRepresentative

# Load data
dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2
)

# Train base model
base_model = DecisionTreeClassifier()
base_model.fit(X_train, y_train)
predictions = base_model.predict(X_train)

# Initialize with all security features
gen = GeneralizeToRepresentative(
    estimator=base_model,
    target_accuracy=0.95,
    min_privacy_threshold=0.15,      # Feature 1: Privacy budget enforcement
    k_anonymity=5,                    # Feature 2: K-anonymity protection
    feature_sensitivity_scores={     # Feature 3: Sensitivity weighting
        'age': 0.9,        
        'income': 0.85,
        'occupation': 0.6,
        'city': 0.3
    }
)

# Fit with security features active
gen.fit(X_train, predictions)

# Transform new data
transformed = gen.transform(X_test)

# Check privacy statistics
print(f"Final NCP score: {gen.ncp.fit_score:.4f}")
if gen.privacy_budget_tracker:
    print(f"Privacy margin: {gen.privacy_budget_tracker.get_privacy_margin():.4f}")
if gen.k_anonymity_enforcer:
    stats = gen.k_anonymity_enforcer.get_statistics()
    print(f"K-anonymity violations fixed: {stats['violation_count']}")
```

---

## Running the Code

### Prerequisites

```bash
pip install -r requirements.txt
```

### Downloading datasets

For experiments that use the UCI datasets (Adult, German Credit, Nursery), you must first download the data into the `datasets/` folder. To do this use the helper script `download_datasets.py` from the repository root:

```bash
python download_datasets.py
```

This script calls the dataset utilities in `apt.utils.dataset_utils` and will create:
- `datasets/adult/{train,test}`
- `datasets/german/data`
- `datasets/nursery/data`

Once this has been run once, all notebooks and tests that rely on these datasets will find them locally.

### Run the security features test

To run the security-enhanced data minimization demo (Iris dataset, all three security features, and printed results), execute from the repository root:

```bash
python run_security_features_test.py
```

This script runs the full pipeline with privacy budget tracking, k-anonymity, and sensitivity weighting, and prints a summary including NCP scores, privacy margin, and accuracy retention.

### Basic Implementation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from apt.minimization.minimizer import GeneralizeToRepresentative

# Load and prepare data
dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_train)

# Apply data minimization with security features
gen = GeneralizeToRepresentative(
    estimator=model,
    target_accuracy=0.95,
    min_privacy_threshold=0.1,
    k_anonymity=3
)
gen.fit(X_train, predictions)

# Transform test data
transformed = gen.transform(X_test)
print(f'Transformation complete. NCP score: {gen.ncp.fit_score:.4f}')
```

### Display Functions

The implementation includes extensive print statements that show:
- Privacy budget tracking status and threshold checks
- K-anonymity enforcement operations and violations fixed
- Sensitivity score calculations
- Feature removal prioritization based on sensitivity
- Privacy margin calculations

All output is visible in the terminal during execution.

---

## Security-Related Technical Details

### Privacy Budget Enforcement Mechanism

The privacy budget tracker uses a **hard threshold enforcement** model:
- **Before Operation:** Checks if current NCP meets threshold
- **After Operation:** Validates that operation didn't violate threshold
- **Rollback:** Automatically reverts operations that would violate threshold
- **Audit Trail:** Records all privacy-affecting operations for compliance verification

This ensures **provable privacy guarantees** - organizations can demonstrate that privacy never falls below acceptable levels, addressing GDPR Article 5(1)(c) (data minimization) and Article 25 (data protection by design).

### K-Anonymity Enforcement Strategy

The k-anonymity enforcer uses a **merge-based approach**:
- **Detection:** Identifies cells with fewer than k records
- **Merging:** Combines violating cells with similar neighbors (sibling cells in decision tree)
- **Maintenance:** Re-enforces k-anonymity after each tree pruning operation

This provides **formal privacy guarantees** against homogeneity attacks, where attackers exploit small clusters. The k-anonymity property ensures that even with perfect knowledge of cluster membership, an attacker cannot uniquely identify individuals with probability > 1/k.

### Sensitivity-Weighted NCP Calculation

The sensitivity weighting modifies the standard NCP calculation:
- **Standard NCP:** `ncp = information_loss / total_features`
- **Weighted NCP:** `weighted_ncp = ncp * (1 + sensitivity_score)`

This ensures that:
- High-sensitivity features (e.g., medical conditions, income) receive stronger generalization
- Low-sensitivity features (e.g., public information) can be left less generalized
- Feature removal prioritizes low-sensitivity features, preserving privacy for sensitive data

---

## Code Structure and File Organization

```
apt/minimization/
├── minimizer.py              # Main class (modified to integrate security features)
├── privacy_budget.py        # Feature 1: Privacy budget tracker
├── k_anonymity.py           # Feature 2: K-anonymity enforcer
├── sensitivity_weights.py    # Feature 3: Sensitivity weight calculator
└── __init__.py              # Module exports
```

### Key Modifications to `minimizer.py`

1. Added imports for security modules
2. Added and Initialized security parameters to `__init__`
3. Calculate sensitivity scores in `fit()`
4. Enforce k-anonymity after cell modification
5. Check privacy budget after initial generalization
6. Integrate privacy budget and k-anonymity in pruning loop
7. Integrate privacy budget in feature removal loop
8. Apply sensitivity weighting in NCP calculation
9. Apply sensitivity weighting in feature removal prioritization
10. Apply sensitivity weighting in feature-level NCP calculation

---

## Security Validity and Significance

### Addressing Gaps in State-of-the-Art

1. **Privacy Budget Management:** The original paper mentions privacy thresholds but doesn't provide enforcement mechanisms. Our implementation adds provable privacy guarantees through hard threshold enforcement.

2. **Homogeneity Attack Protection:** The original method creates clusters with homogeneous predictions, making them vulnerable to homogeneity attacks. K-anonymity enforcement addresses this gap by ensuring sufficient anonymity sets.

3. **Sensitivity-Aware Generalization:** The paper mentions sensitivity-weighted NCP as future work (Section 5.1). Our implementation provides a complete solution with automatic sensitivity calculation and integration into the generalization process.

### Comparison with Related Work

- **Pratesi et al. (2018):** Focuses on privacy risk assessment but doesn't provide enforcement mechanisms. Our privacy budget tracker adds enforcement capabilities.

- **Bakker et al. (2020):** Addresses fairness in data collection but doesn't consider feature sensitivity. Our sensitivity weighting complements their work by adding sensitivity-aware prioritization.

- **Standard k-anonymity implementations:** Typically operate on static datasets. Our implementation integrates k-anonymity into the dynamic generalization process, maintaining the property throughout tree pruning operations.

### Security-Specific Validity

All three features address **real security concerns**:

1. **Privacy Budget:** Prevents accidental privacy degradation during optimization, ensuring GDPR compliance.
2. **K-Anonymity:** Protects against homogeneity attacks, a known vulnerability in clustering-based anonymization.
3. **Sensitivity Weighting:** Enables domain-specific privacy requirements (medical, financial data) while optimizing utility.

The implementation provides **provable guarantees** (privacy threshold enforcement, k-anonymity property) and **practical mechanisms** (sensitivity weighting, audit trails) for compliance verification.

---

## References

1. Goldsteen, A., Ezov, G., Shmelkin, R. et al. Data minimization for GDPR compliance in machine learning models. AI Ethics (2021). https://doi.org/10.1007/s43681-021-00095-8

2. Pratesi, F., Monreale, A., Trasarti, R., Giannotti, F., Pedreschi, D., Yanagihara, T.: Prudence: a system for assessing privacy risk vs utility in data sharing ecosystems. Trans. Data Privacy 11(2) (2018)

3. Bakker, M.A., Riverón Valdés, H., Tu, D.P., Gummadi, K.P., Varshney, K.R., Weller, A., Pentland, A.: Fair enough: improving fairness in budget-constrained decision making using confidence thresholds. In: Proceedings of the Workshop on Artificial Intelligence Safety (2020)

4. Sweeney, L.: k-anonymity: a model for protecting privacy. Int. J. Uncertain. Fuzz. Knowl.-Based Syst. 10, 557–570 (2002)

5. Ghinita, G., Karras, P., Kalnis, P., Mamoulis, N.: Fast data anonymization with low information loss. In: Proceedings of the 33rd International Conference on Very Large Data Bases (2007)

---

## License

This implementation extends the original ai-privacy-toolkit codebase. Please refer to the original repository for license information. https://github.com/IBM/ai-privacy-toolkit.git
