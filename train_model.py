import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv('kaggle_asl_dataset.csv')

# --- NEW FILTERING STEP ---
# Count how many rows exist for each label
label_counts = df.iloc[:, 0].value_counts()
# Keep only labels that have at least 10 examples
valid_labels = label_counts[label_counts >= 10].index
df = df[df.iloc[:, 0].isin(valid_labels)]
print(f"Filtered out incomplete classes. Remaining classes: {list(valid_labels)}")
# --------------------------

X = df.iloc[:, 1:].values  # The 63 coordinate columns
y = df.iloc[:, 0].values   # The label column (A, B, C, etc.)

# 2. Normalization Function
def normalize_landmarks(landmarks):
    # Reshape the flat 63-element list into 21 rows of (x, y, z)
    coords = np.array(landmarks).reshape(21, 3)
    
    # Translation invariance: Shift all points so the wrist (index 0) becomes (0, 0, 0)
    base_coord = coords[0]
    coords = coords - base_coord
    
    # Scale invariance: Normalize hand size by dividing by the maximum coordinate value
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
        
    return coords.flatten()

# 3. Apply normalization to the entire dataset
print("Normalizing coordinates...")
X_normalized = np.array([normalize_landmarks(row) for row in X])

print("Generating mirrored (left-handed) dataset...")
X_mirrored = []
for row in X_normalized:
    # Reshape back to 21x3
    coords = row.copy().reshape(21, 3)
    # Multiply all X coordinates by -1 to mirror the hand horizontally
    coords[:, 0] = coords[:, 0] * -1
    X_mirrored.append(coords.flatten())

# Combine original (Right-Handed) and mirrored (Left-Handed) data
X_combined = np.vstack((X_normalized, X_mirrored))
y_combined = np.concatenate((y, y))
print(f"Dataset doubled to {len(X_combined)} rows for ambidextrous support.")


# 4. Split data: 80% for training, 20% for testing
# X_train, X_test, y_train, y_test = train_test_split(
#     X_normalized, y, test_size=0.2, random_state=42, stratify=y
# )
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)


# 5. Train the Random Forest
print("Training the Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate accuracy on the unseen 20% test data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 7. Save the trained model to disk
with open('kaggle_asl_model_ambi.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Model saved successfully as 'kaggle_asl_model_ambi.pkl'.")