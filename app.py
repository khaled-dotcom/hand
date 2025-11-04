# Set data directory
data_dir = '/kaggle/input/hand-gesture-recognition-dataset-one-hand/Dataset_RGB/Dataset_RGB'

# Process dataset
X, y = process_dataset(data_dir)

# Train model
model, results = train_model(X, y)

# Print results
print(f"Training Score: {results['train_score']:.4f}")
print(f"Testing Score: {results['test_score']:.4f}")
print(f"Cross-validation Scores: {results['cv_scores']}")
print(f"Mean CV Score: {np.mean(results['cv_scores']):.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(results['y_test'], results['y_pred']))

# Plot confusion matrix
plot_confusion_matrix(results['y_test'], results['y_pred'], np.unique(y))

# Save the model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/hand_gesture_classifier.joblib')

# Save feature names
feature_names = [f'landmark_{i}' for i in range(X.shape[1])]
plot_feature_importance(model, feature_names)
