# -Neural-XGBoost-A-Hybrid-Approach-for-Disaster-Prediction-and-Management-Using-Machine-Learning
Disaster Type Balancing with SMOTE
Overview
This project analyzes a disaster events dataset and addresses class imbalance in the disaster type labels using Synthetic Minority Over-sampling Technique (SMOTE). The goal is to prepare balanced training data for machine learning models to avoid bias towards majority disaster categories.

Dataset
The data file emdat_data.csv contains detailed records of various natural and technological disasters in Asia from 2000 to 2002.

Key column used: Disaster Type which categorizes the type of disaster (e.g., Flood, Storm, Earthquake, etc.).

Data Preprocessing
Missing values in numerical columns are imputed with the median.

Missing values in categorical columns are imputed with the most frequent category.

Categorical features are label encoded to convert string labels into numeric form.

Numerical features are standardized using StandardScaler.

The target labels Disaster Type are also label encoded.

Train-Test Split
The dataset is split into training (70%) and test (30%) sets using stratified sampling to preserve the original class distribution in the test set.

Balancing using SMOTE
SMOTE is applied only to the training set to synthetically oversample minority disaster types.

This ensures the training data is balanced and suitable for model training.

The test set remains untouched to provide a realistic evaluation environment.

Visualizations
Bar plots show disaster type counts before and after SMOTE on the training data for comparative analysis.

Usage
To run the preprocessing and balancing pipeline, execute the Python scripts or notebook cells in the order:

Data loading and cleaning

Feature encoding and scaling

Train-test split with stratification

Apply SMOTE to train data

Visualize before/after class distributions

Requirements
Python 3.x

pandas

numpy

scikit-learn

imbalanced-learn (for SMOTE)

matplotlib

seaborn

Notes
Adjust k_neighbors parameter in SMOTE if very small classes cause errors.

For mixed data types, consider SMOTE variants like SMOTE-NC.

Handle ultra-rare classes carefully by merging or excluding if needed.
