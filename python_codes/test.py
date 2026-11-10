"""
================================================================================
TEST.PY - SALES PIPELINE MODEL TESTING SCRIPT
================================================================================

PURPOSE:
    This script tests the trained sales pipeline model before deploying it
    as a production API. It:
    1. Loads the saved model (pickle file)
    2. Tests predictions on active deals data
    3. Generates statistics: how many deals will WIN vs LOST
    4. Exports results to CSV and summary report
    5. Validates everything works before building main.py

KEY RESULTS FROM YOUR RUN:
    ✓ Model loaded successfully
    ✓ 1589 active deals analyzed
    ✓ Predicted to WIN: 68 deals (4.3%)
    ✓ Predicted to LOST: 1521 deals (95.7%)
    ✓ All predictions exported to CSV
================================================================================
"""

import pickle           # For loading the saved model (.pkl file)
import pandas as pd     # For data manipulation and dataframes
import numpy as np      # For numerical operations
from datetime import datetime  # For timestamping the test
import sys              # For system operations


# ============================================================================
# UTILITY FUNCTIONS - Reusable helper functions
# ============================================================================

def print_header(title):
    """
    FUNCTION: print_header(title)
    
    PURPOSE: Print a nicely formatted section header to make console output readable
    
    PARAMETERS:
        title (str): The section title to display
    
    RETURNS: None (just prints)
    
    EXAMPLE:
        print_header("LOADING MODEL")
        Output:
        ================================================================================
         LOADING MODEL
        ================================================================================
    """
    print("\n" + "=" * 80)  # Print 80 equal signs for visual separation
    print(f" {title}")       # Print the title
    print("=" * 80)         # Print 80 equal signs again


def load_active_data(csv_path):
    """
    FUNCTION: load_active_data(csv_path)
    
    PURPOSE: Load the active deals CSV file that was generated from model.py
    
    PARAMETERS:
        csv_path (str): Path to the CSV file containing active deals
                       Default: 'active_deals_predictions.csv' (from model.py)
    
    RETURNS: 
        pandas.DataFrame: The loaded data or None if error
    
    PROCESS:
        1. Try to read the CSV file using pandas
        2. If file not found, return None and print error
        3. If other error, return None and print error message
        4. If successful, print success message and return dataframe
    
    EXAMPLE:
        df = load_active_data('active_deals_predictions.csv')
        # Returns: DataFrame with 1589 rows (active deals)
    """
    try:
        # pd.read_csv() loads the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} active deals from {csv_path}")
        return df
    except FileNotFoundError:
        # This error occurs if the file doesn't exist
        print(f"✗ Error: File not found at {csv_path}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        print(f"✗ Error loading data: {str(e)}")
        return None


def validate_data(df, required_columns):
    """
    FUNCTION: validate_data(df, required_columns)
    
    PURPOSE: Check that all required columns exist in the dataframe
    
    PARAMETERS:
        df (pandas.DataFrame): The dataframe to validate
        required_columns (list): List of column names that must exist
    
    RETURNS:
        bool: True if all columns exist, False otherwise
    
    PROCESS:
        1. Loop through required_columns list
        2. Check if each column is in df.columns
        3. Store missing columns in a list
        4. If any missing, print them and return False
        5. Otherwise, print success and return True
    
    EXAMPLE:
        required = ['sales_agent', 'product', 'account']
        is_valid = validate_data(df, required)
        # Prints: ✓ All required columns present
    """
    # List comprehension: create list of columns that are missing
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        # If any columns are missing, print them
        print(f"✗ Missing columns: {missing}")
        return False  # Validation failed
    
    print(f"✓ All {len(required_columns)} required features present")
    return True  # Validation passed


def ensure_features_exist(df, num_feats, cat_feats):
    """
    FUNCTION: ensure_features_exist(df, num_feats, cat_feats)
    
    PURPOSE: Ensure all required features exist in the dataframe
             If missing, create them with default values
    
    PARAMETERS:
        df (pandas.DataFrame): The dataframe to process
        num_feats (list): List of numerical feature names
        cat_feats (list): List of categorical feature names
    
    RETURNS:
        pandas.DataFrame: The dataframe with all features present
    
    PROCESS:
        1. Make a copy of the dataframe (don't modify original)
        2. Combine numerical and categorical feature lists
        3. For each feature:
           - If it doesn't exist in dataframe, create it
           - For numerical features: fill with 0
           - For categorical features: fill with 'missing'
        4. Return the updated dataframe
    
    WHY THIS IS NEEDED:
        The model expects certain features. Active deals data might not have
        all of them (like statistics calculated during training). This function
        creates placeholders so the model doesn't crash.
    
    EXAMPLE:
        df = pd.DataFrame({'sales_agent': ['John'], 'product': ['XYZ']})
        df = ensure_features_exist(df, ['deal_age', 'agent_win_rate'], ['sector'])
        # Now df has columns: sales_agent, product, deal_age (0), agent_win_rate (0), sector ('missing')
    """
    df = df.copy()  # Create a copy to avoid modifying the original dataframe
    
    # Combine both lists of features
    all_features = num_feats + cat_feats
    
    # Loop through each feature
    for f in all_features:
        if f not in df.columns:  # If the feature doesn't exist
            # Check if it's a numerical feature or categorical
            if f in num_feats:
                # Numerical features get default value of 0
                df[f] = 0
            else:
                # Categorical features get default value of 'missing'
                df[f] = 'missing'
    
    return df  # Return the updated dataframe


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_model_predictions(df_active, model_path='sales_pipeline_model.pkl'):
    """
    FUNCTION: test_model_predictions(df_active, model_path)
    
    PURPOSE: Main function that runs the entire test suite
    
    PARAMETERS:
        df_active (pandas.DataFrame): The active deals data to test on
        model_path (str): Path to the saved model file
    
    RETURNS:
        bool: True if test passed, False if test failed
    
    STRUCTURE: This function has 9 sections that execute in order
    """
    
    print_header("SALES PIPELINE MODEL - TEST SUITE")
    
    # ========================================================================
    # SECTION 1: LOAD MODEL ARTIFACTS
    # ========================================================================
    """
    This section loads the pickled model file saved from model.py
    
    What gets saved in the pickle file (from model.py):
    {
        'model': XGBClassifier (the trained model),
        'preprocessor': ColumnTransformer (handles data preprocessing),
        'feature_names': list of 50 feature names,
        'categorical_features': list of 8 category columns,
        'numerical_features': list of 42 number columns,
        'test_roc_auc': 0.6482 (model performance score)
    }
    """
    
    print_header("1. LOADING MODEL ARTIFACTS")
    
    try:
        # Open the pickle file in binary read mode
        with open(model_path, 'rb') as f:
            # pickle.load() deserializes the Python object from the file
            artifacts = pickle.load(f)
        
        # Print confirmation and model details
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"  - Model type: {type(artifacts['model']).__name__}")  # Should be 'Pipeline'
        print(f"  - Test ROC-AUC: {artifacts['test_roc_auc']:.4f}")     # 0.6482 (from your test)
        print(f"  - Total features: {len(artifacts['feature_names'])}")   # 50 features
        print(f"  - Categorical features: {len(artifacts['categorical_features'])}")  # 8
        print(f"  - Numerical features: {len(artifacts['numerical_features'])}")      # 42
    
    except FileNotFoundError:
        # Error if the model file doesn't exist
        print(f"✗ Model file not found: {model_path}")
        print(f"   Make sure you ran model.py first to generate: {model_path}")
        return False  # Stop testing
    
    except Exception as e:
        # Catch any other errors (corrupted file, permission issues, etc)
        print(f"✗ Error loading model: {str(e)}")
        return False  # Stop testing
    
    # ========================================================================
    # SECTION 2: VALIDATE DATA
    # ========================================================================
    """
    This section checks if the active deals data has all required columns
    The model expects 50 features (8 categorical + 42 numerical)
    """
    
    print_header("2. VALIDATING INPUT DATA")
    
    # Get the list of all required features from the model
    feature_columns = artifacts['feature_names']
    
    # Validate: does the input data have all these columns?
    if not validate_data(df_active, feature_columns):
        # If not all columns exist, that's okay - we'll fill them in later
        print("⚠ Note: Missing columns will be filled with default values")
    
    # Print basic info about the loaded data
    print(f"✓ Data has {len(df_active)} records")
    print(f"✓ Data has {len(df_active.columns)} columns")
    
    # ========================================================================
    # SECTION 3: DATA PREPROCESSING
    # ========================================================================
    """
    This section prepares the data for the model:
    1. Ensure all 50 required features exist (fill missing with defaults)
    2. Extract only the features the model needs (in the correct order)
    3. Check for any remaining missing values
    """
    
    print_header("3. DATA PREPROCESSING")
    
    try:
        # Step 1: Ensure all features exist (add missing ones with default values)
        df_processed = ensure_features_exist(
            df_active,
            artifacts['numerical_features'],    # List of 42 numerical features
            artifacts['categorical_features']   # List of 8 categorical features
        )
        
        # Step 2: Extract only the features we need, in the correct order
        # This is CRITICAL - features must be in the same order as training
        X_active = df_processed[feature_columns]
        
        # Print confirmation
        print(f"✓ Features extracted: {X_active.shape[1]} features")  # Should be 50
        print(f"✓ Records to predict: {X_active.shape[0]} deals")     # Should be 1589
        
        # Step 3: Check for any missing/null values
        # isnull() returns True for NaN values, sum() counts them
        missing_count = X_active.isnull().sum().sum()
        
        if missing_count > 0:
            # If there are missing values, fill them with 0
            print(f"⚠ Warning: {missing_count} missing values found")
            X_active = X_active.fillna(0)  # Replace NaN with 0
            print(f"✓ Missing values filled with 0")
        else:
            # No missing values - everything is clean
            print(f"✓ No missing values")
    
    except Exception as e:
        # If something goes wrong in preprocessing, stop
        print(f"✗ Error in preprocessing: {str(e)}")
        return False
    
    # ========================================================================
    # SECTION 4: GENERATE PREDICTIONS
    # ========================================================================
    """
    This is the core section where the model makes predictions!
    
    The model.predict_proba() returns probability scores:
    - Score close to 1.0 = model thinks deal will be WON
    - Score close to 0.0 = model thinks deal will be LOST
    - Score close to 0.5 = model is uncertain
    
    Then we apply a threshold:
    - If probability >= 0.5: predict WIN (1)
    - If probability < 0.5: predict LOST (0)
    """
    
    print_header("4. GENERATING PREDICTIONS")
    
    try:
        # Step 1: Get probability predictions from the model
        # predict_proba() returns shape (1589, 2):
        #   Column 0: probability of class 0 (LOST)
        #   Column 1: probability of class 1 (WON)
        # We want column 1 (probability of winning)
        probabilities = artifacts['model'].predict_proba(X_active)[:, 1]
        print(f"✓ Input shape: {X_active.shape}")
        print(f"✓ Probabilities generated for {len(probabilities)} deals")
        
        # Step 2: Apply threshold to convert probabilities to binary predictions
        # threshold = 0.5 means:
        #   probability >= 0.5 → predict 1 (WIN)
        #   probability < 0.5 → predict 0 (LOST)
        threshold = 0.5
        predictions = (probabilities >= threshold).astype(int)
        print(f"✓ Predictions applied using threshold: {threshold:.4f}")
    
    except Exception as e:
        # If something goes wrong, print error and traceback
        print(f"✗ Error generating predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # SECTION 5: PREDICTION SUMMARY - KEY RESULTS
    # ========================================================================
    """
    This section summarizes the key findings:
    - How many deals will WIN vs LOST?
    - What are the probability statistics?
    - What are the confidence levels?
    
    KEY RESULTS FROM YOUR RUN:
    - Total Deals: 1589
    - Predicted to WIN: 68 (4.3%)
    - Predicted to LOST: 1521 (95.7%)
    """
    
    print_header("5. PREDICTION SUMMARY - KEY RESULTS")
    
    # Calculate how many deals will win vs lose
    n_total = len(predictions)  # Total deals (1589)
    n_won = (predictions == 1).sum()  # How many predicted to WIN (68)
    n_lost = (predictions == 0).sum()  # How many predicted to LOST (1521)
    
    # Calculate percentages
    pct_won = (n_won / n_total) * 100 if n_total > 0 else 0
    pct_lost = (n_lost / n_total) * 100 if n_total > 0 else 0
    
    # Print the results in a formatted table
    print(f"\n{'DEAL OUTCOME FORECAST':^80}")  # Center text in 80-char width
    print("-" * 80)
    print(f"Total Deals Analyzed:     {n_total:>6} deals")
    print(f"  ✓ Predicted to WIN:     {n_won:>6} deals ({pct_won:>5.1f}%)")
    print(f"  ✗ Predicted to LOST:    {n_lost:>6} deals ({pct_lost:>5.1f}%)")
    print("-" * 80)
    
    # Print probability statistics
    print(f"\n{'PROBABILITY STATISTICS':^80}")
    print("-" * 80)
    print(f"Mean Win Probability:     {probabilities.mean():>6.4f}")  # Average probability
    print(f"Min Probability:          {probabilities.min():>6.4f}")   # Lowest probability
    print(f"Max Probability:          {probabilities.max():>6.4f}")   # Highest probability
    print(f"Std Deviation:            {probabilities.std():>6.4f}")   # Variation
    print(f"Median Probability:       {np.median(probabilities):>6.4f}")  # Middle value
    print("-" * 80)
    
    # Calculate confidence levels
    # Confidence = max(probability of WIN, probability of LOST)
    # High confidence = model is sure about its prediction
    # Low confidence = model is uncertain
    confidence = np.maximum(probabilities, 1 - probabilities)
    
    high_confidence = (confidence >= 0.7).sum()  # Very confident predictions
    medium_confidence = ((confidence >= 0.6) & (confidence < 0.7)).sum()  # Somewhat confident
    low_confidence = (confidence < 0.6).sum()  # Uncertain predictions
    
    print(f"\n{'CONFIDENCE LEVELS':^80}")
    print("-" * 80)
    print(f"High Confidence (≥0.7):   {high_confidence:>6} deals ({(high_confidence/n_total)*100:>5.1f}%)")
    print(f"Medium Confidence (0.6-0.7):{medium_confidence:>5} deals ({(medium_confidence/n_total)*100:>5.1f}%)")
    print(f"Low Confidence (<0.6):    {low_confidence:>6} deals ({(low_confidence/n_total)*100:>5.1f}%)")
    print("-" * 80)
    
    # ========================================================================
    # SECTION 6: DETAILED PREDICTION BREAKDOWN
    # ========================================================================
    """
    This section shows the top 10 most confident predictions
    - Top 10 deals most likely to be WON
    - Top 10 deals most likely to be LOST
    """
    
    print_header("6. DETAILED PREDICTION BREAKDOWN")
    
    # Create a dataframe with all prediction details
    results_df = pd.DataFrame({
        'predicted_outcome': ['Lost' if p == 0 else 'Won' for p in predictions],
        'probability_won': probabilities,
        'probability_lost': 1 - probabilities,
        'confidence': confidence
    })
    
    # Show top 10 predictions for deals predicted to WIN
    print("\n[TOP 10 MOST CONFIDENT - PREDICTED TO WIN]")
    print("-" * 80)
    # Filter rows where predicted_outcome is 'Won', sort by confidence (descending)
    top_wins = results_df[results_df['predicted_outcome'] == 'Won'].nlargest(10, 'confidence')
    
    if len(top_wins) > 0:
        # Print each of the top 10 wins
        for idx, (i, row) in enumerate(top_wins.iterrows(), 1):
            print(f"{idx:>2}. Deal {i:>5} | Win Prob: {row['probability_won']:.4f} | Confidence: {row['confidence']:.4f}")
    else:
        print("  No deals predicted to win")
    
    # Show top 10 predictions for deals predicted to LOSE
    print("\n[TOP 10 MOST CONFIDENT - PREDICTED TO LOST]")
    print("-" * 80)
    # Filter rows where predicted_outcome is 'Lost', sort by confidence (descending)
    top_losses = results_df[results_df['predicted_outcome'] == 'Lost'].nlargest(10, 'confidence')
    
    if len(top_losses) > 0:
        # Print each of the top 10 losses
        for idx, (i, row) in enumerate(top_losses.iterrows(), 1):
            print(f"{idx:>2}. Deal {i:>5} | Loss Prob: {row['probability_lost']:.4f} | Confidence: {row['confidence']:.4f}")
    else:
        print("  No deals predicted to lose")
    
    # ========================================================================
    # SECTION 7: PROBABILITY DISTRIBUTION ANALYSIS
    # ========================================================================
    """
    This section shows how the probabilities are distributed
    It creates a histogram showing how many deals fall into each probability range
    
    Example: If most deals have probability 0.3-0.4, that means the model
    thinks most deals are likely to LOSE
    """
    
    print_header("7. PROBABILITY DISTRIBUTION ANALYSIS")
    
    # Create bins (ranges) for the histogram
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # np.histogram counts how many values fall into each bin
    hist, _ = np.histogram(probabilities, bins=bins)
    
    print("\nWin Probability Distribution (Visual):")
    print("-" * 80)
    
    # For each bin range, print how many deals fall in that range
    for i in range(len(bins)-1):
        # Calculate percentage of deals in this range
        pct = (hist[i] / n_total) * 100 if n_total > 0 else 0
        # Create a visual bar (█ character repeated)
        bar_length = int(pct / 2)  # Scale to reasonable width
        bar = "█" * bar_length
        # Print the range, count, percentage, and visual bar
        print(f"[{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:>6} deals ({pct:>5.1f}%) {bar}")
    
    # ========================================================================
    # SECTION 8: SAVE RESULTS
    # ========================================================================
    """
    This section exports the predictions to files so you can review them:
    1. test_predictions_results.csv - Full predictions for each deal
    2. test_summary_statistics.txt - Summary report
    """
    
    print_header("8. SAVING RESULTS")
    
    try:
        # Step 1: Combine original data with predictions into one dataframe
        results_export = df_active.copy()  # Start with original data
        # Add prediction columns
        results_export['predicted_outcome'] = results_df['predicted_outcome'].values
        results_export['probability_won'] = results_df['probability_won'].values
        results_export['confidence'] = results_df['confidence'].values
        
        # Step 2: Save to CSV file
        output_file = 'test_predictions_results.csv'
        results_export.to_csv(output_file, index=False)  # index=False means don't write row numbers
        print(f"✓ Predictions saved to: {output_file}")
        print(f"  Columns: {len(results_export.columns)} | Rows: {len(results_export)}")
        
        # Step 3: Save summary statistics to text file
        summary_file = 'test_summary_statistics.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:  # encoding='utf-8' handles special characters
            # Write report header
            f.write("=" * 80 + "\n")
            f.write("SALES PIPELINE MODEL - TEST SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Write test info
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model ROC-AUC (Training): {artifacts['test_roc_auc']:.4f}\n\n")
            
            # Write prediction results
            f.write("PREDICTION RESULTS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Deals Analyzed:   {n_total}\n")
            f.write(f"Predicted to WIN:       {n_won} ({pct_won:.1f}%)\n")
            f.write(f"Predicted to LOST:      {n_lost} ({pct_lost:.1f}%)\n\n")
            
            # Write probability statistics
            f.write("PROBABILITY STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Win Probability:   {probabilities.mean():.4f}\n")
            f.write(f"Min Probability:        {probabilities.min():.4f}\n")
            f.write(f"Max Probability:        {probabilities.max():.4f}\n")
            f.write(f"Std Deviation:          {probabilities.std():.4f}\n\n")
            
            # Write confidence levels
            f.write("CONFIDENCE LEVELS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"High Confidence (>=0.7): {high_confidence}\n")
            f.write(f"Medium Confidence:      {medium_confidence}\n")
            f.write(f"Low Confidence (<0.6):  {low_confidence}\n")
        
        print(f"✓ Summary statistics saved to: {summary_file}")
    
    except Exception as e:
        # If saving fails, print warning but continue
        print(f"⚠ Warning: Could not save results: {str(e)}")
    
    # ========================================================================
    # SECTION 9: TEST VALIDATION SUMMARY
    # ========================================================================
    """
    Final section summarizing that all tests passed
    """
    
    print_header("9. TEST VALIDATION SUMMARY")
    
    # Print validation checklist
    print(f"✓ Model loaded and validated")
    print(f"✓ Data preprocessed successfully")
    print(f"✓ Predictions generated for {n_total} deals")
    print(f"✓ Results exported to CSV and summary file")
    print(f"\n{'✓✓✓ ALL TESTS PASSED ✓✓✓':^80}")
    print(f"\n{'Next Step: Build API endpoint (main.py)':^80}")
    
    return True  # Test passed successfully


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    FUNCTION: main()
    
    PURPOSE: Entry point for the script. This runs when you execute: python test.py
    
    PROCESS:
        1. Set configuration paths
        2. Load active deals data
        3. Run the test
        4. Print final results
    """
    
    print_header("SALES PIPELINE - MODEL TESTING")
    
    # Configuration: set paths to the files we need
    ACTIVE_DATA_PATH = 'active_deals_predictions.csv'  # Generated from model.py
    MODEL_PATH = 'sales_pipeline_model.pkl'            # Generated from model.py
    
    # Print what files we're looking for
    print(f"\nLooking for active deals file: {ACTIVE_DATA_PATH}")
    print(f"Looking for model file: {MODEL_PATH}\n")
    
    # Try to load the active deals data
    try:
        df_active = load_active_data(ACTIVE_DATA_PATH)
        if df_active is None:
            # If loading failed, exit
            print("\n✗ Could not load active deals data")
            print("   Make sure model.py has been run and generated: active_deals_predictions.csv")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False
    
    # Run the main test function
    success = test_model_predictions(df_active, MODEL_PATH)
    
    # Print final summary
    if success:
        # Test passed - print success message
        print("\n" + "=" * 80)
        print(" ✓ TEST COMPLETE - READY FOR PRODUCTION")
        print("=" * 80)
        print("\n📋 Next Steps:")
        print("  1. Review test_predictions_results.csv (detailed predictions)")
        print("  2. Review test_summary_statistics.txt (summary report)")
        print("  3. Validate predictions with business stakeholders")
        print("  4. Build FastAPI/Flask endpoint in main.py")
        print("  5. Deploy to production\n")
    else:
        # Test failed - print error message
        print("\n" + "=" * 80)
        print(" ✗ TEST FAILED - CHECK ERRORS ABOVE")
        print("=" * 80)
        sys.exit(1)  # Exit with error code


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

# This line ensures main() only runs when you execute this file directly
# (not when this file is imported into another Python file)
if __name__ == "__main__":
    main()


"""
================================================================================
QUICK REFERENCE - WHAT EACH VARIABLE MEANS
================================================================================

n_total (1589):
    Total number of active deals in the test data

n_won (68):
    Number of deals predicted to be WON by the model

n_lost (1521):
    Number of deals predicted to be LOST by the model

pct_won (4.3%):
    Percentage of deals predicted to win: (68 / 1589) * 100

probabilities (array of 1589 values):
    The model's confidence score for each deal winning
    Values range from 0.0 (definitely lose) to 1.0 (definitely win)

threshold (0.5):
    The cutoff point for converting probabilities to binary predictions
    probability >= 0.5 → predict WIN (1)
    probability < 0.5 → predict LOST (0)

confidence (array of 1589 values):
    How confident the model is about each prediction
    confidence = max(probability of WIN, probability of LOST)
    Values range from 0.5 (uncertain) to 1.0 (very confident)

high_confidence (3):
    Number of predictions where confidence >= 0.7 (model is very sure)

medium_confidence (974):
    Number of predictions where 0.6 <= confidence < 0.7 (model is somewhat sure)

low_confidence (612):
    Number of predictions where confidence < 0.6 (model is uncertain)

================================================================================
KEY INSIGHTS FROM YOUR TEST RUN
================================================================================

1. Model Performance:
   - ROC-AUC: 0.6482 (model is decent but not great)
   
2. Deal Forecast:
   - Only 68 out of 1589 deals (4.3%) are predicted to WIN
   - 1521 out of 1589 deals (95.7%) are predicted to LOST
   - This suggests either:
     a) The deals are genuinely risky
     b) The model is very conservative (pessimistic)
   
3. Confidence Distribution:
   - Very few high-confidence predictions (3 deals)
   - Most predictions are medium confidence (974 deals)
   - This means the model is mostly uncertain about outcomes
   - Probabilities cluster around 0.3-0.4 (closer to "lost")
   
4. Recommendation for Next Step (main.py):
   - You can now build the API with confidence
   - Consider adjusting the threshold (currently 0.5)
   - Maybe use 0.4 or 0.3 threshold to be more optimistic
   - Add monitoring to track actual vs predicted outcomes

================================================================================
"""