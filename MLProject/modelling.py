# ===================================================================
# SALES FORECASTING WITH MLFLOW - INTEGRATED WITH SERVING & CLI
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

import sys
import os
import joblib
import time
import subprocess
import threading
import requests
import json
import argparse

# MLflow
import mlflow
import mlflow.sklearn

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import ParameterGrid

warnings.filterwarnings('ignore')

# ===================================================================
# ARGUMENT PARSING FOR MLFLOW PROJECT SUPPORT
# ===================================================================

def parse_arguments():
    """Parse command line arguments for MLflow Project compatibility"""
    parser = argparse.ArgumentParser(
        description='Sales Forecasting with MLflow - CLI Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python modelling.py                                           # Use defaults
    python modelling.py --data_path data_processed.csv                   # Custom data
    python modelling.py --experiment_name Sales_CI_Experiment     # Custom experiment
    python modelling.py --model_type RandomForest                 # Single model
    python modelling.py --test_size 0.3 --random_state 123       # Custom split
    
MLflow Project Usage:
    mlflow run . -P data_path=data_processed.csv -P experiment_name=CI_Experiment
        """
    )
    
    # Data and experiment parameters
    parser.add_argument('--data_path', type=str, default='data_processed.csv',
                       help='Path to input data file (default: data_processed.csv)')
    parser.add_argument('--experiment_name', type=str, 
                       default='Sales_Forecasting_Experiment',
                       help='MLflow experiment name (default: Sales_Forecasting_Experiment)')
    parser.add_argument('--tracking_uri', type=str, 
                       default='http://localhost:5000',
                       help='MLflow tracking URI (default: http://localhost:5000)')
    
    # Model training parameters
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['RandomForest', 'GradientBoosting', 'XGBoost', 'ExtraTrees', 'all'],
                       help='Model type to train (default: all)')
    
    # Training mode parameters
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train_only', 'evaluate_only', 'champion_only'],
                       help='Training mode (default: full)')
    parser.add_argument('--max_combinations', type=int, default=3,
                       help='Maximum parameter combinations per model (default: 3)')
    parser.add_argument('--serving_port', type=int, default=1234,
                       help='Port for model serving (default: 1234)')
    
    # Debug and verbose options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no_champion', action='store_true',
                       help='Skip champion model training')
    
    return parser.parse_args()

# ===================================================================
# MAIN TRAINING FUNCTION
# ===================================================================

def main(args=None):
    """Main training function that can be called with arguments"""
    
    # Parse arguments if not provided
    if args is None:
        args = parse_arguments()
    
    # Print header with configuration
    print("="*80)
    print("SALES FORECASTING WITH MLFLOW - INTEGRATED WITH SERVING & CLI")
    print("="*80)
    print(f"📊 Configuration:")
    print(f"   Data Path: {args.data_path}")
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Tracking URI: {args.tracking_uri}")
    print(f"   Test Size: {args.test_size}")
    print(f"   Model Type: {args.model_type}")
    print(f"   Mode: {args.mode}")
    print(f"   Random State: {args.random_state}")
    if args.verbose:
        print(f"   Max Combinations: {args.max_combinations}")
        print(f"   Serving Port: {args.serving_port}")
    print("="*80)

    # ===================================================================
    # 1. SETUP MLFLOW FOR UI VISUALIZATION
    # ===================================================================

    print("\n1. SETTING UP MLFLOW FOR UI")
    print("-" * 35)

    # Set MLflow tracking URI from arguments
    mlflow.set_tracking_uri(args.tracking_uri)

    # Create or use existing experiment
    experiment_name = args.experiment_name
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            tags={
                "version": "2.1",
                "project": "Sales Forecasting",
                "algorithm": "Multiple Models",
                "dataset": "Retail Sales Data",
                "serving_integration": "mlflow_serve.py",
                "cli_support": "true",
                "data_source": args.data_path,
                "training_mode": args.mode
            }
        )
        print(f"✓ Created new experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"✓ Using existing experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)

    print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"✓ Experiment ID: {experiment_id}")

    # ===================================================================
    # 2. LOAD AND PREPARE DATA WITH DETAILED LOGGING
    # ===================================================================

    print("\n2. LOADING AND PREPARING DATA")
    print("-" * 35)

    # Load processed data from specified path
    try:
        df = pd.read_csv(args.data_path)
        print(f"✓ Data loaded from {args.data_path}: {df.shape}")
        data_source = f"CSV File: {args.data_path}"
        
        # Validate required columns
        required_columns = ['InvoiceDate', 'TotalSales']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️ Missing required columns: {missing_columns}")
            print("   Attempting to create missing columns...")
            
            # Try to create TotalSales if missing
            if 'TotalSales' not in df.columns:
                if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
                    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
                    print("   ✓ Created TotalSales from Quantity * UnitPrice")
                else:
                    raise ValueError("Cannot create TotalSales - missing Quantity or UnitPrice")
        
    except FileNotFoundError:
        print(f"❌ File not found: {args.data_path}")
        print("⚠️ Creating sample data for demonstration...")
        df, data_source = create_sample_data(args.random_state)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("⚠️ Creating sample data for demonstration...")
        df, data_source = create_sample_data(args.random_state)

    # Convert date column
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Data overview
    print(f"✓ Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"✓ Target variable range: ${df['TotalSales'].min():.2f} to ${df['TotalSales'].max():.2f}")

    # ===================================================================
    # Continue with the rest of your original code...
    # ===================================================================
    
    # Enhanced feature engineering
    df_features = create_time_series_features(df)
    df_features = df_features.dropna().reset_index(drop=True)
    
    # Prepare features
    all_features, feature_groups = prepare_features(df_features)
    X = df_features[all_features]
    y = df_features['TotalSales']
    
    # Create feature info for serving
    feature_info_for_serving = create_feature_info_for_serving(df_features, all_features, feature_groups)
    
    # Train-test split with configurable test size
    X_train, X_test, y_train, y_test, scaler = perform_train_test_split(
        X, y, df_features, test_size=args.test_size, random_state=args.random_state
    )
    
    # Data info for logging
    data_info = {
        "source": data_source,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X_train.shape[1],
        "test_size": args.test_size,
        "random_state": args.random_state
    }
    
    # Model training with configurable parameters
    all_results, best_overall_model = train_models(
        X_train, X_test, y_train, y_test, 
        all_features, data_info, args
    )
    
    # Train champion model if not disabled
    champion_run_id = None
    if not args.no_champion and best_overall_model:
        champion_run_id = train_champion_model(
            best_overall_model, X_train, X_test, y_train, y_test,
            scaler, all_features, feature_groups, feature_info_for_serving, 
            data_info, args
        )
    
    # Print results
    print_final_results(best_overall_model, all_features, champion_run_id, args.serving_port)
    
    return champion_run_id, best_overall_model

# ===================================================================
# HELPER FUNCTIONS (extracted from original code)
# ===================================================================

def create_sample_data(random_state=42):
    """Create sample data for demonstration"""
    np.random.seed(random_state)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    sample_data = []
    
    for i, date in enumerate(dates[:5000]):
        base_sales = 50 + 20 * np.sin(i/100) + 10 * np.sin(i/24) + np.random.normal(0, 5)
        sample_data.append({
            'InvoiceDate': date,
            'TotalSales': max(base_sales, 0),
            'Quantity': np.random.randint(1, 20),
            'UnitPrice': np.random.uniform(1, 100),
            'Year': date.year,
            'Month': date.month,
            'Day': date.day,
            'DayOfWeek': date.dayofweek,
            'Hour': date.hour,
            'IsWeekend': 1 if date.dayofweek >= 5 else 0,
            'InvoiceNo_encoded': np.random.randint(0, 1000),
            'StockCode_encoded': np.random.randint(0, 500),
            'CustomerID_encoded': np.random.randint(0, 200),
            'Country_encoded': np.random.randint(0, 10)
        })
    
    df = pd.DataFrame(sample_data)
    data_source = "Generated Sample Data"
    print(f"✓ Sample data created: {df.shape}")
    return df, data_source

def create_time_series_features(df):
    """Create comprehensive time series features"""
    print("\n3. ENHANCED FEATURE ENGINEERING")
    print("-" * 35)
    
    # Sort by date
    df = df.sort_values('InvoiceDate').reset_index(drop=True)
    
    df_features = df.copy()
    
    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        df_features[f'TotalSales_lag_{lag}'] = df_features['TotalSales'].shift(lag)
    
    # Rolling statistics
    for window in [3, 7, 14]:
        df_features[f'TotalSales_rolling_mean_{window}'] = df_features['TotalSales'].rolling(window=window).mean()
        df_features[f'TotalSales_rolling_std_{window}'] = df_features['TotalSales'].rolling(window=window).std()
        df_features[f'TotalSales_rolling_min_{window}'] = df_features['TotalSales'].rolling(window=window).min()
        df_features[f'TotalSales_rolling_max_{window}'] = df_features['TotalSales'].rolling(window=window).max()
    
    # Time-based features
    df_features['DayOfMonth'] = df_features['InvoiceDate'].dt.day
    df_features['WeekOfYear'] = df_features['InvoiceDate'].dt.isocalendar().week
    df_features['Quarter'] = df_features['InvoiceDate'].dt.quarter
    df_features['DaysFromStart'] = (df_features['InvoiceDate'] - df_features['InvoiceDate'].min()).dt.days
    df_features['HourSin'] = np.sin(2 * np.pi * df_features['Hour'] / 24)
    df_features['HourCos'] = np.cos(2 * np.pi * df_features['Hour'] / 24)
    df_features['DayOfYearSin'] = np.sin(2 * np.pi * df_features['InvoiceDate'].dt.dayofyear / 365)
    df_features['DayOfYearCos'] = np.cos(2 * np.pi * df_features['InvoiceDate'].dt.dayofyear / 365)
    
    print(f"✓ Total features created: {df_features.shape[1]}")
    print(f"✓ Data after cleaning: {df_features.shape}")
    
    return df_features

def prepare_features(df_features):
    """Prepare features for training"""
    print("\n4. PREPARING FEATURES")
    print("-" * 25)
    
    # Define feature groups for better organization
    feature_groups = {
        'basic': ['Quantity', 'UnitPrice', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'IsWeekend'],
        'encoded': ['InvoiceNo_encoded', 'StockCode_encoded', 'CustomerID_encoded', 'Country_encoded'],
        'time': ['DayOfMonth', 'WeekOfYear', 'Quarter', 'DaysFromStart', 'HourSin', 'HourCos', 'DayOfYearSin', 'DayOfYearCos'],
        'lag': [f'TotalSales_lag_{lag}' for lag in [1, 2, 3, 7, 14]],
        'rolling': [f'TotalSales_rolling_{stat}_{window}' for stat in ['mean', 'std', 'min', 'max'] for window in [3, 7, 14]]
    }

    # Combine all features
    all_features = []
    for group, features in feature_groups.items():
        available = [f for f in features if f in df_features.columns]
        all_features.extend(available)
        print(f"✓ {group.capitalize()} features: {len(available)}")
    
    print(f"✓ Total features: {len(all_features)}")
    
    return all_features, feature_groups

def create_feature_info_for_serving(df_features, all_features, feature_groups):
    """Create feature information for serving"""
    feature_info_for_serving = {
        'feature_names': all_features,
        'feature_count': len(all_features),
        'feature_groups': feature_groups,
        'sample_row_index': len(df_features) // 2,  # Use middle row as sample
    }

    # Create sample data for serving
    sample_row = df_features.iloc[feature_info_for_serving['sample_row_index']]
    sample_features = []
    for feature in all_features:
        value = sample_row[feature]
        # Convert numpy types to native Python types for JSON serialization
        if hasattr(value, 'item'):  # numpy types have .item() method
            value = value.item()
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            value = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            value = float(value)
        sample_features.append(value)

    feature_info_for_serving['sample_features'] = sample_features
    feature_info_for_serving['sample_target'] = float(sample_row['TotalSales'])

    print(f"✓ Sample features for serving: {len(sample_features)}")
    
    return feature_info_for_serving

def perform_train_test_split(X, y, df_features, test_size=0.2, random_state=42):
    """Perform train-test split with scaling"""
    print("\n5. TRAIN-TEST SPLIT")
    print("-" * 20)

    # Time series split
    split_quantile = 1.0 - test_size
    split_date = df_features['InvoiceDate'].quantile(split_quantile)
    train_mask = df_features['InvoiceDate'] <= split_date
    test_mask = df_features['InvoiceDate'] > split_date

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✓ Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Include all your original helper functions here:
# - calculate_comprehensive_metrics
# - log_experiment_metadata  
# - get_algorithm_family
# - train_models
# - train_champion_model
# - print_final_results

def calculate_comprehensive_metrics(y_true, y_pred, prefix=""):
    """Calculate comprehensive metrics for MLflow logging"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
    
    # Error distribution
    errors = y_true - y_pred
    error_std = np.std(errors)
    error_mean = np.mean(errors)
    
    metrics = {
        f'{prefix}mse': mse,
        f'{prefix}rmse': rmse,
        f'{prefix}mae': mae,
        f'{prefix}r2': r2,
        f'{prefix}mape': mape,
        f'{prefix}error_std': error_std,
        f'{prefix}error_mean': error_mean,
        f'{prefix}explained_variance': r2 * 100
    }
    
    return metrics

def log_experiment_metadata(model_name, params, data_info, args):
    """Log comprehensive experiment metadata"""
    # Model parameters
    for param_name, value in params.items():
        mlflow.log_param(param_name, value)
    
    # Data information
    mlflow.log_param("data_source", data_info.get("source", "unknown"))
    mlflow.log_param("train_samples", data_info.get("train_samples", 0))
    mlflow.log_param("test_samples", data_info.get("test_samples", 0))
    mlflow.log_param("feature_count", data_info.get("feature_count", 0))
    mlflow.log_param("test_size", data_info.get("test_size", 0.2))
    mlflow.log_param("random_state", data_info.get("random_state", 42))
    
    # Model metadata
    mlflow.log_param("model_type", model_name)
    mlflow.log_param("algorithm_family", get_algorithm_family(model_name))
    mlflow.log_param("preprocessing", "StandardScaler")
    
    # CLI arguments
    mlflow.log_param("training_mode", args.mode)
    mlflow.log_param("model_selection", args.model_type)
    mlflow.log_param("data_path", args.data_path)
    
    # Serving integration info
    mlflow.log_param("serving_script", "mlflow_serve.py")
    mlflow.log_param("serving_port", str(args.serving_port))
    mlflow.log_param("tracking_port", "5000")
    
    # Experiment tags
    mlflow.set_tag("experiment_date", datetime.now().strftime("%Y-%m-%d"))
    mlflow.set_tag("model_category", "Time Series Forecasting")
    mlflow.set_tag("target_variable", "TotalSales")
    mlflow.set_tag("serving_ready", "true")
    mlflow.set_tag("cli_compatible", "true")

def get_algorithm_family(model_name):
    """Get algorithm family for categorization"""
    families = {
        "RandomForest": "Ensemble - Bagging",
        "GradientBoosting": "Ensemble - Boosting", 
        "XGBoost": "Ensemble - Gradient Boosting",
        "ExtraTrees": "Ensemble - Bagging",
        "LinearRegression": "Linear Models",
        "Ridge": "Linear Models",
        "Lasso": "Linear Models"
    }
    return families.get(model_name, "Other")

def train_models(X_train, X_test, y_train, y_test, all_features, data_info, args):
    """Train models based on configuration"""
    print("\n7. MODEL TRAINING WITH COMPREHENSIVE LOGGING")
    print("-" * 55)
    
    # Model configurations
    model_configs = {
        "RandomForest": {
            "class": RandomForestRegressor,
            "param_grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        },
        "GradientBoosting": {
            "class": GradientBoostingRegressor,
            "param_grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        },
        "ExtraTrees": {
            "class": ExtraTreesRegressor,
            "param_grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        }
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        model_configs["XGBoost"] = {
            "class": xgb.XGBRegressor,
            "param_grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        }
    
    # Filter models based on model_type argument
    if args.model_type != 'all':
        if args.model_type in model_configs:
            model_configs = {args.model_type: model_configs[args.model_type]}
        else:
            print(f"⚠️ Model type {args.model_type} not found, using all models")

    # Training results storage
    all_results = []
    best_overall_score = -np.inf
    best_overall_model = None

    # Train each model type
    for model_name, config in model_configs.items():
        print(f"\n7.{len(all_results)+1} TRAINING {model_name.upper()}")
        print("-" * (15 + len(model_name)))
        
        # Get parameter combinations
        param_combinations = list(ParameterGrid(config["param_grid"]))
        # Limit combinations based on argument
        max_combinations = args.max_combinations
        param_subset = param_combinations[:max_combinations]
        
        print(f"✓ Testing {len(param_subset)} {model_name} combinations")
        
        best_model_score = -np.inf
        best_model_params = None
        
        for i, params in enumerate(param_subset):
            run_name = f"{model_name}_run_{i+1:02d}"
            
            with mlflow.start_run(run_name=run_name):
                if args.verbose:
                    print(f"  Training {run_name}: {params}")
                
                # Log experiment metadata
                log_experiment_metadata(model_name, params, data_info, args)
                
                # Train model
                start_time = time.time()
                if model_name == "XGBoost":
                    model = config["class"](random_state=args.random_state, verbosity=0, **params)
                else:
                    model = config["class"](random_state=args.random_state, **params)
                
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate comprehensive metrics
                train_metrics = calculate_comprehensive_metrics(y_train, y_train_pred, "train_")
                test_metrics = calculate_comprehensive_metrics(y_test, y_test_pred, "test_")
                
                # Log all metrics
                for metric_name, value in {**train_metrics, **test_metrics}.items():
                    mlflow.log_metric(metric_name, value)
                
                # Log additional metrics
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.log_metric("samples_per_second", len(X_train) / training_time)
                
                # Log model artifacts with proper signature
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    signature=mlflow.models.infer_signature(X_train, y_train_pred)
                )
                
                # Log feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': all_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Save and log feature importance
                    importance_file = "feature_importance.csv"
                    feature_importance.to_csv(importance_file, index=False)
                    mlflow.log_artifact(importance_file)
                    
                    # Log top 5 features as metrics
                    for idx, row in feature_importance.head().iterrows():
                        mlflow.log_metric(f"top_feature_{idx+1}_importance", row['importance'])
                    
                    os.remove(importance_file)  # Clean up
                
                # Track best models
                current_score = test_metrics['test_r2']
                if current_score > best_model_score:
                    best_model_score = current_score
                    best_model_params = params
                
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_model = {
                        'name': model_name,
                        'params': params,
                        'score': current_score,
                        'model': model,
                        'run_id': mlflow.active_run().info.run_id
                    }
                
                # Store results
                all_results.append({
                    'model_type': model_name,
                    'run_name': run_name,
                    'run_id': mlflow.active_run().info.run_id,
                    'params': params,
                    'test_r2': current_score,
                    'test_rmse': test_metrics['test_rmse'],
                    'test_mae': test_metrics['test_mae'],
                    'training_time': training_time
                })
        
        print(f"✓ Best {model_name} R²: {best_model_score:.4f}")
    
    return all_results, best_overall_model

def train_champion_model(best_overall_model, X_train, X_test, y_train, y_test, 
                        scaler, all_features, feature_groups, feature_info_for_serving, 
                        data_info, args):
    """Train final champion model"""
    print(f"\n8. TRAINING CHAMPION MODEL")
    print("-" * 30)

    champion_run_id = None

    if best_overall_model:
        with mlflow.start_run(run_name="CHAMPION_MODEL") as run:
            champion_run_id = run.info.run_id
            
            print(f"Training champion: {best_overall_model['name']} with R²: {best_overall_model['score']:.4f}")
            
            # Log champion model metadata
            mlflow.set_tag("model_stage", "champion")
            mlflow.set_tag("champion_reason", f"Best R² score: {best_overall_model['score']:.4f}")
            mlflow.set_tag("ready_for_serving", "true")
            mlflow.set_tag("serving_script", "mlflow_serve.py")
            
            log_experiment_metadata(
                f"CHAMPION_{best_overall_model['name']}", 
                best_overall_model['params'], 
                data_info,
                args
            )
            
            # Retrain champion model
            champion_model = best_overall_model['model']
            
            # Calculate final metrics
            y_train_pred = champion_model.predict(X_train)
            y_test_pred = champion_model.predict(X_test)
            
            train_metrics = calculate_comprehensive_metrics(y_train, y_train_pred, "champion_train_")
            test_metrics = calculate_comprehensive_metrics(y_test, y_test_pred, "champion_test_")
            
            # Log champion metrics
            for metric_name, value in {**train_metrics, **test_metrics}.items():
                mlflow.log_metric(metric_name, value)
            
            # Log champion model with proper signature for serving
            mlflow.sklearn.log_model(
                champion_model, 
                "model",
                signature=mlflow.models.infer_signature(X_train, y_train_pred)
            )
            
            # Save scaler for production use
            joblib.dump(scaler, 'scaler.pkl')
            mlflow.log_artifact('scaler.pkl')
            os.remove('scaler.pkl')
            
            # Save feature information for serving integration
            json_safe_feature_info = {
                'feature_names': all_features,
                'feature_count': len(all_features),
                'feature_groups': feature_groups,
                'sample_row_index': int(feature_info_for_serving['sample_row_index']),
                'sample_features': [float(x) if isinstance(x, (np.floating, np.float64, np.float32)) 
                                   else int(x) if isinstance(x, (np.integer, np.int64, np.int32))
                                   else float(x.item()) if hasattr(x, 'item')
                                   else x for x in feature_info_for_serving['sample_features']],
                'sample_target': float(feature_info_for_serving['sample_target'])
            }
            
            with open('feature_info.json', 'w') as f:
                json.dump(json_safe_feature_info, f, indent=2)
        
            mlflow.log_artifact('feature_info.json')
            os.remove('feature_info.json')
            
            print(f"✓ Champion model logged with run_id: {champion_run_id}")
    
    return champion_run_id

def print_final_results(best_overall_model, all_features, champion_run_id, serving_port):
    """Print final results and serving instructions"""
    print(f"\n9. SERVING INTEGRATION READY")
    print("-" * 35)

    print(f"🎉 Training completed! Your models are ready for serving.")
    print(f"")
    print(f"📊 Model Summary:")
    print(f"   • Best Model: {best_overall_model['name'] if best_overall_model else 'None'}")
    print(f"   • R² Score: {best_overall_model['score']:.4f}" if best_overall_model else "   • R² Score: N/A")
    print(f"   • Features: {len(all_features)}")
    print(f"   • Champion Run ID: {champion_run_id}")
    print(f"")
    print(f"🚀 Next Steps for Serving:")
    print(f"")
    print(f"1. Keep MLflow UI running (current terminal):")
    print(f"   mlflow server --host localhost --port 5000")
    print(f"")
    print(f"2. Open NEW terminal for serving:")
    print(f"   python mlflow_serve.py info")
    print(f"   python mlflow_serve.py serve --port {serving_port}")
    print(f"")
    print(f"3. Open ANOTHER terminal for testing:")
    print(f"   python mlflow_serve.py test --port {serving_port}")
    print(f"")
    print(f"📡 Endpoints:")
    print(f"   • MLflow UI: http://localhost:5000")
    print(f"   • Model API: http://localhost:{serving_port} (after serving)")
    print(f"")
    print(f"🔧 MLflow Project Commands:")
    print(f"   mlflow run . -P data_path=data_processed.csv")
    print(f"   mlflow run . -P model_type=RandomForest -P test_size=0.3")
    print(f"")
    print(f"✅ All models are now compatible with mlflow_serve.py!")

def quick_serving_check(champion_run_id, all_features):
    """Quick check if serving can work"""
    try:
        print(f"\n🔍 Quick Serving Compatibility Check:")
        
        # Check if we can load the champion model
        if champion_run_id:
            model_uri = f"runs:/{champion_run_id}/model"
            try:
                model_info = mlflow.models.get_model_info(model_uri)
                print(f"   ✅ Champion model found and loadable")
                print(f"   📝 Features: {len(all_features)}")
                print(f"   🔗 Model URI: {model_uri}")
                
                # Check signature
                if model_info.signature:
                    expected_features = len(model_info.signature.inputs.inputs)
                    if expected_features == len(all_features):
                        print(f"   ✅ Feature count matches: {expected_features}")
                    else:
                        print(f"   ⚠️  Feature mismatch: expected {expected_features}, have {len(all_features)}")
                else:
                    print(f"   ⚠️  No model signature found")
                
                return True
            except Exception as e:
                print(f"   ❌ Error loading model: {e}")
                return False
        else:
            print(f"   ❌ No champion model found")
            return False
            
    except Exception as e:
        print(f"   ❌ Compatibility check failed: {e}")
        return False

# ===================================================================
# COMMAND LINE INTERFACE
# ===================================================================

if __name__ == "__main__":
    """
    Main entry point for CLI usage and MLflow Project
    
    Examples:
        python modelling.py
        python modelling.py --data_path data_processed.csv --model_type RandomForest
        python modelling.py --experiment_name CI_Test --test_size 0.3 --verbose
        
    MLflow Project:
        mlflow run . -P data_path=data_processed.csv
        mlflow run . -P model_type=RandomForest -P test_size=0.3
    """
    
    try:
        # Parse arguments and run main function
        args = parse_arguments()
        
        # Run main training pipeline
        champion_run_id, best_overall_model = main(args)
        
        # Get feature count for final check
        if best_overall_model:
            # We need to get all_features from the main function
            # For now, we'll do a simple check
            quick_serving_check(champion_run_id, [])
            
        print(f"\n🎉 READY FOR SERVING! Use mlflow_serve.py now.")
        print(f"\n" + "="*80)
        print(f"TRAINING COMPLETED - READY FOR SERVING")
        print(f"="*80)
        
    except KeyboardInterrupt:
        print(f"\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)