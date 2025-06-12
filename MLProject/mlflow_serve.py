#!/usr/bin/env python3
"""
Updated MLflow Model Serving Script - Auto-detects Features from Training
Compatible with modelling.py and Windows 11
"""

import os
import sys
import time
import json
import requests
import subprocess
import argparse
import platform
from typing import Dict, List, Optional, Any
import logging

# Windows-specific imports
if platform.system() == "Windows":
    import signal
    STOP_SIGNALS = [signal.SIGINT]
    if hasattr(signal, 'SIGBREAK'):
        STOP_SIGNALS.append(signal.SIGBREAK)
else:
    import signal
    STOP_SIGNALS = [signal.SIGINT, signal.SIGTERM]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow imports
try:
    import mlflow
    import mlflow.models
    import mlflow.sklearn
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Required packages missing: {e}")
    print("💡 Install with: pip install mlflow pandas numpy scikit-learn")
    sys.exit(1)


class MLflowModelServer:
    """MLflow Model Server that auto-detects features from training"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        self.experiment_name = "Sales_Forecasting_Experiment"
        self.is_windows = platform.system() == "Windows"
        
        # Will be populated from model artifacts
        self.feature_names = []
        self.sample_features = []
        self.feature_count = 0
        
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI: {self.tracking_uri}")
            if self.is_windows:
                logger.info("Running on Windows - using Windows-specific configurations")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def check_mlflow_server(self) -> bool:
        """Check if MLflow tracking server is running on port 5000"""
        try:
            response = requests.get(self.tracking_uri, timeout=5)
            if response.status_code == 200:
                logger.info("✅ MLflow UI is running on port 5000")
                return True
            else:
                logger.warning(f"MLflow server returned status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to MLflow server: {e}")
            print(f"❌ MLflow UI not running at {self.tracking_uri}")
            print(f"💡 Start it with: mlflow server --host localhost --port 5000")
            print(f"💡 Or: python -m mlflow server --host localhost --port 5000")
            if self.is_windows:
                print(f"💡 Windows: You might need to allow Python through Windows Firewall")
            return False
    
    def load_feature_info(self, run_id: str) -> bool:
        """Load feature information from model artifacts"""
        try:
            # Try to download feature_info.json from the run
            client = mlflow.tracking.MlflowClient()
            
            try:
                # Download feature_info.json artifact
                local_path = client.download_artifacts(run_id, "feature_info.json")
                
                with open(local_path, 'r') as f:
                    feature_info = json.load(f)
                
                self.feature_names = feature_info.get('feature_names', [])
                self.sample_features = feature_info.get('sample_features', [])
                self.feature_count = feature_info.get('feature_count', len(self.feature_names))
                
                print(f"✅ Loaded feature info from model artifacts:")
                print(f"   📝 Features: {self.feature_count}")
                print(f"   📊 Sample data available: {'Yes' if self.sample_features else 'No'}")
                
                return True
                
            except Exception as e:
                logger.warning(f"Could not load feature_info.json: {e}")
                
                # Fallback: try to infer from model signature
                model_uri = f"runs:/{run_id}/model"
                model_info = mlflow.models.get_model_info(model_uri)
                
                if model_info.signature and hasattr(model_info.signature.inputs, 'inputs'):
                    self.feature_count = len(model_info.signature.inputs.inputs)
                    # Generate generic feature names
                    self.feature_names = [f"feature_{i+1}" for i in range(self.feature_count)]
                    # Generate sample data (will need to be replaced with real data)
                    self.sample_features = [0.0] * self.feature_count
                    
                    print(f"⚠️  Using model signature for feature info:")
                    print(f"   📝 Features: {self.feature_count} (generic names)")
                    print(f"   📊 Sample data: Generated (replace with real data)")
                    
                    return True
                else:
                    print(f"❌ Cannot determine feature structure from model")
                    return False
                    
        except Exception as e:
            logger.error(f"Error loading feature info: {e}")
            print(f"❌ Error loading feature info: {e}")
            return False
    
    def find_best_model(self) -> Optional[Dict[str, Any]]:
        """Find the best model and load its feature information"""
        if not self.check_mlflow_server():
            return None
        
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.error(f"Experiment '{self.experiment_name}' not found")
                print(f"❌ Experiment '{self.experiment_name}' not found")
                print(f"💡 Run your training script first: python modelling.py")
                return None
            
            logger.info(f"Found experiment: {self.experiment_name}")
            
            # Search for runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",
                order_by=["metrics.test_r2 DESC"],
                max_results=100
            )
            
            if runs.empty:
                logger.error("No runs found in experiment")
                print("❌ No runs found in experiment")
                return None
            
            logger.info(f"Found {len(runs)} runs")
            
            # Look for champion model first
            champion_runs = runs[
                (runs['tags.mlflow.runName'].str.contains('CHAMPION', na=False)) |
                (runs['tags.model_stage'] == 'champion')
            ] if 'tags.mlflow.runName' in runs.columns else pd.DataFrame()
            
            if not champion_runs.empty:
                best_run = champion_runs.iloc[0]
                model_type = "Champion Model"
                logger.info("Found Champion model")
            else:
                # Get best run by R2 score
                r2_columns = [col for col in runs.columns if 'r2' in col.lower() and 'metrics' in col]
                if r2_columns:
                    best_r2_col = r2_columns[0]
                    best_run = runs.loc[runs[best_r2_col].idxmax()]
                    model_type = "Best R² Model"
                    logger.info(f"Found best model by {best_r2_col}")
                else:
                    best_run = runs.iloc[0]
                    model_type = "Latest Model"
                    logger.info("Using latest model")
            
            # Extract model info
            model_info = {
                'run_id': best_run['run_id'],
                'model_uri': f"runs:/{best_run['run_id']}/model",
                'model_type': model_type,
                'experiment_id': experiment.experiment_id,
                'run_name': best_run.get('tags.mlflow.runName', 'Unknown'),
            }
            
            # Add metrics if available
            for metric_name in ['test_r2', 'champion_test_r2', 'test_rmse', 'champion_test_rmse', 'test_mae', 'champion_test_mae']:
                col_name = f'metrics.{metric_name}'
                if col_name in runs.columns and not pd.isna(best_run.get(col_name)):
                    model_info[metric_name] = best_run[col_name]
            
            # Load feature information from the model
            if self.load_feature_info(best_run['run_id']):
                model_info['feature_count'] = self.feature_count
                model_info['feature_names'] = self.feature_names
                model_info['sample_features'] = self.sample_features
            
            print(f"🏆 Found {model_type}")
            print(f"   📊 Run ID: {model_info['run_id']}")
            print(f"   📈 Run Name: {model_info['run_name']}")
            if 'test_r2' in model_info:
                print(f"   📈 R² Score: {model_info['test_r2']:.4f}")
            if 'test_rmse' in model_info:
                print(f"   📉 RMSE: {model_info['test_rmse']:.2f}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error finding model: {e}")
            print(f"❌ Error finding model: {e}")
            return None
    
    def validate_model(self, model_uri: str) -> bool:
        """Validate the model and check feature requirements"""
        try:
            logger.info(f"Validating model: {model_uri}")
            
            # Get model info
            model_info = mlflow.models.get_model_info(model_uri)
            logger.info(f"Model flavors: {list(model_info.flavors.keys())}")
            
            if model_info.signature:
                logger.info(f"Model signature: {model_info.signature}")
                
                # Check expected features
                if hasattr(model_info.signature.inputs, 'inputs'):
                    expected_features = len(model_info.signature.inputs.inputs)
                    current_features = len(self.feature_names)
                    
                    if expected_features != current_features:
                        print(f"⚠️  Feature count mismatch:")
                        print(f"   Model expects: {expected_features} features")
                        print(f"   Script provides: {current_features} features")
                        print(f"   This might cause prediction errors")
                    else:
                        print(f"✅ Feature count matches: {expected_features} features")
            else:
                print("⚠️  Model has no signature - cannot validate feature count")
            
            # Try to load the model
            try:
                if 'sklearn' in model_info.flavors:
                    model = mlflow.sklearn.load_model(model_uri)
                    logger.info(f"Model loaded: {type(model).__name__}")
                elif 'python_function' in model_info.flavors:
                    model = mlflow.pyfunc.load_model(model_uri)
                    logger.info("Model loaded as Python function")
                
                print(f"✅ Model validation successful")
                return True
                
            except Exception as load_error:
                logger.error(f"Model loading failed: {load_error}")
                print(f"❌ Model loading failed: {load_error}")
                return False
                
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            print(f"❌ Model validation failed: {e}")
            return False
    
    def serve_model(self, model_uri: str, host: str = "127.0.0.1", port: int = 1234, 
                   env_manager: str = "local", enable_mlserver: bool = False) -> bool:
        """Serve MLflow model on specified port (different from tracking port 5000)"""
        
        print(f"\n🚀 Starting MLflow Model Server")
        print(f"=" * 60)
        print(f"📊 Model URI: {model_uri}")
        print(f"🌐 Host: {host}")
        print(f"📡 Serving Port: {port} (Model API)")
        print(f"📋 Tracking Port: 5000 (MLflow UI)")
        print(f"🔧 Environment Manager: {env_manager}")
        print(f"🚀 MLServer: {enable_mlserver}")
        print(f"📝 Expected Features: {len(self.feature_names)}")
        
        # Validate model
        if not self.validate_model(model_uri):
            print("❌ Cannot serve invalid model")
            return False
        
        # Build serving command
        if self.is_windows:
            cmd = [
                sys.executable, "-m", "mlflow", "models", "serve",
                "-m", model_uri,
                "--host", host,
                "--port", str(port),
                "--env-manager", env_manager
            ]
        else:
            cmd = [
                "mlflow", "models", "serve",
                "-m", model_uri,
                "--host", host,
                "--port", str(port),
                "--env-manager", env_manager
            ]
        
        if enable_mlserver:
            cmd.append("--enable-mlserver")
        
        print(f"\n💻 Command: {' '.join(cmd)}")
        print(f"\n📡 Model API will be at: http://{host}:{port}")
        print(f"🔗 Health check: http://{host}:{port}/ping")
        print(f"🎯 Predictions: http://{host}:{port}/invocations")
        print(f"📊 MLflow UI: {self.tracking_uri}")
        print(f"\n⏳ Starting server...")
        print(f"💡 Press Ctrl+C to stop server")
        print(f"=" * 60)
        
        try:
            # Start serving process
            if self.is_windows:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, universal_newlines=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, universal_newlines=True
                )
            
            # Signal handler
            def signal_handler(sig, frame):
                print(f"\n🛑 Shutting down server...")
                if self.is_windows:
                    try:
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
                    except:
                        pass
                else:
                    process.terminate()
                print(f"✅ Server stopped")
                sys.exit(0)
            
            for sig in STOP_SIGNALS:
                signal.signal(sig, signal_handler)
            
            # Monitor output
            server_ready = False
            startup_timeout = 180
            start_time = time.time()
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
                    
                    if any(indicator in line.lower() for indicator in [
                        "listening at", "application startup complete", 
                        "started server process", "uvicorn running on", "serving at"
                    ]):
                        server_ready = True
                        print(f"\n🎉 Server is ready!")
                        print(f"🧪 Test: python {os.path.basename(sys.argv[0])} test --port {port}")
                
                if time.time() - start_time > startup_timeout:
                    print(f"\n⚠️ Server startup timeout ({startup_timeout}s)")
                    break
                
                if process.poll() is not None:
                    print(f"\n❌ Server process ended unexpectedly")
                    break
            
            return server_ready
            
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped by user")
            return True
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def test_model_server(self, host: str = "127.0.0.1", port: int = 1234) -> bool:
        """Test the served model with correct feature structure from training"""
        
        base_url = f"http://{host}:{port}"
        print(f"\n🧪 Testing MLflow Model Server")
        print(f"=" * 50)
        print(f"📡 Testing endpoint: {base_url}")
        print(f"📝 Using {len(self.feature_names)} features from training")
        
        if not self.sample_features:
            print(f"⚠️  No sample features available - using zeros")
            self.sample_features = [0.0] * len(self.feature_names)
        
        try:
            # Test 1: Health Check
            print("1. 🔍 Health Check...")
            try:
                health_response = requests.get(f"{base_url}/ping", timeout=15)
                if health_response.status_code == 200:
                    print("   ✅ Server is healthy")
                else:
                    print(f"   ⚠️ Health check status: {health_response.status_code}")
            except requests.exceptions.ConnectionError:
                print("   ❌ Connection refused - server not running")
                print(f"   💡 Start server: python {os.path.basename(sys.argv[0])} serve --port {port}")
                return False
            except Exception as e:
                print(f"   ⚠️ Health check error: {e}")
            
            # Test 2: Single Prediction with features from training
            print("2. 🎯 Single Prediction...")
            print(f"   📝 Features ({len(self.feature_names)}):")
            if len(self.feature_names) <= 10:
                for i, (name, value) in enumerate(zip(self.feature_names, self.sample_features)):
                    print(f"     {i+1:2d}. {name}: {value}")
            else:
                for i in range(5):
                    print(f"     {i+1:2d}. {self.feature_names[i]}: {self.sample_features[i]}")
                print(f"     ... ({len(self.feature_names)-10} more features)")
                for i in range(len(self.feature_names)-5, len(self.feature_names)):
                    print(f"     {i+1:2d}. {self.feature_names[i]}: {self.sample_features[i]}")
            
            prediction_data = {"instances": [self.sample_features]}
            headers = {"Content-Type": "application/json"}
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/invocations",
                json=prediction_data,
                headers=headers,
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    prediction = response.json()
                    if isinstance(prediction, list) and len(prediction) > 0:
                        pred_value = prediction[0]
                        print(f"   ✅ Prediction successful!")
                        print(f"   📊 Predicted Sales: ${pred_value:.2f}")
                        print(f"   ⏱️  Response time: {response_time:.3f}s")
                    else:
                        print(f"   ⚠️ Unexpected prediction format: {prediction}")
                except json.JSONDecodeError:
                    print(f"   ❌ Invalid JSON response: {response.text}")
                    return False
            else:
                print(f"   ❌ Prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
            
            # Test 3: Batch Prediction (if we have enough sample data)
            print("3. 📦 Batch Prediction...")
            if len(self.sample_features) >= 2:
                # Create a modified version of sample features
                sample_2 = [val * 1.1 if isinstance(val, (int, float)) else val for val in self.sample_features]
                batch_data = {"instances": [self.sample_features, sample_2]}
            else:
                batch_data = {"instances": [self.sample_features, self.sample_features]}
            
            batch_response = requests.post(
                f"{base_url}/invocations",
                json=batch_data,
                headers=headers,
                timeout=30
            )
            
            if batch_response.status_code == 200:
                try:
                    batch_predictions = batch_response.json()
                    print(f"   ✅ Batch prediction successful!")
                    print(f"   📊 Results: {[f'${p:.2f}' for p in batch_predictions]}")
                except:
                    print(f"   ⚠️ Batch response: {batch_response.text}")
            else:
                print(f"   ⚠️ Batch prediction failed: {batch_response.status_code}")
            
            # Test 4: Error Handling (wrong feature count)
            print("4. 🚨 Error Handling...")
            bad_data = {"instances": [[1, 2, 3]]}  # Wrong number of features
            
            error_response = requests.post(
                f"{base_url}/invocations",
                json=bad_data,
                headers=headers,
                timeout=15
            )
            
            if error_response.status_code in [400, 422, 500]:
                print("   ✅ Error handling works correctly")
            else:
                print(f"   ⚠️ Unexpected error response: {error_response.status_code}")
            
            print(f"\n🎉 All tests completed!")
            print(f"📝 Your model expects {len(self.feature_names)} features")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            return False
    
    def show_model_info(self, model_info: Dict[str, Any], port: int = 1234):
        """Show model and serving information with auto-detected features"""
        
        print(f"\n📋 MODEL SERVING INFORMATION")
        print(f"=" * 60)
        
        # Model Information
        print(f"🏆 Model Details:")
        print(f"   Type: {model_info['model_type']}")
        print(f"   Run ID: {model_info['run_id']}")
        print(f"   Run Name: {model_info['run_name']}")
        print(f"   Model URI: {model_info['model_uri']}")
        
        # Metrics
        for metric in ['test_r2', 'champion_test_r2']:
            if metric in model_info:
                print(f"   R² Score: {model_info[metric]:.4f}")
                break
        
        # Port Information
        print(f"\n🔌 Port Configuration:")
        print(f"   MLflow UI (Tracking): {self.tracking_uri} (port 5000)")
        print(f"   Model API (Serving): http://localhost:{port} (port {port})")
        
        # Feature Information (auto-detected)
        print(f"\n📝 Feature Information (Auto-detected from Training):")
        print(f"   Expected Features: {len(self.feature_names)}")
        
        if len(self.feature_names) <= 15:
            print(f"   Feature Names:")
            for i, name in enumerate(self.feature_names):
                print(f"     {i+1:2d}. {name}")
        else:
            print(f"   Feature Names (showing first 10, last 5):")
            for i in range(10):
                print(f"     {i+1:2d}. {self.feature_names[i]}")
            print(f"     ... ({len(self.feature_names)-15} more features)")
            for i in range(len(self.feature_names)-5, len(self.feature_names)):
                print(f"     {i+1:2d}. {self.feature_names[i]}")
        
        # Sample Data
        if self.sample_features:
            print(f"\n📊 Sample Input Data (from Training):")
            if len(self.sample_features) <= 10:
                for i, (name, value) in enumerate(zip(self.feature_names, self.sample_features)):
                    print(f"     {name}: {value}")
            else:
                print(f"     First 5 features:")
                for i in range(5):
                    print(f"       {self.feature_names[i]}: {self.sample_features[i]}")
                print(f"     ... ({len(self.sample_features)-10} more)")
                print(f"     Last 5 features:")
                for i in range(len(self.sample_features)-5, len(self.sample_features)):
                    print(f"       {self.feature_names[i]}: {self.sample_features[i]}")
        
        # Commands
        print(f"\n🚀 Serving Commands:")
        script_name = os.path.basename(sys.argv[0])
        print(f"   Info: python {script_name} info")
        print(f"   Serve: python {script_name} serve --port {port}")
        print(f"   Test: python {script_name} test --port {port}")
        
        # API Usage
        print(f"\n📡 API Usage:")
        print(f"   Health: GET http://localhost:{port}/ping")
        print(f"   Predict: POST http://localhost:{port}/invocations")
        
        # Sample API call with real features
        if self.sample_features:
            print(f"\n🧪 Sample API Call (PowerShell):")
            sample_json = json.dumps({"instances": [self.sample_features]}, indent=2)
            print(f'''$body = @"
{sample_json}
"@
Invoke-RestMethod -Uri "http://localhost:{port}/invocations" -Method Post -Body $body -ContentType "application/json"''')


def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="MLflow Model Serving - Auto-detects Features from Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mlflow_serve.py info                      # Show model info
  python mlflow_serve.py serve                     # Serve on port 1234
  python mlflow_serve.py serve --port 8080         # Custom serving port
  python mlflow_serve.py test                      # Test on port 1234
  python mlflow_serve.py test --port 8080          # Test custom port

Prerequisites:
  1. Run: python modelling.py (to train models)
  2. Keep running: mlflow server --host localhost --port 5000
  3. Then use this script for serving

Port Usage:
  - Port 5000: MLflow UI/Tracking (mlflow server)
  - Port 1234: Model Serving API (this script)
        """
    )
    
    parser.add_argument("command", choices=["info", "serve", "test"], help="Command to execute")
    parser.add_argument("--port", "-p", type=int, default=1234, help="Serving port (default: 1234)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--env-manager", choices=["local", "conda"], default="local", help="Environment manager")
    parser.add_argument("--enable-mlserver", action="store_true", help="Enable MLServer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Show system info
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    
    # Initialize server
    server = MLflowModelServer(tracking_uri=args.tracking_uri)
    
    # Execute command
    if args.command == "info":
        model_info = server.find_best_model()
        if model_info:
            server.show_model_info(model_info, args.port)
        else:
            print("❌ No model found")
            print("💡 Steps to fix:")
            print("   1. Make sure MLflow UI is running: mlflow server --host localhost --port 5000")
            print("   2. Train models first: python modelling.py")
            return 1
    
    elif args.command == "serve":
        model_info = server.find_best_model()
        if model_info:
            success = server.serve_model(
                model_info['model_uri'], host=args.host, port=args.port,
                env_manager=args.env_manager, enable_mlserver=args.enable_mlserver
            )
            return 0 if success else 1
        else:
            print("❌ No model found")
            print("💡 Train a model first: python modelling.py")
            return 1
    
    elif args.command == "test":
        # First, load model info to get correct features
        model_info = server.find_best_model()
        if not model_info:
            print("❌ No model found for testing")
            print("💡 Make sure you have:")
            print("   1. MLflow UI running: mlflow server --host localhost --port 5000")
            print("   2. Trained models: python modelling.py")
            print("   3. Model server running: python mlflow_serve.py serve --port {port}")
            return 1
        
        success = server.test_model_server(host=args.host, port=args.port)
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())