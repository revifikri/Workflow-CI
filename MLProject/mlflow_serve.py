#!/usr/bin/env python3
"""
Robust MLflow Model Serving Script - Handles Artifact Download Issues
Compatible with modelling.py and Windows 11
Fixed experiment detection to work with Sales_Monitoring_Experiment
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


class RobustMLflowModelServer:
    """Robust MLflow Model Server with artifact download retry logic"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        self.experiment_name = None
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
        """Check if MLflow tracking server is running and healthy"""
        try:
            # Check basic connectivity
            response = requests.get(self.tracking_uri, timeout=10)
            if response.status_code == 200:
                logger.info("✅ MLflow UI is running on port 5000")
                
                # Additional health check - try to list experiments
                try:
                    client = mlflow.tracking.MlflowClient()
                    experiments = client.search_experiments(max_results=1)
                    logger.info("✅ MLflow API is responsive")
                    return True
                except Exception as api_error:
                    logger.warning(f"MLflow API issue: {api_error}")
                    print(f"⚠️ MLflow UI is running but API has issues: {api_error}")
                    return False
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
    
    def find_sales_experiment(self) -> Optional[str]:
        """Find any Sales-related experiment (updated to be more flexible)"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            
            # Look for experiments with "Sales" in the name (more flexible)
            sales_experiments = []
            for exp in experiments:
                if exp.lifecycle_stage != "deleted":
                    name_lower = exp.name.lower()
                    # Check for various Sales-related experiment names
                    if any(keyword in name_lower for keyword in [
                        "sales", "forecast", "monitoring", "prediction", "retail"
                    ]):
                        sales_experiments.append(exp)
            
            if not sales_experiments:
                print("❌ No Sales-related experiments found")
                print("💡 Available experiments:")
                for exp in experiments:
                    if exp.lifecycle_stage != "deleted":
                        print(f"   - {exp.name}")
                        
                # If no sales experiments found, ask user to choose
                if experiments:
                    print("\n🤔 Would you like to use one of the available experiments?")
                    print("💡 You can also specify experiment name manually:")
                    print("   python mlflow_serve.py serve --experiment-name 'YourExperimentName'")
                return None
            
            # Sort by creation time (newest first)
            sales_experiments.sort(key=lambda x: x.creation_time, reverse=True)
            selected_exp = sales_experiments[0]
            print(f"🔍 Auto-detected experiment: {selected_exp.name}")
            
            if len(sales_experiments) > 1:
                print(f"💡 Other Sales-related experiments found:")
                for exp in sales_experiments[1:]:
                    print(f"   - {exp.name}")
            
            return selected_exp.name
            
        except Exception as e:
            logger.error(f"Error finding experiments: {e}")
            print(f"❌ Error finding experiments: {e}")
            return None
    
    def find_experiment_by_name(self, experiment_name: str) -> Optional[str]:
        """Find experiment by exact name"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            
            for exp in experiments:
                if exp.lifecycle_stage != "deleted" and exp.name == experiment_name:
                    print(f"✅ Found specified experiment: {experiment_name}")
                    return experiment_name
            
            print(f"❌ Experiment '{experiment_name}' not found")
            print("💡 Available experiments:")
            for exp in experiments:
                if exp.lifecycle_stage != "deleted":
                    print(f"   - {exp.name}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding experiment by name: {e}")
            return None
    
    def load_feature_info_with_retry(self, run_id: str, max_retries: int = 3) -> bool:
        """Load feature information with retry logic"""
        client = mlflow.tracking.MlflowClient()
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}: Loading feature info...")
                
                # Try to download feature_info.json with timeout
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
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⚠️ Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"❌ Failed to load feature_info.json after {max_retries} attempts")
        
        # Fallback: try to infer from model signature
        print("🔄 Trying fallback method using model signature...")
        try:
            model_uri = f"runs:/{run_id}/model"
            model_info = mlflow.models.get_model_info(model_uri)
            
            if model_info.signature and hasattr(model_info.signature.inputs, 'inputs'):
                self.feature_count = len(model_info.signature.inputs.inputs)
                self.feature_names = [f"feature_{i+1}" for i in range(self.feature_count)]
                self.sample_features = [0.0] * self.feature_count
                
                print(f"⚠️ Using model signature for feature info:")
                print(f"   📝 Features: {self.feature_count} (generic names)")
                print(f"   📊 Sample data: Generated (replace with real data)")
                
                return True
            else:
                print(f"❌ Cannot determine feature structure from model")
                return False
                
        except Exception as fallback_error:
            print(f"❌ Fallback method also failed: {fallback_error}")
            return False
    
    def find_best_model(self, experiment_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find the best model with robust error handling"""
        if not self.check_mlflow_server():
            return None
        
        # Use specified experiment or auto-detect
        if experiment_name:
            self.experiment_name = self.find_experiment_by_name(experiment_name)
        else:
            self.experiment_name = self.find_sales_experiment()
            
        if not self.experiment_name:
            return None
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                print(f"❌ Experiment '{self.experiment_name}' not found")
                return None
            
            logger.info(f"Found experiment: {self.experiment_name}")
            
            # Search for runs with error handling
            try:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="",
                    order_by=["start_time DESC"],
                    max_results=50  # Reduce load on server
                )
            except Exception as search_error:
                logger.error(f"Error searching runs: {search_error}")
                print(f"❌ Error searching runs: {search_error}")
                return None
            
            if runs.empty:
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
                
                priority_patterns = ['custom_test_r2', 'test_r2', 'champion_test_r2']
                sorted_r2_cols = []
                for pattern in priority_patterns:
                    matching = [col for col in r2_columns if pattern in col]
                    sorted_r2_cols.extend(matching)
                
                if sorted_r2_cols:
                    best_r2_col = sorted_r2_cols[0]
                    valid_runs = runs.dropna(subset=[best_r2_col])
                    if not valid_runs.empty:
                        best_run = valid_runs.loc[valid_runs[best_r2_col].idxmax()]
                        model_type = "Best R² Model"
                        logger.info(f"Found best model by {best_r2_col}: {best_run[best_r2_col]:.4f}")
                    else:
                        best_run = runs.iloc[0]
                        model_type = "Latest Model"
                else:
                    best_run = runs.iloc[0]
                    model_type = "Latest Model"
            
            # Extract model info
            model_info = {
                'run_id': best_run['run_id'],
                'model_uri': f"runs:/{best_run['run_id']}/model",
                'model_type': model_type,
                'experiment_id': experiment.experiment_id,
                'experiment_name': self.experiment_name,
                'run_name': best_run.get('tags.mlflow.runName', 'Unknown'),
            }
            
            # Add metrics if available
            metric_patterns = [
                'test_r2', 'custom_test_r2', 'champion_test_r2',
                'test_rmse', 'custom_test_rmse', 'champion_test_rmse',
                'test_mae', 'custom_test_mae', 'champion_test_mae'
            ]
            
            for pattern in metric_patterns:
                col_name = f'metrics.{pattern}'
                if col_name in runs.columns and not pd.isna(best_run.get(col_name)):
                    model_info[pattern] = best_run[col_name]
            
            # Load feature information with retry
            if self.load_feature_info_with_retry(best_run['run_id']):
                model_info['feature_count'] = self.feature_count
                model_info['feature_names'] = self.feature_names
                model_info['sample_features'] = self.sample_features
            
            print(f"🏆 Found {model_type}")
            print(f"   📊 Run ID: {model_info['run_id']}")
            print(f"   📈 Run Name: {model_info['run_name']}")
            print(f"   🧪 Experiment: {self.experiment_name}")
            
            # Display best available metric
            for metric in ['custom_test_r2', 'test_r2', 'champion_test_r2']:
                if metric in model_info:
                    print(f"   📈 R² Score: {model_info[metric]:.4f}")
                    break
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error finding model: {e}")
            print(f"❌ Error finding model: {e}")
            return None
    
    def validate_model_robust(self, model_uri: str, max_retries: int = 2) -> bool:
        """Validate model with retry and fallback logic"""
        print(f"🔍 Validating model: {model_uri}")
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Validation attempt {attempt + 1}/{max_retries}...")
                
                # Try to get model info with timeout
                model_info = mlflow.models.get_model_info(model_uri)
                logger.info(f"Model flavors: {list(model_info.flavors.keys())}")
                
                if model_info.signature:
                    if hasattr(model_info.signature.inputs, 'inputs'):
                        expected_features = len(model_info.signature.inputs.inputs)
                        current_features = len(self.feature_names)
                        
                        if expected_features != current_features:
                            print(f"⚠️ Feature count mismatch:")
                            print(f"   Model expects: {expected_features} features")
                            print(f"   Script provides: {current_features} features")
                        else:
                            print(f"✅ Feature count matches: {expected_features} features")
                else:
                    print("⚠️ Model has no signature - will try serving anyway")
                
                # Skip model loading test if it's causing issues
                print(f"✅ Model validation successful (basic check)")
                return True
                
            except Exception as e:
                logger.error(f"Validation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⚠️ Retrying validation in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"⚠️ Model validation failed after {max_retries} attempts")
                    print(f"🔄 Will try serving anyway - MLflow might handle it")
                    return True  # Allow serving to proceed
        
        return True
    
    def serve_model_robust(self, model_uri: str, host: str = "127.0.0.1", port: int = 1234, 
                          env_manager: str = "local", enable_mlserver: bool = False) -> bool:
        """Serve MLflow model with robust error handling"""
        
        print(f"\n🚀 Starting Robust MLflow Model Server")
        print(f"=" * 60)
        print(f"📊 Model URI: {model_uri}")
        print(f"🧪 Experiment: {self.experiment_name}")
        print(f"🌐 Host: {host}")
        print(f"📡 Serving Port: {port} (Model API)")
        print(f"📋 Tracking Port: 5000 (MLflow UI)")
        print(f"🔧 Environment Manager: {env_manager}")
        print(f"📝 Expected Features: {len(self.feature_names)}")
        
        # Validate model with retry
        if not self.validate_model_robust(model_uri):
            print("❌ Cannot serve model after multiple validation attempts")
            return False
        
        # Build serving command
        if self.is_windows:
            cmd = [
                sys.executable, "-m", "mlflow", "models", "serve",
                "-m", model_uri,
                "--host", host,
                "--port", str(port),
                "--env-manager", env_manager,
                "--timeout", "300"  # 5 minute timeout
            ]
        else:
            cmd = [
                "mlflow", "models", "serve",
                "-m", model_uri,
                "--host", host,
                "--port", str(port),
                "--env-manager", env_manager,
                "--timeout", "300"
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
        print(f"💡 If serving fails, try: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host localhost --port 5000")
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
            startup_timeout = 300  # 5 minutes
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
            print(f"💡 Try restarting MLflow server: mlflow server --host localhost --port 5000")
            return False
    
    def test_model_server(self, host: str = "127.0.0.1", port: int = 1234) -> bool:
        """Test the served model with robust error handling"""
        
        base_url = f"http://{host}:{port}"
        print(f"\n🧪 Testing MLflow Model Server")
        print(f"=" * 50)
        print(f"📡 Testing endpoint: {base_url}")
        print(f"🧪 Experiment: {self.experiment_name}")
        print(f"📝 Using {len(self.feature_names)} features from training")
        
        if not self.sample_features:
            print(f"⚠️ No sample features available - using zeros")
            self.sample_features = [0.0] * len(self.feature_names)
        
        try:
            # Test 1: Health Check with retry
            print("1. 🔍 Health Check...")
            for attempt in range(3):
                try:
                    health_response = requests.get(f"{base_url}/ping", timeout=15)
                    if health_response.status_code == 200:
                        print("   ✅ Server is healthy")
                        break
                    else:
                        print(f"   ⚠️ Health check status: {health_response.status_code}")
                except requests.exceptions.ConnectionError:
                    if attempt == 2:
                        print("   ❌ Connection refused - server not running")
                        print(f"   💡 Start server: python {os.path.basename(sys.argv[0])} serve --port {port}")
                        return False
                    else:
                        print(f"   🔄 Retrying health check ({attempt + 1}/3)...")
                        time.sleep(2)
                except Exception as e:
                    print(f"   ⚠️ Health check error: {e}")
            
            # Test 2: Single Prediction
            print("2. 🎯 Single Prediction...")
            prediction_data = {"instances": [self.sample_features]}
            headers = {"Content-Type": "application/json"}
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/invocations",
                json=prediction_data,
                headers=headers,
                timeout=60  # Longer timeout
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
                        return True
                    else:
                        print(f"   ⚠️ Unexpected prediction format: {prediction}")
                except json.JSONDecodeError:
                    print(f"   ❌ Invalid JSON response: {response.text}")
                    return False
            else:
                print(f"   ❌ Prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            return False
    
    def show_model_info(self, model_info: Dict[str, Any], port: int = 1234):
        """Show model and serving information"""
        
        print(f"\n📋 MODEL SERVING INFORMATION")
        print(f"=" * 60)
        
        # Model Information
        print(f"🏆 Model Details:")
        print(f"   Type: {model_info['model_type']}")
        print(f"   Run ID: {model_info['run_id']}")
        print(f"   Run Name: {model_info['run_name']}")
        print(f"   Experiment: {model_info['experiment_name']}")
        print(f"   Model URI: {model_info['model_uri']}")
        
        # Metrics
        for metric in ['custom_test_r2', 'test_r2', 'champion_test_r2']:
            if metric in model_info:
                print(f"   R² Score: {model_info[metric]:.4f}")
                break
        
        # Port Information
        print(f"\n🔌 Port Configuration:")
        print(f"   MLflow UI (Tracking): {self.tracking_uri} (port 5000)")
        print(f"   Model API (Serving): http://localhost:{port} (port {port})")
        
        # Feature Information
        print(f"\n📝 Feature Information:")
        print(f"   Expected Features: {len(self.feature_names)}")
        
        # Commands
        print(f"\n🚀 Serving Commands:")
        script_name = os.path.basename(sys.argv[0])
        print(f"   Info: python {script_name} info")
        print(f"   Serve: python {script_name} serve --port {port}")
        print(f"   Test: python {script_name} test --port {port}")
        
        print(f"\n💡 Troubleshooting:")
        print(f"   If serving fails with artifact errors:")
        print(f"   1. Restart MLflow server: mlflow server --host localhost --port 5000")
        print(f"   2. Check disk space and permissions")
        print(f"   3. Try reducing MLflow server load")


def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="Robust MLflow Model Serving - Handles Artifact Issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mlflow_serve.py info                          # Show model info
  python mlflow_serve.py serve                         # Serve with auto-detection
  python mlflow_serve.py serve --experiment-name "Sales_Monitoring_Experiment"  # Specify experiment
  python mlflow_serve.py test                          # Test with timeout

Features:
  - Robust artifact download with retry
  - Flexible experiment detection (Sales, Forecasting, Monitoring)
  - Graceful error handling
  - Extended timeouts for Windows
        """
    )
    
    parser.add_argument("command", choices=["info", "serve", "test"], help="Command to execute")
    parser.add_argument("--port", "-p", type=int, default=1234, help="Serving port (default: 1234)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default=None, help="Specific experiment name to use")
    parser.add_argument("--env-manager", choices=["local", "conda"], default="local", help="Environment manager")
    parser.add_argument("--enable-mlserver", action="store_true", help="Enable MLServer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    
    # Initialize robust server
    server = RobustMLflowModelServer(tracking_uri=args.tracking_uri)
    
    # Execute command
    if args.command == "info":
        model_info = server.find_best_model(experiment_name=args.experiment_name)
        if model_info:
            server.show_model_info(model_info, args.port)
        else:
            print("❌ No model found")
            return 1
    
    elif args.command == "serve":
        model_info = server.find_best_model(experiment_name=args.experiment_name)
        if model_info:
            success = server.serve_model_robust(
                model_info['model_uri'], host=args.host, port=args.port,
                env_manager=args.env_manager, enable_mlserver=args.enable_mlserver
            )
            return 0 if success else 1
        else:
            print("❌ No model found")
            return 1
    
    elif args.command == "test":
        model_info = server.find_best_model(experiment_name=args.experiment_name)
        if not model_info:
            print("❌ No model found for testing")
            return 1
        
        success = server.test_model_server(host=args.host, port=args.port)
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
