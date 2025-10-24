import time
import psutil
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import mlflow
from typing import Dict, Any, List
import json
from datetime import datetime

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions', ['model_type', 'prediction'])
MODEL_CONFIDENCE = Histogram('model_confidence_score', 'Model confidence scores', ['model_type'])
SYSTEM_MEMORY = Gauge('system_memory_usage_bytes', 'System memory usage')
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'System CPU usage')
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])

class ModelMonitor:
    def __init__(self):
        self.prediction_history = []
        self.performance_metrics = {}
        
    def log_prediction(self, model_type: str, input_text: str, prediction: Dict, confidence: float):
        """Log model prediction for monitoring"""
        timestamp = datetime.now().isoformat()
        
        # Update Prometheus metrics
        MODEL_PREDICTIONS.labels(model_type=model_type, prediction=prediction.get('intent', 'unknown')).inc()
        MODEL_CONFIDENCE.labels(model_type=model_type).observe(confidence)
        
        # Store prediction history
        prediction_record = {
            'timestamp': timestamp,
            'model_type': model_type,
            'input_length': len(input_text),
            'prediction': prediction,
            'confidence': confidence,
            'input_hash': hash(input_text)  # For privacy
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"prediction_{timestamp}"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("confidence", confidence)
            mlflow.log_metric("input_length", len(input_text))
            mlflow.log_param("prediction", json.dumps(prediction))
    
    def detect_drift(self) -> Dict[str, Any]:
        """Detect model drift based on recent predictions"""
        if len(self.prediction_history) < 100:
            return {"drift_detected": False, "reason": "Insufficient data"}
        
        recent_predictions = self.prediction_history[-100:]
        older_predictions = self.prediction_history[-200:-100] if len(self.prediction_history) >= 200 else []
        
        if not older_predictions:
            return {"drift_detected": False, "reason": "Insufficient historical data"}
        
        # Calculate confidence drift
        recent_confidence = [p['confidence'] for p in recent_predictions]
        older_confidence = [p['confidence'] for p in older_predictions]
        
        recent_avg = sum(recent_confidence) / len(recent_confidence)
        older_avg = sum(older_confidence) / len(older_confidence)
        
        confidence_drift = abs(recent_avg - older_avg)
        
        # Calculate prediction distribution drift
        recent_intents = [p['prediction'].get('intent', 'unknown') for p in recent_predictions]
        older_intents = [p['prediction'].get('intent', 'unknown') for p in older_predictions]
        
        recent_dist = {intent: recent_intents.count(intent) / len(recent_intents) 
                      for intent in set(recent_intents)}
        older_dist = {intent: older_intents.count(intent) / len(older_intents) 
                     for intent in set(older_intents)}
        
        # Simple drift detection (threshold-based)
        drift_detected = confidence_drift > 0.1  # 10% confidence drop
        
        return {
            "drift_detected": drift_detected,
            "confidence_drift": confidence_drift,
            "recent_avg_confidence": recent_avg,
            "older_avg_confidence": older_avg,
            "recent_distribution": recent_dist,
            "older_distribution": older_dist
        }
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get overall model health metrics"""
        if not self.prediction_history:
            return {"status": "no_data"}
        
        recent_predictions = self.prediction_history[-50:]  # Last 50 predictions
        
        avg_confidence = sum(p['confidence'] for p in recent_predictions) / len(recent_predictions)
        low_confidence_count = sum(1 for p in recent_predictions if p['confidence'] < 0.7)
        
        health_score = avg_confidence * (1 - low_confidence_count / len(recent_predictions))
        
        status = "healthy" if health_score > 0.8 else "degraded" if health_score > 0.6 else "unhealthy"
        
        return {
            "status": status,
            "health_score": health_score,
            "avg_confidence": avg_confidence,
            "low_confidence_rate": low_confidence_count / len(recent_predictions),
            "total_predictions": len(self.prediction_history)
        }

class SystemMonitor:
    @staticmethod
    def update_system_metrics():
        """Update system resource metrics"""
        # Memory usage
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY.set(memory.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        SYSTEM_CPU.set(cpu_percent)
    
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """Get system health status"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            "memory": {
                "total": memory.total,
                "used": memory.used,
                "percent": memory.percent,
                "available": memory.available
            },
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "percent": (disk.used / disk.total) * 100
            }
        }

def monitor_api_call(func):
    """Decorator to monitor API calls"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            REQUEST_COUNT.labels(method='POST', endpoint=func.__name__, status='success').inc()
            return result
        except Exception as e:
            REQUEST_COUNT.labels(method='POST', endpoint=func.__name__, status='error').inc()
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    
    return wrapper

def monitor_model_prediction(func):
    """Decorator to monitor model predictions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Log prediction metrics
            if isinstance(result, dict) and 'confidence' in result:
                MODEL_CONFIDENCE.labels(model_type=func.__name__).observe(result['confidence'])
            
            return result
        except Exception as e:
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start_time
            logging.info(f"Model prediction took {duration:.3f} seconds")
    
    return wrapper

class AlertManager:
    def __init__(self):
        self.alert_thresholds = {
            'low_confidence': 0.6,
            'high_error_rate': 0.1,
            'high_memory_usage': 0.85,
            'high_cpu_usage': 0.9
        }
    
    def check_alerts(self, model_health: Dict, system_health: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        
        # Model health alerts
        if model_health.get('avg_confidence', 1.0) < self.alert_thresholds['low_confidence']:
            alerts.append({
                'type': 'model_performance',
                'severity': 'warning',
                'message': f"Low model confidence: {model_health['avg_confidence']:.3f}",
                'timestamp': datetime.now().isoformat()
            })
        
        # System health alerts
        if system_health['memory']['percent'] > self.alert_thresholds['high_memory_usage'] * 100:
            alerts.append({
                'type': 'system_resource',
                'severity': 'warning',
                'message': f"High memory usage: {system_health['memory']['percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        if system_health['cpu']['percent'] > self.alert_thresholds['high_cpu_usage'] * 100:
            alerts.append({
                'type': 'system_resource',
                'severity': 'critical',
                'message': f"High CPU usage: {system_health['cpu']['percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts

# Global instances
model_monitor = ModelMonitor()
system_monitor = SystemMonitor()
alert_manager = AlertManager()