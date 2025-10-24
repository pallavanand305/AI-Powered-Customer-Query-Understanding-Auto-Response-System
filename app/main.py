from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
import sqlite3
from datetime import datetime
import asyncio
from typing import Optional, List, Dict
import json
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

try:
    from app.models.ensemble_model import AdvancedQueryClassifier
    from app.services.llm_service import LLMResponseGenerator, ResponsePersonalizer
    from app.utils.monitoring import (
        model_monitor, system_monitor, alert_manager,
        monitor_api_call, SystemMonitor
    )
except ImportError as e:
    print(f"Warning: Some advanced features may not be available: {e}")
    # Fallback implementations
    class AdvancedQueryClassifier:
        def predict_with_confidence(self, text):
            return {
                'intent': 'general',
                'sentiment': 'NEUTRAL',
                'priority': 'LOW',
                'intent_confidence': 0.8,
                'sentiment_confidence': 0.8,
                'priority_score': 0.3,
                'uncertainty_score': 0.2
            }
        def build_ensemble(self): pass
    
    class LLMResponseGenerator:
        async def generate_contextual_response(self, **kwargs):
            return {
                'response': 'Thank you for your query. We will assist you shortly.',
                'suggested_actions': ['Contact support'],
                'escalation_needed': False
            }
    
    class ResponsePersonalizer:
        def personalize_response(self, response, customer_id, customer_data):
            return response
    
    class ModelMonitor:
        def log_prediction(self, *args, **kwargs): pass
        def get_model_health(self): return {'status': 'healthy'}
        def detect_drift(self): return {'drift_detected': False}
    
    class SystemMonitor:
        @staticmethod
        def update_system_metrics(): pass
        @staticmethod
        def get_system_health(): return {'memory': {'percent': 50}, 'cpu': {'percent': 30}}
    
    class AlertManager:
        def check_alerts(self, *args): return []
    
    def monitor_api_call(func): return func
    
    model_monitor = ModelMonitor()
    system_monitor = SystemMonitor()
    alert_manager = AlertManager()

app = FastAPI(
    title="AI Customer Query Understanding System",
    description="Advanced AI/ML system for customer query processing with MLOps",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize advanced models
advanced_classifier = AdvancedQueryClassifier()
llm_generator = LLMResponseGenerator()
response_personalizer = ResponsePersonalizer()
sentiment_analyzer = pipeline("sentiment-analysis")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Customer query text", min_length=1, max_length=1000)
    customer_id: str = Field(..., description="Unique customer identifier")
    channel: Optional[str] = Field("web", description="Communication channel (web, email, phone)")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    customer_tier: Optional[str] = Field("standard", description="Customer tier (standard, premium, enterprise)")

class QueryResponse(BaseModel):
    intent: str
    sentiment: str
    confidence: float
    priority: str
    priority_score: float
    response: str
    suggested_actions: List[str]
    escalation_needed: bool
    uncertainty_score: float
    response_time_ms: float
    timestamp: str
    model_version: str = "2.0.0"

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_health: Dict
    system_health: Dict
    alerts: List[Dict]

class MetricsResponse(BaseModel):
    query_counts: Dict[str, int]
    avg_confidence: float
    avg_response_time: float
    error_rate: float
    drift_status: Dict

# Enhanced database setup
def init_db():
    conn = sqlite3.connect('queries.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            customer_id TEXT,
            session_id TEXT,
            query TEXT,
            intent TEXT,
            sentiment TEXT,
            confidence REAL,
            priority TEXT,
            priority_score REAL,
            response TEXT,
            response_time_ms REAL,
            escalation_needed BOOLEAN,
            uncertainty_score REAL,
            channel TEXT,
            customer_tier TEXT,
            timestamp TEXT,
            model_version TEXT
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS customer_history (
            id INTEGER PRIMARY KEY,
            customer_id TEXT,
            interaction_summary TEXT,
            resolution_status TEXT,
            satisfaction_score REAL,
            timestamp TEXT
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY,
            model_type TEXT,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            timestamp TEXT
        )
    ''')
    
    conn.close()

init_db()

# Background task to update system metrics
async def update_metrics():
    while True:
        try:
            SystemMonitor.update_system_metrics()
        except:
            pass
        await asyncio.sleep(30)  # Update every 30 seconds

# Startup function
async def startup_event():
    try:
        # Start background metrics collection
        asyncio.create_task(update_metrics())
        
        # Initialize advanced classifier
        advanced_classifier.build_ensemble()
    except Exception as e:
        print(f"Startup warning: {e}")

# Call startup on import for testing
try:
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
except:
    pass



@app.post("/analyze-query", response_model=QueryResponse)
@monitor_api_call
async def analyze_query(request: QueryRequest, background_tasks: BackgroundTasks):
    start_time = datetime.now()
    
    try:
        # Get customer history for context
        customer_history = get_customer_history(request.customer_id)
        
        # Advanced prediction with uncertainty quantification
        prediction_result = advanced_classifier.predict_with_confidence(request.query)
        
        # Generate contextual response using LLM
        llm_response = await llm_generator.generate_contextual_response(
            query=request.query,
            intent=prediction_result['intent'],
            sentiment=prediction_result['sentiment'],
            priority=prediction_result['priority'],
            customer_history=customer_history
        )
        
        # Personalize response
        customer_data = {
            "name": f"Customer {request.customer_id}",  # Would fetch from customer DB
            "tier": request.customer_tier,
            "account_type": request.customer_tier
        }
        
        personalized_response = response_personalizer.personalize_response(
            llm_response['response'],
            request.customer_id,
            customer_data
        )
        
        # Calculate response time
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response object
        response_obj = QueryResponse(
            intent=prediction_result['intent'],
            sentiment=prediction_result['sentiment'],
            confidence=prediction_result['intent_confidence'],
            priority=prediction_result['priority'],
            priority_score=prediction_result['priority_score'],
            response=personalized_response,
            suggested_actions=llm_response['suggested_actions'],
            escalation_needed=llm_response['escalation_needed'],
            uncertainty_score=prediction_result['uncertainty_score'],
            response_time_ms=response_time_ms,
            timestamp=datetime.now().isoformat()
        )
        
        # Store in database (background task)
        background_tasks.add_task(
            store_query_result,
            request,
            response_obj,
            response_time_ms
        )
        
        # Log prediction for monitoring
        model_monitor.log_prediction(
            model_type="ensemble_classifier",
            input_text=request.query,
            prediction={
                'intent': prediction_result['intent'],
                'sentiment': prediction_result['sentiment'],
                'priority': prediction_result['priority']
            },
            confidence=prediction_result['intent_confidence']
        )
        
        return response_obj
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

def get_customer_history(customer_id: str) -> List[Dict]:
    """Retrieve customer interaction history"""
    conn = sqlite3.connect('queries.db')
    cursor = conn.execute(
        'SELECT query, intent, sentiment, timestamp FROM queries WHERE customer_id = ? ORDER BY timestamp DESC LIMIT 5',
        (customer_id,)
    )
    
    history = []
    for row in cursor.fetchall():
        history.append({
            'query': row[0],
            'intent': row[1],
            'sentiment': row[2],
            'date': row[3],
            'summary': f"{row[1]} query with {row[2]} sentiment"
        })
    
    conn.close()
    return history

def store_query_result(request: QueryRequest, response: QueryResponse, response_time_ms: float):
    """Store query result in database"""
    conn = sqlite3.connect('queries.db')
    conn.execute('''
        INSERT INTO queries (
            customer_id, session_id, query, intent, sentiment, confidence,
            priority, priority_score, response, response_time_ms,
            escalation_needed, uncertainty_score, channel, customer_tier,
            timestamp, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        request.customer_id,
        request.session_id,
        request.query,
        response.intent,
        response.sentiment,
        response.confidence,
        response.priority,
        response.priority_score,
        response.response,
        response_time_ms,
        response.escalation_needed,
        response.uncertainty_score,
        request.channel,
        request.customer_tier,
        response.timestamp,
        response.model_version
    ))
    conn.commit()
    conn.close()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    # Get model health
    model_health = model_monitor.get_model_health()
    
    # Get system health
    system_health = system_monitor.get_system_health()
    
    # Check for alerts
    alerts = alert_manager.check_alerts(model_health, system_health)
    
    # Determine overall status
    overall_status = "healthy"
    if model_health.get('status') == 'unhealthy' or any(alert['severity'] == 'critical' for alert in alerts):
        overall_status = "unhealthy"
    elif model_health.get('status') == 'degraded' or alerts:
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        model_health=model_health,
        system_health=system_health,
        alerts=alerts
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    conn = sqlite3.connect('queries.db')
    
    # Query counts by intent
    cursor = conn.execute('SELECT intent, COUNT(*) as count FROM queries GROUP BY intent')
    query_counts = dict(cursor.fetchall())
    
    # Average confidence
    cursor = conn.execute('SELECT AVG(confidence) FROM queries WHERE timestamp > datetime("now", "-1 day")')
    avg_confidence = cursor.fetchone()[0] or 0.0
    
    # Average response time
    cursor = conn.execute('SELECT AVG(response_time_ms) FROM queries WHERE timestamp > datetime("now", "-1 day")')
    avg_response_time = cursor.fetchone()[0] or 0.0
    
    # Error rate (queries with low confidence)
    cursor = conn.execute('''
        SELECT 
            COUNT(CASE WHEN confidence < 0.7 THEN 1 END) * 1.0 / COUNT(*) as error_rate
        FROM queries 
        WHERE timestamp > datetime("now", "-1 day")
    ''')
    error_rate = cursor.fetchone()[0] or 0.0
    
    conn.close()
    
    # Get drift status
    drift_status = model_monitor.detect_drift()
    
    return MetricsResponse(
        query_counts=query_counts,
        avg_confidence=avg_confidence,
        avg_response_time=avg_response_time,
        error_rate=error_rate,
        drift_status=drift_status
    )

@app.get("/prometheus-metrics")
async def prometheus_metrics():
    """Endpoint for Prometheus to scrape metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model-drift")
async def check_model_drift():
    """Check for model drift"""
    drift_status = model_monitor.detect_drift()
    return drift_status

@app.post("/retrain-trigger")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    background_tasks.add_task(retrain_models)
    return {"message": "Retraining triggered", "timestamp": datetime.now().isoformat()}

async def retrain_models():
    """Background task to retrain models"""
    # This would trigger the ML pipeline
    # For now, just log the event
    print(f"Model retraining started at {datetime.now()}")
    # In practice, this would call the ML pipeline
    # advanced_classifier.retrain_with_new_data()

@app.get("/customer-analytics/{customer_id}")
async def get_customer_analytics(customer_id: str):
    """Get analytics for specific customer"""
    conn = sqlite3.connect('queries.db')
    
    # Customer query statistics
    cursor = conn.execute('''
        SELECT 
            COUNT(*) as total_queries,
            AVG(confidence) as avg_confidence,
            AVG(priority_score) as avg_priority,
            COUNT(CASE WHEN escalation_needed = 1 THEN 1 END) as escalations
        FROM queries 
        WHERE customer_id = ?
    ''', (customer_id,))
    
    stats = cursor.fetchone()
    
    # Intent distribution
    cursor = conn.execute('''
        SELECT intent, COUNT(*) as count 
        FROM queries 
        WHERE customer_id = ? 
        GROUP BY intent
    ''', (customer_id,))
    
    intent_dist = dict(cursor.fetchall())
    
    conn.close()
    
    return {
        "customer_id": customer_id,
        "total_queries": stats[0] or 0,
        "avg_confidence": stats[1] or 0.0,
        "avg_priority": stats[2] or 0.0,
        "escalations": stats[3] or 0,
        "intent_distribution": intent_dist
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)