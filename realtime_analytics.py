# realtime_analytics.py
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
import asyncio
import websockets
import json
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import threading
import queue
import time
import aioredis
from prophet import Prophet
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViewerEngagementPredictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, mask=None):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=mask)
        
        # Global pooling
        if mask is not None:
            masked_attn = attn_out * (~mask).unsqueeze(-1).float()
            pooled = masked_attn.sum(dim=1) / (~mask).sum(dim=1).unsqueeze(-1).float()
        else:
            pooled = attn_out.mean(dim=1)
            
        # Final prediction
        output = self.fc_layers(pooled)
        return output

class StreamAnalyticsEngine:
    def __init__(self, websocket_url=None, redis_host='localhost', redis_port=6379):
        self.websocket_url = websocket_url
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Custom engagement predictor
        self.engagement_model = ViewerEngagementPredictor().to(self.device)
        self.engagement_optimizer = torch.optim.Adam(self.engagement_model.parameters())
        
        # Analytics data structures
        self.viewer_metrics = defaultdict(lambda: {
            'messages': deque(maxlen=100),
            'timestamps': deque(maxlen=100),
            'sentiments': deque(maxlen=100),
            'engagement_score': 0.0,
            'activity_pattern': []
        })
        
        self.stream_metrics = {
            'viewer_count': deque(maxlen=1000),
            'chat_rate': deque(maxlen=1000),
            'sentiment_trend': deque(maxlen=1000),
            'engagement_trend': deque(maxlen=1000),
            'timestamps': deque(maxlen=1000)
        }
        
        # Real-time buffers
        self.message_buffer = deque(maxlen=500)
        self.event_buffer = deque(maxlen=200)
        self.prediction_buffer = deque(maxlen=100)
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.anomaly_threshold = 0.7
        
        # Time series forecasting
        self.forecast_models = {}
        
        # Processing queues
        self.analytics_queue = queue.Queue(maxsize=1000)
        self.prediction_queue = queue.Queue(maxsize=500)
        
        # Start background processing
        self._start_background_tasks()
        
        # Feature extractors
        self.feature_scaler = StandardScaler()
        
        # Content recommendation engine
        self.content_embeddings = {}
        self.content_performance = defaultdict(lambda: {
            'views': 0,
            'engagement': 0.0,
            'retention': 0.0
        })
        
        # Initialize Redis connection
        self.redis_pool = None
        self._init_redis(redis_host, redis_port)
        
    def _init_redis(self, host, port):
        """Initialize Redis connection pool"""
        try:
            self.redis_pool = aioredis.ConnectionPool.from_url(
                f"redis://{host}:{port}",
                max_connections=10
            )
            logger.info("Redis connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            
    def _start_background_tasks(self):
        """Start background processing tasks"""
        # Analytics processing thread
        analytics_thread = threading.Thread(target=self._analytics_worker, daemon=True)
        analytics_thread.start()
        
        # Prediction thread
        prediction_thread = threading.Thread(target=self._prediction_worker, daemon=True)
        prediction_thread.start()
        
        # Metrics aggregation thread
        metrics_thread = threading.Thread(target=self._metrics_aggregator, daemon=True)
        metrics_thread.start()
        
    def _analytics_worker(self):
        """Process analytics data in background"""
        while True:
            try:
                if not self.analytics_queue.empty():
                    data = self.analytics_queue.get(timeout=0.1)
                    self._process_analytics_data(data)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in analytics worker: {e}")
                
    def _prediction_worker(self):
        """Generate predictions in background"""
        while True:
            try:
                if len(self.message_buffer) >= 10:
                    predictions = self._generate_predictions()
                    self.prediction_queue.put(predictions)
                time.sleep(5)  # Generate predictions every 5 seconds
            except Exception as e:
                logger.error(f"Error in prediction worker: {e}")
                
    def _metrics_aggregator(self):
        """Aggregate metrics periodically"""
        while True:
            try:
                self._aggregate_stream_metrics()
                time.sleep(60)  # Aggregate every minute
            except Exception as e:
                logger.error(f"Error in metrics aggregator: {e}")
                
    async def process_chat_message(self, username: str, message: str, 
                                 timestamp: datetime = None) -> Dict[str, Any]:
        """Process incoming chat message for analytics"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Extract features
        features = await self._extract_message_features(message)
        
        # Update viewer metrics
        self.viewer_metrics[username]['messages'].append(message)
        self.viewer_metrics[username]['timestamps'].append(timestamp)
        
        # Add to buffers
        message_data = {
            'username': username,
            'message': message,
            'timestamp': timestamp,
            'features': features
        }
        
        self.message_buffer.append(message_data)
        self.analytics_queue.put(message_data)
        
        # Real-time engagement analysis
        engagement = await self._analyze_engagement(username, message, features)
        
        # Anomaly detection
        is_anomaly = self._detect_anomaly(features)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(username, features)
        
        return {
            'username': username,
            'engagement_score': engagement,
            'is_anomaly': is_anomaly,
            'recommendations': recommendations,
            'features': features
        }
        
    async def _extract_message_features(self, message: str) -> Dict[str, Any]:
        """Extract features from message using BERT"""
        # Tokenize and encode
        inputs = self.bert_tokenizer(
            message,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
        # Extract linguistic features
        features = {
            'embedding': embeddings[0].tolist(),
            'length': len(message),
            'word_count': len(message.split()),
            'exclamation_count': message.count('!'),
            'question_count': message.count('?'),
            'caps_ratio': sum(1 for c in message if c.isupper()) / max(1, len(message)),
            'emoji_count': sum(1 for c in message if ord(c) > 127),
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', message))
        }
        
        return features
        
    async def _analyze_engagement(self, username: str, message: str, 
                                features: Dict) -> float:
        """Analyze user engagement level"""
        # Get user history
        user_data = self.viewer_metrics[username]
        
        # Calculate message frequency
        if len(user_data['timestamps']) > 1:
            time_diffs = [
                (user_data['timestamps'][i] - user_data['timestamps'][i-1]).total_seconds()
                for i in range(1, len(user_data['timestamps']))
            ]
            avg_time_between_messages = np.mean(time_diffs) if time_diffs else float('inf')
        else:
            avg_time_between_messages = float('inf')
            
        # Message complexity score
        complexity_score = (
            features['word_count'] * 0.1 +
            features['exclamation_count'] * 0.2 +
            features['question_count'] * 0.3 +
            features['emoji_count'] * 0.1
        )
        
        # Frequency score (inverse of time between messages)
        frequency_score = 1.0 / (1.0 + avg_time_between_messages / 60)  # Normalize by minutes
        
        # Calculate engagement
        engagement_score = (
            complexity_score * 0.4 +
            frequency_score * 0.6
        )
        
        # Update user engagement
        user_data['engagement_score'] = 0.8 * user_data['engagement_score'] + 0.2 * engagement_score
        
        return float(engagement_score)
        
    def _detect_anomaly(self, features: Dict) -> bool:
        """Detect anomalous behavior"""
        # Create feature vector
        feature_vector = np.array([
            features['length'],
            features['word_count'],
            features['exclamation_count'],
            features['question_count'],
            features['caps_ratio'],
            features['emoji_count'],
            features['url_count']
        ]).reshape(1, -1)
        
        # Check if we have enough data to detect anomalies
        if len(self.message_buffer) < 50:
            return False
            
        # Fit anomaly detector if not fitted
        if not hasattr(self.anomaly_detector, 'offset_'):
            # Prepare training data
            training_features = []
            for msg in list(self.message_buffer)[-100:]:
                if 'features' in msg:
                    training_features.append([
                        msg['features']['length'],
                        msg['features']['word_count'],
                        msg['features']['exclamation_count'],
                        msg['features']['question_count'],
                        msg['features']['caps_ratio'],
                        msg['features']['emoji_count'],
                        msg['features']['url_count']
                    ])
                    
            if len(training_features) > 10:
                self.anomaly_detector.fit(training_features)
            else:
                return False
                
        # Predict anomaly
        try:
            anomaly_score = self.anomaly_detector.decision_function(feature_vector)[0]
            return anomaly_score < -self.anomaly_threshold
        except:
            return False
            
    async def _generate_recommendations(self, username: str, 
                                      features: Dict) -> List[Dict[str, Any]]:
        """Generate content recommendations"""
        recommendations = []
        
        # Get user profile
        user_data = self.viewer_metrics[username]
        
        # Analyze message patterns
        if len(user_data['messages']) >= 5:
            # Topic modeling on recent messages
            recent_messages = list(user_data['messages'])[-10:]
            topics = self._extract_topics(recent_messages)
            
            # Generate content recommendations based on topics
            for topic, score in topics[:3]:
                recommendations.append({
                    'type': 'content',
                    'topic': topic,
                    'confidence': score,
                    'reason': f"User shows interest in {topic}"
                })
                
        # Engagement-based recommendations
        if user_data['engagement_score'] > 0.7:
            recommendations.append({
                'type': 'interaction',
                'action': 'direct_response',
                'confidence': 0.8,
                'reason': 'High engagement user'
            })
            
        return recommendations
        
    def _extract_topics(self, messages: List[str]) -> List[Tuple[str, float]]:
        """Extract topics from messages"""
        # Simple keyword extraction (can be enhanced with LDA or other topic models)
        word_freq = defaultdict(int)
        
        for message in messages:
            words = message.lower().split()
            for word in words:
                if len(word) > 4:  # Filter short words
                    word_freq[word] += 1
                    
        # Sort by frequency
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        if topics:
            max_freq = topics[0][1]
            topics = [(word, freq/max_freq) for word, freq in topics]
            
        return topics[:5]
        
    def _process_analytics_data(self, data: Dict):
        """Process analytics data"""
        timestamp = data['timestamp']
        
        # Update stream metrics
        current_time = datetime.now()
        self.stream_metrics['timestamps'].append(current_time)
        
        # Calculate chat rate (messages per minute)
        recent_messages = [
            msg for msg in self.message_buffer 
            if (current_time - msg['timestamp']).total_seconds() < 60
        ]
        chat_rate = len(recent_messages)
        self.stream_metrics['chat_rate'].append(chat_rate)
        
        # Update viewer count (approximate based on unique users)
        recent_users = set(msg['username'] for msg in recent_messages)
        self.stream_metrics['viewer_count'].append(len(recent_users))
        
    def _generate_predictions(self) -> Dict[str, Any]:
        """Generate real-time predictions"""
        try:
            # Prepare sequence data
            sequence_data = self._prepare_sequence_data()
            
            if sequence_data is None:
                return {}
                
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            
            # Generate predictions
            self.engagement_model.eval()
            with torch.no_grad():
                predictions = self.engagement_model(sequence_tensor)
                
            # Interpret predictions
            pred_values = predictions.cpu().numpy()[0]
            
            return {
                'viewer_trend': 'increasing' if pred_values[0] > 0.5 else 'decreasing',
                'engagement_forecast': float(pred_values[1]),
                'optimal_content_type': self._get_content_recommendation(pred_values[2]),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {}
            
    def _prepare_sequence_data(self) -> Optional[np.ndarray]:
        """Prepare sequence data for prediction model"""
        if len(self.stream_metrics['chat_rate']) < 10:
            return None
            
        # Extract features from recent metrics
        features = []
        
        # Chat rate features
        chat_rates = list(self.stream_metrics['chat_rate'])[-20:]
        features.extend([
            np.mean(chat_rates),
            np.std(chat_rates),
            np.max(chat_rates),
            np.min(chat_rates)
        ])
        
        # Viewer count features
        viewer_counts = list(self.stream_metrics['viewer_count'])[-20:]
        features.extend([
            np.mean(viewer_counts),
            np.std(viewer_counts),
            np.max(viewer_counts),
            np.min(viewer_counts)
        ])
        
        # Time-based features
        current_hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        # One-hot encode time features
        hour_features = [0] * 24
        hour_features[current_hour] = 1
        features.extend(hour_features)
        
        day_features = [0] * 7
        day_features[day_of_week] = 1
        features.extend(day_features)
        
        # Pad or truncate to expected size
        expected_size = 128
        if len(features) < expected_size:
            features.extend([0] * (expected_size - len(features)))
        else:
            features = features[:expected_size]
            
        return np.array(features)
        
    def _get_content_recommendation(self, score: float) -> str:
        """Get content recommendation based on score"""
        if score < 0.3:
            return "interactive_game"
        elif score < 0.6:
            return "discussion_topic"
        else:
            return "high_energy_content"
            
    def _aggregate_stream_metrics(self):
        """Aggregate stream metrics for reporting"""
        try:
            if len(self.stream_metrics['timestamps']) < 2:
                return
                
            # Calculate aggregate metrics
            metrics = {
                'avg_chat_rate': np.mean(list(self.stream_metrics['chat_rate'])),
                'avg_viewer_count': np.mean(list(self.stream_metrics['viewer_count'])),
                'peak_viewers': np.max(list(self.stream_metrics['viewer_count'])),
                'total_messages': len(self.message_buffer),
                'unique_chatters': len(self.viewer_metrics),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in Redis if available
            if self.redis_pool:
                asyncio.create_task(self._store_metrics_redis(metrics))
                
            logger.info(f"Aggregated metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            
    async def _store_metrics_redis(self, metrics: Dict):
        """Store metrics in Redis"""
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Store with timestamp key
            key = f"stream_metrics:{metrics['timestamp']}"
            await redis.setex(key, 3600, json.dumps(metrics))  # Expire after 1 hour
            
            # Update running averages
            await redis.hincrby('stream_stats', 'total_messages', int(metrics['total_messages']))
            await redis.hincrbyfloat('stream_stats', 'total_viewers', metrics['avg_viewer_count'])
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
            
    async def get_viewer_insights(self, username: str) -> Dict[str, Any]:
        """Get detailed insights for a specific viewer"""
        user_data = self.viewer_metrics.get(username, {})
        
        if not user_data or not user_data['messages']:
            return {'status': 'no_data'}
            
        # Calculate statistics
        message_count = len(user_data['messages'])
        avg_message_length = np.mean([len(msg) for msg in user_data['messages']])
        
        # Activity pattern analysis
        if user_data['timestamps']:
            timestamps = list(user_data['timestamps'])
            hours = [ts.hour for ts in timestamps]
            
            # Find most active hour
            hour_counts = defaultdict(int)
            for hour in hours:
                hour_counts[hour] += 1
                
            most_active_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
        else:
            most_active_hour = None
            
        # Topic analysis
        all_messages = ' '.join(user_data['messages'])
        topics = self._extract_topics([all_messages])
        
        return {
            'username': username,
            'message_count': message_count,
            'avg_message_length': avg_message_length,
            'engagement_score': user_data.get('engagement_score', 0.0),
            'most_active_hour': most_active_hour,
            'favorite_topics': topics,
            'last_seen': user_data['timestamps'][-1].isoformat() if user_data['timestamps'] else None
        }
        
    async def forecast_metrics(self, metric_name: str, periods: int = 24) -> Dict[str, Any]:
        """Forecast future metrics using Prophet"""
        try:
            # Prepare data for Prophet
            if metric_name not in self.stream_metrics:
                return {'error': 'Invalid metric name'}
                
            metric_data = list(self.stream_metrics[metric_name])
            timestamps = list(self.stream_metrics['timestamps'])
            
            if len(metric_data) < 100:
                return {'error': 'Insufficient data for forecasting'}
                
            # Create DataFrame
            df = pd.DataFrame({
                'ds': timestamps,
                'y': metric_data
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(df)
            
            # Make predictions
            future = model.make_future_dataframe(periods=periods, freq='H')
            forecast = model.predict(future)
            
            # Extract relevant predictions
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            return {
                'metric': metric_name,
                'forecasts': predictions.to_dict('records'),
                'current_value': metric_data[-1],
                'trend': 'increasing' if forecast['trend'].iloc[-1] > forecast['trend'].iloc[-periods] else 'decreasing'
            }
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            return {'error': str(e)}
            
    async def detect_content_performance(self, content_id: str, 
                                       engagement_metrics: Dict) -> Dict[str, Any]:
        """Analyze content performance in real-time"""
        # Update content performance metrics
        self.content_performance[content_id]['views'] += 1
        self.content_performance[content_id]['engagement'] = (
            0.9 * self.content_performance[content_id]['engagement'] +
            0.1 * engagement_metrics.get('engagement_rate', 0.0)
        )
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(content_id)
        
        # Determine if content is performing well
        all_scores = [
            self._calculate_performance_score(cid) 
            for cid in self.content_performance.keys()
        ]
        
        if all_scores:
            percentile = stats.percentileofscore(all_scores, performance_score)
            is_top_performer = percentile > 75
        else:
            percentile = 50
            is_top_performer = False
            
        # Generate recommendations
        recommendations = []
        
        if is_top_performer:
            recommendations.append({
                'action': 'extend_content',
                'reason': 'High engagement detected'
            })
        elif percentile < 25:
            recommendations.append({
                'action': 'switch_content',
                'reason': 'Low engagement detected'
            })
            
        return {
            'content_id': content_id,
            'performance_score': performance_score,
            'percentile': percentile,
            'is_top_performer': is_top_performer,
            'recommendations': recommendations,
            'metrics': self.content_performance[content_id]
        }
        
    def _calculate_performance_score(self, content_id: str) -> float:
        """Calculate overall performance score for content"""
        metrics = self.content_performance[content_id]
        
        # Normalize metrics
        view_score = np.tanh(metrics['views'] / 100)  # Normalize views
        engagement_score = metrics['engagement']
        retention_score = metrics['retention']
        
        # Weighted combination
        performance_score = (
            view_score * 0.3 +
            engagement_score * 0.5 +
            retention_score * 0.2
        )
        
        return float(performance_score)
        
    async def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get data for real-time dashboard"""
        # Current metrics
        current_metrics = {
            'viewer_count': list(self.stream_metrics['viewer_count'])[-1] if self.stream_metrics['viewer_count'] else 0,
            'chat_rate': list(self.stream_metrics['chat_rate'])[-1] if self.stream_metrics['chat_rate'] else 0,
            'active_chatters': len([u for u, d in self.viewer_metrics.items() if d['engagement_score'] > 0.5]),
            'total_messages': len(self.message_buffer)
        }
        
        # Trends (last hour)
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_indices = [
            i for i, ts in enumerate(self.stream_metrics['timestamps'])
            if ts > hour_ago
        ]
        
        if recent_indices:
            viewer_trend = [
                list(self.stream_metrics['viewer_count'])[i] 
                for i in recent_indices
            ]
            chat_trend = [
                list(self.stream_metrics['chat_rate'])[i] 
                for i in recent_indices
            ]
        else:
            viewer_trend = []
            chat_trend = []
            
        # Top chatters
        top_chatters = sorted(
            [(u, d['engagement_score']) for u, d in self.viewer_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Recent predictions
        recent_predictions = []
        try:
            while not self.prediction_queue.empty():
                recent_predictions.append(self.prediction_queue.get_nowait())
        except queue.Empty:
            pass
            
        return {
            'current_metrics': current_metrics,
            'trends': {
                'viewer_count': viewer_trend,
                'chat_rate': chat_trend
            },
            'top_chatters': [
                {'username': u, 'engagement': e} 
                for u, e in top_chatters
            ],
            'recent_predictions': recent_predictions[-5:] if recent_predictions else [],
            'timestamp': datetime.now().isoformat()
        }
        
    async def start_websocket_server(self, host='localhost', port=8765):
        """Start WebSocket server for real-time updates"""
        async def handler(websocket, path):
            try:
                while True:
                    # Send dashboard update
                    dashboard_data = await self.get_real_time_dashboard()
                    await websocket.send(json.dumps(dashboard_data))
                    
                    # Wait before next update
                    await asyncio.sleep(1)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                
        server = await websockets.serve(handler, host, port)
        logger.info(f"WebSocket server started on {host}:{port}")
        
        await server.wait_closed()