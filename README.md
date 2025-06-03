# Hikari AI VTuber - Advanced ML Intelligence Layer

## Overview

This repository contains Python modules that provide an advanced Machine Learning intelligence layer for Hikari A.I. These modules leverage the latest developments in Natural Language Processing (NLP), neural memory systems, and real-time analytics to create a sophisticated, context-aware AI companion.

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Hikari AI VTuber System                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │   Sentiment    │  │    Context      │  │  Real-time   │  │
│  │   Analysis     │  │    Memory       │  │  Analytics   │  │
│  │                │  │                 │  │              │  │
│  │ • BERT Models  │  │ • FAISS Vectors │  │ • Prophet    │  │
│  │ • Multi-model  │  │ • LMDB Storage  │  │ • PyTorch    │  │
│  │ • Real-time    │  │ • Knowledge     │  │ • WebSocket  │  │
│  │                │  │   Graph         │  │              │  │
│  └────────────────┘  └─────────────────┘  └──────────────┘  │
│                             ↓                               │
│                    Rust Engine Integration                  │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Table of Contents

- [Core Technologies](#-core-technologies)
- [Module Breakdown](#-module-breakdown)
- [Installation](#-installation)
- [Technical Deep Dive](#-technical-deep-dive)
- [Performance Optimizations](#-performance-optimizations)
- [Why These Technologies?](#-why-these-technologies)
- [Usage Examples](#-usage-examples)
- [API Reference](#-api-reference)

## 🔧 Core Technologies

### State-of-the-Art Models
- **BERT (Bidirectional Encoder Representations from Transformers)**: Latest multilingual sentiment analysis
- **DistilRoBERTa**: Emotion detection with 7-class classification
- **T5 (Text-to-Text Transfer Transformer)**: Neural summarization
- **Sentence-BERT**: Semantic similarity and embeddings

### Advanced Storage & Retrieval
- **FAISS (Facebook AI Similarity Search)**: Billion-scale vector similarity search
- **LMDB (Lightning Memory-Mapped Database)**: Ultra-fast key-value storage
- **Redis**: Real-time caching and pub/sub messaging

### Neural Architecture
- **PyTorch**: Custom LSTM + Attention networks
- **Prophet**: Facebook's time-series forecasting
- **Isolation Forest**: Anomaly detection
- **DBSCAN**: Density-based clustering

## 📦 Module Breakdown

### 1. `sentiment_analysis.py` - Advanced Multi-Model Sentiment Engine

**Purpose**: Real-time emotional intelligence and sentiment understanding

**Key Features**:
- **Multi-model Ensemble**: Combines BERT, VADER, and emoji analysis for 360° sentiment understanding
- **Emotion Detection**: 7-class emotion classification (anger, disgust, fear, joy, neutral, sadness, surprise)
- **Toxicity Detection**: Protects Hikari from harmful content using toxic-bert
- **Conversation Context**: Tracks sentiment trends per user and globally
- **Real-time Processing**: Asynchronous queue-based architecture for <100ms response times

**Technical Highlights**:
```python
# Weighted ensemble approach
aggregate_score = (
    bert['score'] * 0.5 +      # Deep contextual understanding
    vader_normalized * 0.3 +    # Rule-based sentiment
    emoji['score'] * 0.2        # Emoji sentiment analysis
)
```

### 2. `context_memory.py` - Neural Memory System

**Purpose**: Human-like memory with encoding, consolidation, and retrieval

**Key Features**:
- **Three-tier Memory Architecture**:
  - Short-term memory (working buffer)
  - Long-term memory (consolidated important memories)
  - Episodic buffer (narrative memory)
- **Semantic Knowledge Graph**: NetworkX-based entity relationships
- **Memory Consolidation**: Mimics human memory with importance-based transfer
- **Forgetting Curve**: Implements Ebbinghaus forgetting curve for realistic memory decay
- **Vector Similarity Search**: FAISS-powered semantic memory retrieval

**Technical Highlights**:
```python
# Memory decay implementation
decayed_importance = importance * (self.decay_rate ** age_days)

# FAISS indexing for similarity search
self.long_term_index = faiss.IndexIVFFlat(
    faiss.IndexFlatL2(embedding_dim), 
    embedding_dim, 
    100  # Number of clusters
)
```

### 3. `realtime_analytics.py` - Stream Intelligence Engine

**Purpose**: Real-time viewer engagement prediction and content optimization

**Key Features**:
- **Custom Neural Architecture**: LSTM + Multi-head Attention for engagement prediction
- **Time-series Forecasting**: Prophet-based viewer count and chat rate predictions
- **Anomaly Detection**: Isolation Forest for unusual behavior detection
- **WebSocket Server**: Real-time dashboard updates
- **Content Performance Tracking**: A/B testing and performance analytics

**Technical Highlights**:
```python
class ViewerEngagementPredictor(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
```

## 🚀 Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version

# CUDA (optional but recommended for GPU acceleration)
nvidia-smi
```

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
faiss-gpu>=1.7.2  # or faiss-cpu
lmdb>=1.3.0
redis>=4.3.0
prophet>=1.1
scikit-learn>=1.0.0
spacy>=3.0.0
python -m spacy download en_core_web_sm
vaderSentiment>=3.3.2
emoji>=2.0.0
networkx>=2.8
plotly>=5.0.0
websockets>=10.0
aioredis>=2.0.0
msgpack>=1.0.0
```

## 🔬 Technical Deep Dive

### Why BERT for Sentiment Analysis?

BERT revolutionized NLP by introducing bidirectional context understanding. Unlike traditional models that read text sequentially, BERT processes entire sequences simultaneously, capturing nuanced relationships between words.

```python
# BERT processes context bidirectionally
"I love this game" → Positive (considering full context)
"I love this game... NOT!" → Negative (understanding sarcasm through context)
```

### FAISS: The Secret to Fast Similarity Search

Facebook AI Similarity Search (FAISS) enables searching through millions of vectors in milliseconds:

1. **Flat Index**: Exact search for small datasets
2. **IVF (Inverted File) Index**: Approximate search for large datasets
3. **GPU Acceleration**: 10-100x speedup for similarity computations

### Memory Consolidation Algorithm

Inspired by neuroscience, our memory system mimics human memory consolidation:

```python
# Consolidation score calculation
consolidation_score = importance * 0.6 + np.tanh(access_count / 5) * 0.4

# Only important, frequently accessed memories move to long-term storage
if consolidation_score > threshold:
    transfer_to_long_term(memory)
```

### Real-time Processing Pipeline

```
Message → Queue → Background Thread → Feature Extraction → 
    ↓
    ├── Sentiment Analysis (BERT)
    ├── Emotion Detection (DistilRoBERTa)
    ├── Toxicity Check (Toxic-BERT)
    └── Context Update
    ↓
Response Recommendation → Hikari Engine
```

## ⚡ Performance Optimizations

### 1. GPU Acceleration
- All models automatically use CUDA if available
- Batch processing for multiple messages
- Mixed precision training (FP16) support

### 2. Caching Strategy
- Redis for frequently accessed data
- LRU cache for embeddings
- Pre-computed features for common phrases

### 3. Asynchronous Processing
- Non-blocking I/O for all external calls
- Queue-based architecture prevents bottlenecks
- Background threads for heavy computations

### 4. Memory Efficiency
- Circular buffers for streaming data
- Automatic memory cleanup
- Compression for long-term storage

## 🤔 Why These Technologies?

### BERT vs Traditional NLP
- **Context Understanding**: BERT understands "bank" differently in "river bank" vs "money bank"
- **Transfer Learning**: Pre-trained on massive datasets, fine-tunable for specific tasks
- **Multilingual Support**: Can understand multiple languages without separate models

### FAISS vs Traditional Databases
- **Vector Search**: Finds semantically similar content, not just keyword matches
- **Scalability**: Handles billions of vectors efficiently
- **Approximate Search**: Trades minimal accuracy for 100x speed improvement

### Prophet vs ARIMA
- **Automatic Seasonality**: Detects daily, weekly, yearly patterns automatically
- **Robust to Missing Data**: Handles gaps in time series gracefully
- **Intuitive Parameters**: Easy to tune for domain experts

## 💻 Usage Examples

### Basic Sentiment Analysis
```python
# Initialize the sentiment analyzer
analyzer = AdvancedSentimentAnalyzer()

# Analyze a message
result = await analyzer.analyze_message(
    text="This stream is absolutely amazing! 🔥",
    username="viewer123",
    context={"stream_game": "Minecraft"}
)

print(result['sentiment']['aggregate'])  # {'score': 0.92, 'label': 'very_positive'}
print(result['emotion']['primary'])      # 'joy'
print(result['response_recommendation']) # {'tone': 'enthusiastic', 'priority': 'high'}
```

### Memory Storage and Retrieval
```python
# Initialize memory system
memory = NeuralMemorySystem()

# Store a memory
memory_id = await memory.store_memory(
    content="Viewer asked about favorite anime",
    memory_type="interaction",
    metadata={"topic": "anime", "user": "otaku_fan"},
    importance=0.8
)

# Retrieve related memories
memories = await memory.retrieve_memories(
    query="What anime do you like?",
    k=5,
    memory_types=["interaction", "preference"]
)
```

### Real-time Analytics
```python
# Initialize analytics engine
analytics = StreamAnalyticsEngine()

# Process chat message
insights = await analytics.process_chat_message(
    username="active_viewer",
    message="POG! This gameplay is insane!",
    timestamp=datetime.now()
)

# Get viewer insights
profile = await analytics.get_viewer_insights("active_viewer")
print(profile['engagement_score'])  # 0.85
print(profile['favorite_topics'])   # [('gameplay', 0.9), ('pog', 0.7)]

# Forecast metrics
forecast = await analytics.forecast_metrics(
    metric_name="viewer_count",
    periods=24  # Next 24 hours
)
```

## 📊 API Reference

### SentimentAnalyzer

```python
async def analyze_message(text: str, username: str, context: Dict) -> Dict[str, Any]
```
Analyzes sentiment, emotion, and toxicity of a message.

**Returns**:
- `sentiment`: Aggregate and individual model scores
- `emotion`: Primary emotion and confidence scores
- `toxicity`: Toxicity detection results
- `response_recommendation`: Suggested response parameters

### NeuralMemorySystem

```python
async def store_memory(content: str, memory_type: str, metadata: Dict, importance: float) -> str
```
Stores a memory with semantic encoding.

```python
async def retrieve_memories(query: str, k: int, memory_types: List[str]) -> List[Dict]
```
Retrieves semantically similar memories.

### StreamAnalyticsEngine

```python
async def process_chat_message(username: str, message: str, timestamp: datetime) -> Dict[str, Any]
```
Processes chat messages for real-time analytics.

```python
async def forecast_metrics(metric_name: str, periods: int) -> Dict[str, Any]
```
Forecasts future metrics using Prophet.

## 🔮 Future Enhancements

1. **Multimodal Analysis**: Integrate audio/video emotion detection
2. **Federated Learning**: Privacy-preserving model updates
3. **Graph Neural Networks**: Enhanced knowledge graph reasoning
4. **Transformer Memory**: Attention-based memory mechanisms
5. **AutoML Integration**: Automatic model selection and tuning

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any enhancements.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Facebook AI for FAISS and Prophet
- The PyTorch team for the excellent deep learning framework
- The open-source community for continuous improvements

---

Built with ❤️ for Hikari A.I.
