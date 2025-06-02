# sentiment_analysis.py
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline
)
from sentence_transformers import SentenceTransformer
import asyncio
import aiohttp
import redis
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from collections import deque, defaultdict
import re
from textblob import TextBlob
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSentimentAnalyzer:
    def __init__(self, redis_host='localhost', redis_port=6379, cache_ttl=3600):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.bert_sentiment = AutoModelForSequenceClassification.from_pretrained(
            'nlptown/bert-base-multilingual-uncased-sentiment'
        ).to(self.device)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            'nlptown/bert-base-multilingual-uncased-sentiment'
        )
        
        # Emotion detection model
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
            'j-hartmann/emotion-english-distilroberta-base'
        ).to(self.device)
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(
            'j-hartmann/emotion-english-distilroberta-base'
        )
        
        # Toxicity detection
        self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
            'unitary/toxic-bert'
        ).to(self.device)
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained('unitary/toxic-bert')
        
        # Sentence embeddings for semantic similarity
        self.sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
        
        # Additional analyzers
        self.vader = SentimentIntensityAnalyzer()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Redis for caching
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.cache_ttl = cache_ttl
        
        # Real-time processing queues
        self.message_queue = queue.Queue(maxsize=1000)
        self.result_queue = queue.Queue(maxsize=1000)
        
        # Conversation context tracking
        self.conversation_buffer = deque(maxlen=50)
        self.user_sentiment_history = defaultdict(lambda: deque(maxlen=10))
        self.global_sentiment_trend = deque(maxlen=100)
        
        # Emotion labels
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Start background processing thread
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processing_thread.start()
        
        # TF-IDF for topic modeling
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.topic_buffer = deque(maxlen=200)
        
    def _process_messages(self):
        """Background thread for processing messages"""
        while True:
            try:
                if not self.message_queue.empty():
                    message_data = self.message_queue.get(timeout=0.1)
                    result = self._analyze_message_sync(
                        message_data['text'],
                        message_data['username'],
                        message_data.get('context', {})
                    )
                    self.result_queue.put({
                        'id': message_data['id'],
                        'result': result
                    })
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                
    async def analyze_message(self, text: str, username: str, context: Dict = None) -> Dict[str, Any]:
        """Async wrapper for message analysis"""
        message_id = f"{username}_{datetime.now().timestamp()}"
        
        # Check cache first
        cache_key = f"sentiment:{hash(text)}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Queue for processing
        self.message_queue.put({
            'id': message_id,
            'text': text,
            'username': username,
            'context': context or {}
        })
        
        # Wait for result
        start_time = time.time()
        while time.time() - start_time < 5:  # 5 second timeout
            try:
                while not self.result_queue.empty():
                    result_data = self.result_queue.get_nowait()
                    if result_data['id'] == message_id:
                        # Cache result
                        self.redis_client.setex(
                            cache_key, 
                            self.cache_ttl, 
                            json.dumps(result_data['result'])
                        )
                        return result_data['result']
            except queue.Empty:
                await asyncio.sleep(0.01)
                
        # Timeout fallback
        return self._get_simple_sentiment(text)
        
    def _analyze_message_sync(self, text: str, username: str, context: Dict) -> Dict[str, Any]:
        """Synchronous message analysis"""
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Multi-model sentiment analysis
        bert_sentiment = self._get_bert_sentiment(cleaned_text)
        emotion = self._get_emotion(cleaned_text)
        toxicity = self._get_toxicity(cleaned_text)
        vader_sentiment = self.vader.polarity_scores(cleaned_text)
        
        # Entity and keyword extraction
        entities = self._extract_entities(cleaned_text)
        keywords = self._extract_keywords(cleaned_text)
        
        # Emoji analysis
        emoji_sentiment = self._analyze_emojis(text)
        
        # Conversation context
        conv_context = self._get_conversation_context(username, cleaned_text)
        
        # Update histories
        self._update_sentiment_history(username, bert_sentiment['score'])
        
        # Aggregate results
        aggregate_sentiment = self._aggregate_sentiments(
            bert_sentiment, vader_sentiment, emoji_sentiment
        )
        
        # Determine response recommendation
        response_rec = self._get_response_recommendation(
            aggregate_sentiment, emotion, toxicity, conv_context
        )
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'sentiment': {
                'aggregate': aggregate_sentiment,
                'bert': bert_sentiment,
                'vader': vader_sentiment,
                'emoji': emoji_sentiment
            },
            'emotion': emotion,
            'toxicity': toxicity,
            'entities': entities,
            'keywords': keywords,
            'conversation_context': conv_context,
            'response_recommendation': response_rec,
            'user_sentiment_trend': self._get_user_trend(username)
        }
        
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Convert emojis to text description
        text = emoji.demojize(text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!?]{2,}', '!', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
        
    def _get_bert_sentiment(self, text: str) -> Dict[str, float]:
        """Get BERT-based sentiment scores"""
        inputs = self.bert_tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_sentiment(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convert 5-star rating to sentiment score
        star_scores = scores[0].cpu().numpy()
        sentiment_score = np.sum(star_scores * np.array([1, 2, 3, 4, 5])) / 5.0
        
        return {
            'score': float(sentiment_score),
            'confidence': float(torch.max(scores).item()),
            'distribution': star_scores.tolist()
        }
        
    def _get_emotion(self, text: str) -> Dict[str, Any]:
        """Detect emotion using DistilRoBERTa"""
        inputs = self.emotion_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        emotion_scores = scores[0].cpu().numpy()
        primary_emotion = self.emotion_labels[np.argmax(emotion_scores)]
        
        return {
            'primary': primary_emotion,
            'scores': {
                label: float(score) 
                for label, score in zip(self.emotion_labels, emotion_scores)
            },
            'confidence': float(np.max(emotion_scores))
        }
        
    def _get_toxicity(self, text: str) -> Dict[str, float]:
        """Detect toxicity levels"""
        inputs = self.toxicity_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.toxicity_model(**inputs)
            scores = torch.nn.functional.sigmoid(outputs.logits)
            
        toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        toxicity_scores = scores[0].cpu().numpy()
        
        return {
            'overall': float(np.max(toxicity_scores)),
            'categories': {
                label: float(score)
                for label, score in zip(toxicity_labels, toxicity_scores)
            },
            'is_toxic': bool(np.max(toxicity_scores) > 0.7)
        }
        
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            
        return entities
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF"""
        self.topic_buffer.append(text)
        
        if len(self.topic_buffer) < 10:
            # Simple keyword extraction for small corpus
            doc = self.nlp(text)
            return [token.text for token in doc if token.pos_ in ['NOUN', 'VERB'] and len(token.text) > 3][:5]
            
        # TF-IDF based extraction
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(list(self.topic_buffer))
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top keywords for current text
            doc_tfidf = tfidf_matrix[-1]
            scores = zip(feature_names, doc_tfidf.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            return [word for word, score in sorted_scores[:5] if score > 0]
        except:
            return []
            
    def _analyze_emojis(self, text: str) -> Dict[str, float]:
        """Analyze emoji sentiment"""
        emojis = [char for char in text if char in emoji.EMOJI_DATA]
        
        if not emojis:
            return {'score': 0.5, 'count': 0}
            
        # Simple emoji sentiment mapping
        positive_emojis = ['😊', '😄', '❤️', '👍', '🎉', '✨', '💖', '🔥']
        negative_emojis = ['😢', '😭', '😡', '👎', '💔', '😤', '😞']
        
        positive_count = sum(1 for e in emojis if e in positive_emojis)
        negative_count = sum(1 for e in emojis if e in negative_emojis)
        
        if positive_count + negative_count == 0:
            score = 0.5
        else:
            score = positive_count / (positive_count + negative_count)
            
        return {
            'score': score,
            'count': len(emojis),
            'positive': positive_count,
            'negative': negative_count
        }
        
    def _get_conversation_context(self, username: str, text: str) -> Dict[str, Any]:
        """Analyze conversation context"""
        self.conversation_buffer.append({
            'username': username,
            'text': text,
            'timestamp': datetime.now()
        })
        
        # Get recent messages
        recent_messages = list(self.conversation_buffer)[-10:]
        
        # Calculate conversation metrics
        user_messages = [m for m in recent_messages if m['username'] == username]
        message_frequency = len(user_messages) / max(1, len(recent_messages))
        
        # Semantic similarity to recent messages
        if len(recent_messages) > 1:
            embeddings = self.sentence_transformer.encode(
                [m['text'] for m in recent_messages]
            )
            
            current_embedding = embeddings[-1]
            other_embeddings = embeddings[:-1]
            
            similarities = cosine_similarity(
                [current_embedding], 
                other_embeddings
            )[0]
            
            avg_similarity = float(np.mean(similarities))
            max_similarity = float(np.max(similarities))
        else:
            avg_similarity = 0.0
            max_similarity = 0.0
            
        return {
            'message_frequency': message_frequency,
            'recent_message_count': len(user_messages),
            'semantic_similarity': {
                'average': avg_similarity,
                'max': max_similarity
            },
            'is_conversation_continuation': max_similarity > 0.7
        }
        
    def _update_sentiment_history(self, username: str, sentiment_score: float):
        """Update user sentiment history"""
        self.user_sentiment_history[username].append({
            'score': sentiment_score,
            'timestamp': datetime.now()
        })
        
        self.global_sentiment_trend.append(sentiment_score)
        
    def _get_user_trend(self, username: str) -> Dict[str, Any]:
        """Get user sentiment trend"""
        history = list(self.user_sentiment_history[username])
        
        if len(history) < 2:
            return {'trend': 'neutral', 'change': 0.0}
            
        recent_scores = [h['score'] for h in history[-5:]]
        older_scores = [h['score'] for h in history[:-5]]
        
        if older_scores:
            recent_avg = np.mean(recent_scores)
            older_avg = np.mean(older_scores)
            change = recent_avg - older_avg
            
            if change > 0.1:
                trend = 'improving'
            elif change < -0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            change = 0.0
            
        return {
            'trend': trend,
            'change': float(change),
            'average': float(np.mean(recent_scores))
        }
        
    def _aggregate_sentiments(self, bert: Dict, vader: Dict, emoji: Dict) -> Dict[str, float]:
        """Aggregate multiple sentiment scores"""
        # Weighted average
        bert_weight = 0.5
        vader_weight = 0.3
        emoji_weight = 0.2
        
        # Normalize VADER compound score to 0-1
        vader_normalized = (vader['compound'] + 1) / 2
        
        aggregate_score = (
            bert['score'] * bert_weight +
            vader_normalized * vader_weight +
            emoji['score'] * emoji_weight
        )
        
        return {
            'score': float(aggregate_score),
            'label': self._score_to_label(aggregate_score),
            'confidence': float(bert['confidence'])
        }
        
    def _score_to_label(self, score: float) -> str:
        """Convert score to sentiment label"""
        if score >= 0.8:
            return 'very_positive'
        elif score >= 0.6:
            return 'positive'
        elif score >= 0.4:
            return 'neutral'
        elif score >= 0.2:
            return 'negative'
        else:
            return 'very_negative'
            
    def _get_response_recommendation(self, sentiment: Dict, emotion: Dict, 
                                   toxicity: Dict, context: Dict) -> Dict[str, Any]:
        """Generate response recommendations for Hikari"""
        recommendations = {
            'should_respond': True,
            'response_tone': 'neutral',
            'priority': 'normal',
            'suggested_emotions': [],
            'warning_flags': []
        }
        
        # Check toxicity first
        if toxicity['is_toxic']:
            recommendations['should_respond'] = False
            recommendations['warning_flags'].append('toxic_content')
            return recommendations
            
        # Determine response tone based on sentiment and emotion
        if sentiment['label'] in ['very_positive', 'positive']:
            recommendations['response_tone'] = 'enthusiastic'
            recommendations['suggested_emotions'] = ['happy', 'excited']
            
            if emotion['primary'] == 'joy':
                recommendations['priority'] = 'high'
        elif sentiment['label'] in ['very_negative', 'negative']:
            recommendations['response_tone'] = 'supportive'
            recommendations['suggested_emotions'] = ['concerned', 'thoughtful']
            
            if emotion['primary'] in ['sadness', 'anger']:
                recommendations['priority'] = 'high'
        else:
            recommendations['response_tone'] = 'casual'
            recommendations['suggested_emotions'] = ['neutral', 'curious']
            
        # Adjust based on conversation context
        if context['is_conversation_continuation']:
            recommendations['priority'] = 'high'
            
        if context['message_frequency'] > 0.5:
            recommendations['warning_flags'].append('high_frequency_user')
            
        return recommendations
        
    def _get_simple_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback simple sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        return {
            'text': text,
            'sentiment': {
                'aggregate': {
                    'score': (polarity + 1) / 2,
                    'label': 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
                }
            },
            'response_recommendation': {
                'should_respond': True,
                'response_tone': 'neutral'
            }
        }
        
    async def get_chat_mood(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get overall chat mood"""
        recent_sentiments = list(self.global_sentiment_trend)
        
        if not recent_sentiments:
            return {'mood': 'neutral', 'energy': 'low'}
            
        avg_sentiment = np.mean(recent_sentiments)
        sentiment_variance = np.var(recent_sentiments)
        
        # Determine mood
        if avg_sentiment >= 0.7:
            mood = 'very_positive'
        elif avg_sentiment >= 0.55:
            mood = 'positive'
        elif avg_sentiment >= 0.45:
            mood = 'neutral'
        elif avg_sentiment >= 0.3:
            mood = 'negative'
        else:
            mood = 'very_negative'
            
        # Determine energy level
        if sentiment_variance > 0.1:
            energy = 'high'
        elif sentiment_variance > 0.05:
            energy = 'medium'
        else:
            energy = 'low'
            
        return {
            'mood': mood,
            'energy': energy,
            'average_sentiment': float(avg_sentiment),
            'variance': float(sentiment_variance),
            'sample_size': len(recent_sentiments)
        }
        
    async def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Get sentiment profile for a user"""
        history = list(self.user_sentiment_history[username])
        
        if not history:
            return {'profile': 'unknown', 'interaction_count': 0}
            
        scores = [h['score'] for h in history]
        avg_score = np.mean(scores)
        
        # Determine user profile
        if avg_score >= 0.7:
            profile = 'very_positive'
        elif avg_score >= 0.55:
            profile = 'positive'
        elif avg_score >= 0.45:
            profile = 'neutral'
        elif avg_score >= 0.3:
            profile = 'negative'
        else:
            profile = 'very_negative'
            
        return {
            'profile': profile,
            'average_sentiment': float(avg_score),
            'interaction_count': len(history),
            'trend': self._get_user_trend(username)
        }