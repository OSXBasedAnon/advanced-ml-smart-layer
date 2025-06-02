# context_memory.py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
import asyncio
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re
import hashlib
import msgpack
import lmdb
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer
)
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import threading
import queue
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralMemorySystem:
    def __init__(self, memory_dir='./neural_memory', embedding_dim=768):
        self.memory_dir = memory_dir
        self.embedding_dim = embedding_dim
        
        # Create directories
        os.makedirs(memory_dir, exist_ok=True)
        os.makedirs(os.path.join(memory_dir, 'indices'), exist_ok=True)
        os.makedirs(os.path.join(memory_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Sentence embedding model
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Summarization model
        self.summarizer = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)
        self.summarizer_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        # Initialize FAISS indices
        self.short_term_index = faiss.IndexFlatL2(embedding_dim)
        self.long_term_index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(embedding_dim), 
            embedding_dim, 
            100  # Number of clusters
        )
        self.long_term_index.train(np.random.rand(1000, embedding_dim).astype('float32'))
        
        # LMDB for persistent storage
        self.env = lmdb.open(
            os.path.join(memory_dir, 'memory_db'), 
            map_size=10 * 1024 * 1024 * 1024  # 10GB
        )
        
        # Memory structures
        self.short_term_memory = deque(maxlen=1000)
        self.working_memory = {}
        self.episodic_buffer = deque(maxlen=50)
        
        # Semantic network
        self.knowledge_graph = nx.DiGraph()
        
        # Memory metadata
        self.memory_metadata = {}
        self.memory_importance_scores = {}
        self.memory_access_counts = defaultdict(int)
        self.memory_timestamps = {}
        
        # Processing queues
        self.encoding_queue = queue.Queue(maxsize=500)
        self.retrieval_queue = queue.Queue(maxsize=100)
        
        # Start background threads
        self._start_background_processing()
        
        # Memory consolidation parameters
        self.consolidation_threshold = 0.7
        self.decay_rate = 0.995
        
        # Clustering for memory organization
        self.memory_clusters = {}
        self.cluster_centers = []
        
        # Load existing memories
        self._load_memories()
        
    def _start_background_processing(self):
        """Start background processing threads"""
        # Encoding thread
        encoding_thread = threading.Thread(target=self._encoding_worker, daemon=True)
        encoding_thread.start()
        
        # Consolidation thread
        consolidation_thread = threading.Thread(target=self._consolidation_worker, daemon=True)
        consolidation_thread.start()
        
        # Decay thread
        decay_thread = threading.Thread(target=self._decay_worker, daemon=True)
        decay_thread.start()
        
    def _encoding_worker(self):
        """Background worker for encoding memories"""
        while True:
            try:
                if not self.encoding_queue.empty():
                    memory_data = self.encoding_queue.get(timeout=0.1)
                    self._encode_memory_sync(memory_data)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in encoding worker: {e}")
                
    def _consolidation_worker(self):
        """Background worker for memory consolidation"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._consolidate_memories()
            except Exception as e:
                logger.error(f"Error in consolidation worker: {e}")
                
    def _decay_worker(self):
        """Background worker for memory decay"""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self._apply_memory_decay()
            except Exception as e:
                logger.error(f"Error in decay worker: {e}")
                
    async def store_memory(self, content: str, memory_type: str, 
                          metadata: Dict = None, importance: float = 0.5) -> str:
        """Store a memory with semantic encoding"""
        memory_id = self._generate_memory_id(content)
        
        memory_data = {
            'id': memory_id,
            'content': content,
            'type': memory_type,
            'metadata': metadata or {},
            'importance': importance,
            'timestamp': datetime.now().isoformat()
        }
        
        # Queue for encoding
        self.encoding_queue.put(memory_data)
        
        # Add to short-term memory immediately
        self.short_term_memory.append(memory_data)
        
        # Update working memory if high importance
        if importance > 0.7:
            self.working_memory[memory_id] = memory_data
            
        return memory_id
        
    def _encode_memory_sync(self, memory_data: Dict):
        """Synchronously encode and store memory"""
        try:
            # Generate embedding
            embedding = self.sentence_model.encode(memory_data['content'])
            
            # Add to FAISS index
            self.short_term_index.add(np.array([embedding]).astype('float32'))
            
            # Store in LMDB
            with self.env.begin(write=True) as txn:
                # Store memory data
                txn.put(
                    memory_data['id'].encode(),
                    msgpack.packb(memory_data)
                )
                
                # Store embedding
                txn.put(
                    f"emb_{memory_data['id']}".encode(),
                    embedding.tobytes()
                )
                
            # Update metadata
            self.memory_metadata[memory_data['id']] = memory_data['metadata']
            self.memory_importance_scores[memory_data['id']] = memory_data['importance']
            self.memory_timestamps[memory_data['id']] = datetime.now()
            
            # Extract entities and update knowledge graph
            self._update_knowledge_graph(memory_data)
            
            # Check for episodic relevance
            if memory_data['type'] == 'episodic':
                self.episodic_buffer.append(memory_data)
                
        except Exception as e:
            logger.error(f"Error encoding memory: {e}")
            
    def _update_knowledge_graph(self, memory_data: Dict):
        """Update semantic knowledge graph"""
        # Simple entity extraction (could be enhanced with NER)
        entities = re.findall(r'\b[A-Z][a-z]+\b', memory_data['content'])
        
        # Add nodes
        for entity in entities:
            if not self.knowledge_graph.has_node(entity):
                self.knowledge_graph.add_node(entity, memories=[])
            self.knowledge_graph.nodes[entity]['memories'].append(memory_data['id'])
            
        # Add edges between co-occurring entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if self.knowledge_graph.has_edge(entity1, entity2):
                    self.knowledge_graph[entity1][entity2]['weight'] += 1
                else:
                    self.knowledge_graph.add_edge(entity1, entity2, weight=1)
                    
    async def retrieve_memories(self, query: str, k: int = 5, 
                              memory_types: List[str] = None) -> List[Dict]:
        """Retrieve relevant memories using semantic search"""
        # Generate query embedding
        query_embedding = self.sentence_model.encode(query)
        
        # Search in short-term memory
        short_term_results = self._search_short_term(query_embedding, k)
        
        # Search in long-term memory
        long_term_results = self._search_long_term(query_embedding, k)
        
        # Combine and rank results
        all_results = short_term_results + long_term_results
        
        # Filter by memory type if specified
        if memory_types:
            all_results = [r for r in all_results if r.get('type') in memory_types]
            
        # Re-rank by relevance and importance
        ranked_results = self._rank_memories(all_results, query_embedding)
        
        # Update access counts
        for result in ranked_results[:k]:
            self.memory_access_counts[result['id']] += 1
            
        return ranked_results[:k]
        
    def _search_short_term(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Search short-term memory"""
        if self.short_term_index.ntotal == 0:
            return []
            
        # Search FAISS index
        distances, indices = self.short_term_index.search(
            np.array([query_embedding]).astype('float32'), 
            min(k, self.short_term_index.ntotal)
        )
        
        results = []
        for idx in indices[0]:
            if idx < len(self.short_term_memory):
                memory = self.short_term_memory[idx]
                results.append(memory)
                
        return results
        
    def _search_long_term(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Search long-term memory"""
        if self.long_term_index.ntotal == 0:
            return []
            
        # Search FAISS index
        distances, indices = self.long_term_index.search(
            np.array([query_embedding]).astype('float32'),
            min(k, self.long_term_index.ntotal)
        )
        
        results = []
        with self.env.begin() as txn:
            for idx, distance in zip(indices[0], distances[0]):
                # Retrieve memory data from LMDB
                # This is a simplified version - in practice, you'd maintain
                # an index mapping between FAISS indices and memory IDs
                cursor = txn.cursor()
                for key, value in cursor:
                    if not key.startswith(b'emb_'):
                        memory_data = msgpack.unpackb(value)
                        results.append(memory_data)
                        if len(results) >= k:
                            break
                            
        return results
        
    def _rank_memories(self, memories: List[Dict], query_embedding: np.ndarray) -> List[Dict]:
        """Rank memories by relevance and importance"""
        scored_memories = []
        
        for memory in memories:
            # Calculate semantic similarity
            memory_embedding = self._get_embedding(memory['id'])
            if memory_embedding is not None:
                similarity = np.dot(query_embedding, memory_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                )
            else:
                similarity = 0.0
                
            # Calculate recency score
            timestamp = self.memory_timestamps.get(memory['id'], datetime.now())
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            recency_score = np.exp(-age_hours / 168)  # Decay over a week
            
            # Get importance and access frequency
            importance = self.memory_importance_scores.get(memory['id'], 0.5)
            access_count = self.memory_access_counts.get(memory['id'], 0)
            access_score = np.tanh(access_count / 10)  # Normalize access count
            
            # Combined score
            score = (
                0.4 * similarity +
                0.2 * recency_score +
                0.2 * importance +
                0.2 * access_score
            )
            
            scored_memories.append((memory, score))
            
        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, score in scored_memories]
        
    def _get_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding for a memory"""
        with self.env.begin() as txn:
            embedding_bytes = txn.get(f"emb_{memory_id}".encode())
            if embedding_bytes:
                return np.frombuffer(embedding_bytes, dtype=np.float32)
        return None
        
    def _consolidate_memories(self):
        """Consolidate short-term memories to long-term"""
        logger.info("Starting memory consolidation...")
        
        # Get memories ready for consolidation
        consolidation_candidates = []
        
        for memory in list(self.short_term_memory):
            # Check importance and access frequency
            importance = self.memory_importance_scores.get(memory['id'], 0.5)
            access_count = self.memory_access_counts.get(memory['id'], 0)
            
            # Calculate consolidation score
            consolidation_score = importance * 0.6 + np.tanh(access_count / 5) * 0.4
            
            if consolidation_score > self.consolidation_threshold:
                consolidation_candidates.append(memory)
                
        # Consolidate memories
        for memory in consolidation_candidates:
            try:
                # Get embedding
                embedding = self._get_embedding(memory['id'])
                if embedding is not None:
                    # Add to long-term index
                    self.long_term_index.add(np.array([embedding]).astype('float32'))
                    
                    # Remove from short-term
                    self.short_term_memory.remove(memory)
                    
                    logger.info(f"Consolidated memory {memory['id']} to long-term storage")
            except Exception as e:
                logger.error(f"Error consolidating memory {memory['id']}: {e}")
                
        # Cluster memories for better organization
        self._cluster_memories()
        
    def _cluster_memories(self):
        """Cluster memories for efficient retrieval"""
        # Get all embeddings
        embeddings = []
        memory_ids = []
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b'emb_'):
                    memory_id = key[4:].decode()
                    embedding = np.frombuffer(value, dtype=np.float32)
                    embeddings.append(embedding)
                    memory_ids.append(memory_id)
                    
        if len(embeddings) < 10:
            return
            
        # Apply DBSCAN clustering
        embeddings_array = np.array(embeddings)
        clustering = DBSCAN(eps=0.3, min_samples=5, metric='cosine')
        labels = clustering.fit_predict(embeddings_array)
        
        # Update cluster assignments
        self.memory_clusters = defaultdict(list)
        for memory_id, label in zip(memory_ids, labels):
            self.memory_clusters[label].append(memory_id)
            
        # Calculate cluster centers
        unique_labels = set(labels) - {-1}  # Exclude noise
        self.cluster_centers = []
        
        for label in unique_labels:
            cluster_embeddings = embeddings_array[labels == label]
            center = np.mean(cluster_embeddings, axis=0)
            self.cluster_centers.append((label, center))
            
        logger.info(f"Organized memories into {len(unique_labels)} clusters")
        
    def _apply_memory_decay(self):
        """Apply forgetting curve to memories"""
        logger.info("Applying memory decay...")
        
        current_time = datetime.now()
        memories_to_forget = []
        
        for memory_id, timestamp in self.memory_timestamps.items():
            # Calculate age
            age_days = (current_time - timestamp).days
            
            # Get current importance
            importance = self.memory_importance_scores.get(memory_id, 0.5)
            
            # Apply decay
            decayed_importance = importance * (self.decay_rate ** age_days)
            
            # Update importance
            self.memory_importance_scores[memory_id] = decayed_importance
            
            # Mark for forgetting if below threshold
            if decayed_importance < 0.1:
                memories_to_forget.append(memory_id)
                
        # Forget memories
        for memory_id in memories_to_forget:
            self._forget_memory(memory_id)
            
        logger.info(f"Forgot {len(memories_to_forget)} memories due to decay")
        
    def _forget_memory(self, memory_id: str):
        """Remove a memory from the system"""
        try:
            # Remove from LMDB
            with self.env.begin(write=True) as txn:
                txn.delete(memory_id.encode())
                txn.delete(f"emb_{memory_id}".encode())
                
            # Remove from metadata
            self.memory_metadata.pop(memory_id, None)
            self.memory_importance_scores.pop(memory_id, None)
            self.memory_timestamps.pop(memory_id, None)
            self.memory_access_counts.pop(memory_id, None)
            
            # Remove from working memory
            self.working_memory.pop(memory_id, None)
            
            logger.debug(f"Forgot memory {memory_id}")
        except Exception as e:
            logger.error(f"Error forgetting memory {memory_id}: {e}")
            
    async def get_context_summary(self, topic: str, max_length: int = 500) -> str:
        """Generate a summary of memories related to a topic"""
        # Retrieve relevant memories
        memories = await self.retrieve_memories(topic, k=10)
        
        if not memories:
            return f"No memories found related to {topic}"
            
        # Combine memory contents
        combined_text = " ".join([m['content'] for m in memories[:5]])
        
        # Generate summary using T5
        input_text = f"summarize: {combined_text}"
        inputs = self.summarizer_tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.summarizer.generate(
                inputs,
                max_length=max_length,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
        summary = self.summarizer_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
        return summary
        
    async def find_associations(self, entity: str, max_hops: int = 2) -> List[Tuple[str, float]]:
        """Find associated entities using knowledge graph"""
        if not self.knowledge_graph.has_node(entity):
            return []
            
        associations = []
        visited = set()
        
        # BFS with decreasing weights
        queue = [(entity, 1.0, 0)]
        
        while queue:
            current, weight, hops = queue.pop(0)
            
            if current in visited or hops > max_hops:
                continue
                
            visited.add(current)
            
            if current != entity:
                associations.append((current, weight))
                
            # Add neighbors
            if hops < max_hops:
                for neighbor in self.knowledge_graph.neighbors(current):
                    if neighbor not in visited:
                        edge_weight = self.knowledge_graph[current][neighbor]['weight']
                        new_weight = weight * (0.7 ** hops) * np.tanh(edge_weight / 10)
                        queue.append((neighbor, new_weight, hops + 1))
                        
        # Sort by weight
        associations.sort(key=lambda x: x[1], reverse=True)
        
        return associations[:10]
        
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        with self.env.begin() as txn:
            total_memories = txn.stat()['entries'] // 2  # Divide by 2 for data+embedding
            
        return {
            'total_memories': total_memories,
            'short_term_count': len(self.short_term_memory),
            'working_memory_count': len(self.working_memory),
            'episodic_buffer_count': len(self.episodic_buffer),
            'knowledge_graph_nodes': self.knowledge_graph.number_of_nodes(),
            'knowledge_graph_edges': self.knowledge_graph.number_of_edges(),
            'cluster_count': len(self.cluster_centers),
            'average_importance': np.mean(list(self.memory_importance_scores.values())) if self.memory_importance_scores else 0.0
        }
        
    def _generate_memory_id(self, content: str) -> str:
        """Generate unique memory ID"""
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"mem_{timestamp}_{content_hash}"
        
    def _load_memories(self):
        """Load existing memories from disk"""
        try:
            # Load FAISS indices if they exist
            short_term_path = os.path.join(self.memory_dir, 'indices', 'short_term.faiss')
            long_term_path = os.path.join(self.memory_dir, 'indices', 'long_term.faiss')
            
            if os.path.exists(short_term_path):
                self.short_term_index = faiss.read_index(short_term_path)
                logger.info(f"Loaded short-term index with {self.short_term_index.ntotal} vectors")
                
            if os.path.exists(long_term_path):
                self.long_term_index = faiss.read_index(long_term_path)
                logger.info(f"Loaded long-term index with {self.long_term_index.ntotal} vectors")
                
            # Load metadata
            metadata_path = os.path.join(self.memory_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.memory_metadata = saved_data.get('metadata', {})
                    self.memory_importance_scores = saved_data.get('importance', {})
                    self.memory_timestamps = saved_data.get('timestamps', {})
                    self.memory_access_counts = defaultdict(int, saved_data.get('access_counts', {}))
                    
            # Load knowledge graph
            graph_path = os.path.join(self.memory_dir, 'knowledge_graph.pkl')
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                    
            logger.info("Successfully loaded existing memories")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            
    def save_memories(self):
        """Save memories to disk"""
        try:
            # Save FAISS indices
            indices_dir = os.path.join(self.memory_dir, 'indices')
            faiss.write_index(self.short_term_index, os.path.join(indices_dir, 'short_term.faiss'))
            faiss.write_index(self.long_term_index, os.path.join(indices_dir, 'long_term.faiss'))
            
            # Save metadata
            saved_data = {
                'metadata': self.memory_metadata,
                'importance': self.memory_importance_scores,
                'timestamps': self.memory_timestamps,
                'access_counts': dict(self.memory_access_counts)
            }
            
            with open(os.path.join(self.memory_dir, 'metadata.pkl'), 'wb') as f:
                pickle.dump(saved_data, f)
                
            # Save knowledge graph
            with open(os.path.join(self.memory_dir, 'knowledge_graph.pkl'), 'wb') as f:
                pickle.dump(self.knowledge_graph, f)
                
            logger.info("Successfully saved memories to disk")
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
