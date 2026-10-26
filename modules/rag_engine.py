"""
RAG (Retrieval-Augmented Generation) Engine for VisionTalk AI
Retrieves factual domain knowledge based on detected objects and conversational queries.
Includes TF-IDF/Cosine Similarity vector retrieval with pure-Python fallback.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DEFAULT_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_base",
    "objects_knowledge.json"
)


class RAGEngine:
    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = kb_path or DEFAULT_KB_PATH
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer: Optional[Any] = None
        self.doc_vectors: Optional[Any] = None
        self.load_knowledge_base()

    def load_knowledge_base(self) -> None:
        """Loads knowledge documents from disk and builds retrieval index."""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        else:
            self.documents = []
        self._build_index()

    def _build_index(self) -> None:
        """Constructs vector representations of documents."""
        if not self.documents:
            self.vectorizer = None
            self.doc_vectors = None
            return

        corpus = []
        for doc in self.documents:
            # Combine title, category, tags, and content for rich matching
            tags_str = " ".join(doc.get("tags", []))
            text = f"{doc.get('title', '')} {doc.get('category', '')} {tags_str} {doc.get('content', '')}"
            corpus.append(text)

        if SKLEARN_AVAILABLE:
            try:
                self.vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2))
                self.doc_vectors = self.vectorizer.fit_transform(corpus)
            except Exception:
                self.vectorizer = None
                self.doc_vectors = None

    def add_document(self, title: str, category: str, content: str, tags: Optional[List[str]] = None, source: str = "User Added") -> Dict[str, Any]:
        """Dynamically adds a new knowledge document and updates the index."""
        doc_id = f"custom_{len(self.documents) + 1}_{re.sub(r'[^a-zA-Z0-9]', '_', title.lower())[:15]}"
        new_doc = {
            "id": doc_id,
            "category": category,
            "title": title,
            "tags": tags or [t.lower().strip() for t in title.split()],
            "content": content,
            "source": source
        }
        self.documents.append(new_doc)
        # Persist back to file
        os.makedirs(os.path.dirname(os.path.abspath(self.kb_path)), exist_ok=True)
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)
        self._build_index()
        return new_doc

    def _fallback_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Lightweight token-overlap fallback when TF-IDF is unavailable."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return []

        scored = []
        for doc in self.documents:
            tags = " ".join(doc.get("tags", []))
            doc_text = f"{doc.get('title', '')} {tags} {doc.get('content', '')}".lower()
            doc_tokens = set(re.findall(r"\w+", doc_text))
            
            # Jaccard overlap + tag matching bonus
            overlap = query_tokens.intersection(doc_tokens)
            tag_overlap = set(t.lower() for t in doc.get("tags", [])).intersection(query_tokens)
            score = (len(overlap) / max(len(query_tokens), 1)) + (len(tag_overlap) * 0.5)

            if score > 0.05:
                res = dict(doc)
                res["score"] = round(min(score, 1.0), 3)
                scored.append(res)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Dict[str, Any]]:
        """Searches the knowledge base using TF-IDF cosine similarity or fallback."""
        if not self.documents or not query.strip():
            return []

        if SKLEARN_AVAILABLE and self.vectorizer is not None and self.doc_vectors is not None:
            try:
                query_vec = self.vectorizer.transform([query])
                scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
                
                # Pair with documents
                results = []
                for idx, score in enumerate(scores):
                    if score >= min_score:
                        doc_copy = dict(self.documents[idx])
                        doc_copy["score"] = round(float(score), 4)
                        results.append(doc_copy)

                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
            except Exception:
                return self._fallback_search(query, top_k)
        else:
            return self._fallback_search(query, top_k)

    def retrieve_context(
        self,
        detected_labels: Optional[List[str]] = None,
        user_query: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves context considering both detected labels and user question.
        Ensures the AI has facts about the exact objects detected in the image.
        """
        query_parts = []
        if detected_labels:
            query_parts.append(" ".join(detected_labels))
        if user_query:
            query_parts.append(user_query)

        combined_query = " ".join(query_parts).strip()
        if not combined_query:
            return []

        return self.search(combined_query, top_k=top_k)

    def format_context_for_prompt(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved documents into a structured prompt block with citations."""
        if not retrieved_docs:
            return "No domain knowledge context retrieved."

        blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.get("source", "Local Knowledge Base")
            category = doc.get("category", "General")
            title = doc.get("title", "Item")
            content = doc.get("content", "")
            score = doc.get("score", 0.0)
            blocks.append(
                f"[{i}] {title} (Category: {category} | Relevance: {score:.2f} | Source: {source})\n"
                f"    {content}"
            )
        return "\n\n".join(blocks)
