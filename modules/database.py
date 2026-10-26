"""
SQLite Database Manager for VisionTalk AI
Handles conversation history, multi-turn session persistence, and metadata storage.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visiontalk.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database with Row factory."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes tables for sessions, messages, and detections."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                image_name TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                image_path TEXT,
                detected_objects TEXT,
                retrieved_context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
        """)
    conn.close()


def create_or_update_session(
    session_id: str,
    title: Optional[str] = None,
    image_name: Optional[str] = None,
    db_path: Optional[str] = None
) -> None:
    """Inserts a new session or updates an existing one."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO sessions (session_id, title, image_name)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                title = COALESCE(excluded.title, sessions.title),
                image_name = COALESCE(excluded.image_name, sessions.image_name)
        """, (session_id, title or "New Conversation", image_name))
    conn.close()


def save_message(
    session_id: str,
    role: str,
    content: str,
    image_path: Optional[str] = None,
    detected_objects: Optional[List[Dict[str, Any]]] = None,
    retrieved_context: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[str] = None
) -> int:
    """Saves a single turn (user or assistant message) to SQLite."""
    init_db(db_path)
    create_or_update_session(session_id, db_path=db_path)

    det_json = json.dumps(detected_objects) if detected_objects is not None else None
    rag_json = json.dumps(retrieved_context) if retrieved_context is not None else None

    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.execute("""
            INSERT INTO messages (session_id, role, content, image_path, detected_objects, retrieved_context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, role, content, image_path, det_json, rag_json))
        msg_id = cursor.lastrowid
    conn.close()
    return msg_id


def get_session_history(
    session_id: str,
    limit: int = 50,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves conversation history for a given session in chronological order."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.execute("""
        SELECT id, session_id, role, content, image_path, detected_objects, retrieved_context, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        LIMIT ?
    """, (session_id, limit))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "session_id": row["session_id"],
            "role": row["role"],
            "content": row["content"],
            "image_path": row["image_path"],
            "detected_objects": json.loads(row["detected_objects"]) if row["detected_objects"] else [],
            "retrieved_context": json.loads(row["retrieved_context"]) if row["retrieved_context"] else [],
            "timestamp": row["timestamp"],
        })
    return history


def list_sessions(db_path: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Lists recent conversation sessions."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.execute("""
        SELECT s.session_id, s.created_at, s.title, s.image_name,
               COUNT(m.id) as message_count,
               MAX(m.timestamp) as last_activity
        FROM sessions s
        LEFT JOIN messages m ON s.session_id = m.session_id
        GROUP BY s.session_id
        ORDER BY COALESCE(MAX(m.timestamp), s.created_at) DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def clear_session_history(session_id: str, db_path: Optional[str] = None) -> None:
    """Deletes all messages and session entry for the given session_id."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.close()
