"""
Mem0 Provider Implementation
Implements the MemoryProvider interface for Mem0
"""

from typing import Dict, Any, Optional
import os
import re
import time
import uuid
from difflib import SequenceMatcher
from dotenv import load_dotenv, find_dotenv
from mem0 import MemoryClient

# Load environment variables from .env file
ENV_PATH = find_dotenv(usecwd=True)
if ENV_PATH:
    load_dotenv(ENV_PATH)


class Mem0Provider:
    """
    Mem0 memory provider implementation
    Docs: https://docs.mem0.ai/
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Mem0 provider
        
        Args:
            api_key: Mem0 API key (or from MEM0_API_KEY env var)
            config: Additional configuration for Mem0
        """
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError("Mem0 API key is required. Set MEM0_API_KEY env var or pass api_key")
        
        self.config = config or {}

        # Initialize Mem0 hosted API client (uses MEM0_API_KEY)
        self.client = MemoryClient(
            api_key=self.api_key,
            host=self.config.get("host") or os.getenv("MEM0_HOST"),
            org_id=self.config.get("org_id") or os.getenv("MEM0_ORG_ID"),
            project_id=self.config.get("project_id") or os.getenv("MEM0_PROJECT_ID"),
        )
        
        # Track stored memories for evaluation purposes
        self.memory_mapping = {}  # scenario_id -> list of mem0 memory ids
        self.memory_content = {}  # scenario_id -> source content
        self.user_namespace = {}  # logical user_id -> isolated user_id

    @staticmethod
    def _extract_results(payload: Any) -> list:
        """Normalize Mem0 responses across SDK variants."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                return payload["results"]
            if isinstance(payload.get("memories"), list):
                return payload["memories"]
            if payload.get("id"):
                return [payload]
        return []

    @staticmethod
    def _extract_memory_id(payload: Any) -> Optional[str]:
        """Extract a concrete memory id from add/get style responses."""
        if isinstance(payload, dict):
            if payload.get("id"):
                return payload["id"]
            results = payload.get("results")
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    return first.get("id")
                if isinstance(first, str):
                    return first
            memory = payload.get("memory")
            if isinstance(memory, dict):
                return memory.get("id")
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return first.get("id")
            if isinstance(first, str):
                return first
        return None

    def _reverse_mapping(self) -> Dict[str, str]:
        """Build mem0_id -> local_id lookup for benchmark scoring."""
        reverse = {}
        for local_id, mem0_id in self.memory_mapping.items():
            if isinstance(mem0_id, str) and mem0_id:
                reverse[mem0_id] = local_id
        return reverse

    def _effective_user_id(self, user_id: str) -> str:
        """Return isolated user namespace to avoid cross-scenario bleed-through."""
        return self.user_namespace.get(user_id, user_id)

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _match_local_id_by_content(self, content: str) -> Optional[str]:
        """
        Map provider-returned paraphrased content back to benchmark memory IDs.
        This stabilizes scoring when Mem0 returns new memory UUIDs.
        """
        target = self._normalize_text(content)
        if not target:
            return None

        best_id = None
        best_score = 0.0
        target_tokens = set(target.split())

        for local_id, original in self.memory_content.items():
            source = self._normalize_text(original)
            if not source:
                continue
            source_tokens = set(source.split())
            overlap = 0.0
            if source_tokens and target_tokens:
                overlap = len(source_tokens & target_tokens) / len(source_tokens | target_tokens)
            seq_ratio = SequenceMatcher(None, target, source).ratio()
            score = max(overlap, seq_ratio)
            if score > best_score:
                best_score = score
                best_id = local_id

        # Conservative threshold to avoid incorrect remaps.
        if best_score >= 0.42:
            return best_id
        return None
        
    def get_name(self) -> str:
        """Get provider name"""
        return "mem0"
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return which features this provider supports
        """
        return {
            "store": True,
            "retrieve": True,
            "update": True,
            "delete": True,
            "semantic_search": True,
            "temporal_awareness": True,
            "metadata_filtering": True,
            "ttl_expiration": False,  # Mem0 doesn't have built-in TTL
            "summarization": True,  # Mem0 can summarize memories
            "conflict_detection": False,  # Not built-in
        }
    
    def store(self, content: str, metadata: Optional[Dict] = None, 
              user_id: str = "test_user", memory_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store a memory in Mem0
        
        Args:
            content: The memory content/text
            metadata: Optional metadata dict
            user_id: User ID for the memory
            memory_id: Optional custom memory ID (for tracking)
            
        Returns:
            Dict with memory_id and status
        """
        try:
            effective_user_id = self._effective_user_id(user_id)
            # Mem0's add method
            result = self.client.add(
                messages=content,
                user_id=effective_user_id,
                metadata=metadata or {},
                async_mode=False,
            )

            created_id = self._extract_memory_id(result)

            # Store mapping if custom ID provided
            if memory_id:
                self.memory_mapping[memory_id] = created_id or result
                self.memory_content[memory_id] = content

            return {
                "success": True,
                "memory_id": memory_id or created_id,
                "mem0_id": created_id,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def retrieve(self, query: str, user_id: str = "test_user", 
                 k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retrieve memories from Mem0
        
        Args:
            query: Search query
            user_id: User ID to search for
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Dict with retrieved memories
        """
        try:
            effective_user_id = self._effective_user_id(user_id)
            # Mem0's search method
            merged_filters = dict(filters or {})
            # Mem0 v2 search requires non-empty filters.
            if "user_id" not in merged_filters:
                merged_filters["user_id"] = effective_user_id

            results = self.client.search(
                query=query,
                top_k=k,
                filters=merged_filters
            )

            # Format results
            memories = []
            reverse_map = self._reverse_mapping()
            for result in self._extract_results(results):
                mem0_id = result.get("id")
                content = result.get("memory") or result.get("text")
                local_id = reverse_map.get(mem0_id)
                if not local_id:
                    local_id = self._match_local_id_by_content(content)
                    if local_id and mem0_id:
                        self.memory_mapping[local_id] = mem0_id
                score = result.get("score", 0.0) or 0.0
                memories.append({
                    "id": local_id or mem0_id,
                    "content": content,
                    "score": score,
                    "metadata": result.get("metadata", {}),
                    "created_at": result.get("created_at"),
                    "updated_at": result.get("updated_at"),
                    "mem0_id": mem0_id,
                })
                if len(memories) >= k:
                    break

            return {
                "success": True,
                "memories": memories,
                "count": len(memories)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "memories": []
            }

    def update(self, memory_id: str, new_content: str, 
               user_id: str = "test_user", metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Update an existing memory

        Args:
            memory_id: The memory ID to update
            user_id: User ID
            new_content: New content for the memory
            metadata: Optional new metadata

        Returns:
            Dict with update status
        """
        try:
            # Get the actual mem0 ID if we have a mapping
            mem0_id = self.memory_mapping.get(memory_id, memory_id)
            if isinstance(mem0_id, dict):
                mem0_id = self._extract_memory_id(mem0_id) or memory_id

            # Mem0's update method
            result = self.client.update(
                memory_id=mem0_id,
                text=new_content,
                metadata=metadata
            )
            self.memory_content[memory_id] = new_content

            return {
                "success": True,
                "memory_id": memory_id,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def delete(self, memory_id: str, user_id: str = "test_user") -> Dict[str, Any]:
        """
        Delete a memory

        Args:
            memory_id: The memory ID to delete
            user_id: User ID

        Returns:
            Dict with deletion status
        """
        try:
            # Get the actual mem0 ID if we have a mapping
            mem0_id = self.memory_mapping.get(memory_id, memory_id)
            if isinstance(mem0_id, dict):
                mem0_id = self._extract_memory_id(mem0_id) or memory_id

            # Mem0's delete method
            result = self.client.delete(memory_id=mem0_id)

            # Remove from mapping
            if memory_id in self.memory_mapping:
                del self.memory_mapping[memory_id]
            if memory_id in self.memory_content:
                del self.memory_content[memory_id]

            return {
                "success": True,
                "memory_id": memory_id,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def delete_all(self, user_id: str = "test_user") -> Dict[str, Any]:
        """
        Delete all memories for a user

        Args:
            user_id: User ID

        Returns:
            Dict with deletion status
        """
        try:
            effective_user_id = self._effective_user_id(user_id)
            result = self.client.delete_all(user_id=effective_user_id)

            # Clear mapping
            self.memory_mapping = {}
            self.memory_content = {}

            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def list_all(self, user_id: str = "test_user") -> Dict[str, Any]:
        """
        List all memories for a user

        Args:
            user_id: User ID

        Returns:
            Dict with all memories
        """
        try:
            effective_user_id = self._effective_user_id(user_id)
            results = self.client.get_all(user_id=effective_user_id)

            memories = []
            reverse_map = self._reverse_mapping()
            for result in self._extract_results(results):
                mem0_id = result.get("id")
                memories.append({
                    "id": reverse_map.get(mem0_id, mem0_id),
                    "content": result.get("memory") or result.get("text"),
                    "metadata": result.get("metadata", {}),
                    "created_at": result.get("created_at"),
                    "updated_at": result.get("updated_at"),
                    "mem0_id": mem0_id,
                })

            return {
                "success": True,
                "memories": memories,
                "count": len(memories)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "memories": []
            }
    
    def reset(self, user_id: str = "test_user") -> Dict[str, Any]:
        """
        Reset/clear all memories (for testing)
        
        Args:
            user_id: User ID
            
        Returns:
            Dict with reset status
        """
        previous_effective = self._effective_user_id(user_id)
        try:
            self.client.delete_all(user_id=previous_effective)
        except Exception:
            pass

        isolated_user_id = f"{user_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.user_namespace[user_id] = isolated_user_id
        self.memory_mapping = {}
        self.memory_content = {}

        return {"success": True, "user_id": isolated_user_id}
