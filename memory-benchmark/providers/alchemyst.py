"""
Alchemyst AI Provider Implementation
Implements the MemoryProvider interface for Alchemyst AI
"""

from typing import Dict, Any, Optional, List
import os
import time
import uuid
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
ENV_PATH = find_dotenv(usecwd=True)
if ENV_PATH:
    load_dotenv(ENV_PATH)


class AlchemystProvider:
    """
    Alchemyst AI memory provider implementation
    Docs: https://getalchemystai.com/docs
    
    Alchemyst AI is the ONLY AI context engine that you can verify.
    It provides persistent memory, business data, and operational context
    for AI agents with Pareto Frontier performance on benchmarks.
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Alchemyst AI provider

        Args:
            api_key: Alchemyst AI API key (or from ALCHEMYST_AI_API_KEY env var)
            config: Additional configuration for Alchemyst
        """
        self.api_key = api_key or os.getenv("ALCHEMYST_AI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Alchemyst AI API key is required. Set ALCHEMYST_AI_API_KEY env var or pass api_key"
            )

        self.config = config or {}

        try:
            from alchemyst_ai import AlchemystAI
        except ImportError as e:
            raise ImportError(
                "alchemyst_ai is not installed. Install with: pip install alchemystai"
            ) from e

        # Initialize Alchemyst AI client
        self.client = AlchemystAI(api_key=self.api_key)

        # Configuration
        self.default_scope = self.config.get("scope", "internal")  # internal | external
        self.default_context_type = self.config.get("context_type", "conversation")  # resource | conversation | instructions
        self.similarity_threshold = float(self.config.get("similarity_threshold", 0.8))
        self.min_similarity_threshold = float(self.config.get("minimum_similarity_threshold", 0.5))
        self.source_prefix = self.config.get("source", "memory-benchmark")
        
        # Track stored memories for evaluation purposes
        # Maps benchmark memory_id -> alchemyst context id
        self.memory_mapping = {}
        # Maps benchmark memory_id -> original content
        self.memory_content = {}
        # Track user/org namespaces for isolation
        self.user_namespace = {}
        self.org_namespace = {}

    def _effective_user_id(self, user_id: str) -> str:
        """Return isolated user namespace to avoid cross-scenario bleed-through."""
        return self.user_namespace.get(user_id, user_id)
    
    def _effective_org_id(self, user_id: str) -> str:
        """Return organization ID for the user."""
        return self.org_namespace.get(user_id, f"org_{user_id}")

    def get_name(self) -> str:
        """Get provider name"""
        return "alchemyst"

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return which features this provider supports
        
        Alchemyst AI is the Pareto Frontier for AI Context - best-in-class performance
        """
        return {
            "store": True,
            "retrieve": True,
            "update": True,  # Via re-adding with same source
            "delete": True,
            "semantic_search": True,
            "temporal_awareness": True,
            "metadata_filtering": True,
            "ttl_expiration": False,
            "summarization": True,
            "conflict_detection": True,  # Advanced context management
            "multi_context_types": True,  # resource, conversation, instructions
            "scope_management": True,  # internal vs external context
            "organization_level": True,  # Org-level context sharing
        }

    def store(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        user_id: str = "test_user",
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store a memory in Alchemyst AI

        Args:
            content: The memory content/text
            metadata: Optional metadata dict
            user_id: User ID for the memory
            memory_id: Optional custom memory ID (for tracking)

        Returns:
            Dict with memory_id and status
        """
        try:
            effective_user = self._effective_user_id(user_id)
            effective_org = self._effective_org_id(user_id)
            local_id = memory_id or f"mem_{uuid.uuid4().hex[:12]}"

            # Prepare metadata
            doc_metadata = dict(metadata or {})
            doc_metadata["benchmark_memory_id"] = local_id
            doc_metadata["timestamp"] = doc_metadata.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Determine context type from metadata or use default
            context_type = metadata.get("context_type", self.default_context_type) if metadata else self.default_context_type
            category = metadata.get("category", "general") if metadata else "general"
            
            # Create source identifier
            source = f"{self.source_prefix}.{category}.{local_id}"

            # Add context to Alchemyst
            result = self.client.v1.context.add(
                context_type=context_type,
                documents=[{"content": content}],
                metadata=doc_metadata,
                scope=self.default_scope,
                source=source,
                user_id=effective_user,
                organization_id=effective_org,
            )

            # Store mapping
            if memory_id:
                self.memory_mapping[memory_id] = {
                    "source": source,
                    "user_id": effective_user,
                    "org_id": effective_org,
                    "context_type": context_type
                }
                self.memory_content[memory_id] = content

            return {
                "success": True,
                "memory_id": memory_id or local_id,
                "alchemyst_source": source,
                "result": result.model_dump() if hasattr(result, "model_dump") else str(result),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def retrieve(
        self,
        query: str,
        user_id: str = "test_user",
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve memories from Alchemyst AI

        Args:
            query: Search query
            user_id: User ID to search for
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Dict with retrieved memories
        """
        try:
            effective_user = self._effective_user_id(user_id)
            
            # Prepare search parameters
            search_params = {
                "query": query,
                "user_id": effective_user,
                "similarity_threshold": self.similarity_threshold,
                "minimum_similarity_threshold": self.min_similarity_threshold,
                "scope": self.default_scope,
            }
            
            # Add metadata filters if provided
            if filters:
                # Alchemyst uses metadata filtering in search
                search_params["metadata"] = filters

            # Search context
            results = self.client.v1.context.search(**search_params)

            # Format results
            memories = []
            seen_ids = set()

            # Extract documents from response
            documents = []
            if hasattr(results, "documents"):
                documents = results.documents
            elif isinstance(results, dict) and "documents" in results:
                documents = results["documents"]
            elif hasattr(results, "data") and hasattr(results.data, "documents"):
                documents = results.data.documents

            for doc in documents[:k]:  # Limit to k results
                # Extract data from document
                if hasattr(doc, "content"):
                    doc_content = doc.content
                    doc_metadata = getattr(doc, "metadata", {}) or {}
                    doc_source = getattr(doc, "source", None)
                    doc_score = getattr(doc, "similarity_score", 0.0) or getattr(doc, "score", 0.0)
                    doc_id = getattr(doc, "id", None)
                    created_at = getattr(doc, "created_at", None)
                else:
                    doc_content = doc.get("content", "")
                    doc_metadata = doc.get("metadata", {})
                    doc_source = doc.get("source")
                    doc_score = doc.get("similarity_score", 0.0) or doc.get("score", 0.0)
                    doc_id = doc.get("id")
                    created_at = doc.get("created_at")

                # Get benchmark memory ID from metadata
                benchmark_id = doc_metadata.get("benchmark_memory_id")
                
                # Fallback: find by source in mapping
                if not benchmark_id:
                    for mem_id, mapping in self.memory_mapping.items():
                        if mapping.get("source") == doc_source:
                            benchmark_id = mem_id
                            break
                
                # Use source or doc_id as fallback
                if not benchmark_id:
                    benchmark_id = doc_source or doc_id

                # Avoid duplicates
                if benchmark_id in seen_ids:
                    continue
                seen_ids.add(benchmark_id)

                memories.append(
                    {
                        "id": benchmark_id,
                        "content": doc_content,
                        "score": float(doc_score) if doc_score else 0.0,
                        "metadata": doc_metadata,
                        "created_at": created_at,
                        "updated_at": None,
                        "alchemyst_source": doc_source,
                        "alchemyst_id": doc_id,
                    }
                )

            return {"success": True, "memories": memories, "count": len(memories)}
        except Exception as e:
            return {"success": False, "error": str(e), "memories": []}

    def update(
        self,
        memory_id: str,
        new_content: str,
        user_id: str = "test_user",
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing memory
        
        Alchemyst doesn't have a direct update API, so we:
        1. Delete the old context by source
        2. Add new context with the same source

        Args:
            memory_id: The memory ID to update
            new_content: New content for the memory
            user_id: User ID
            metadata: Optional new metadata

        Returns:
            Dict with update status
        """
        try:
            # Get the mapping info
            mapping = self.memory_mapping.get(memory_id, {})
            source = mapping.get("source")
            
            if not source:
                # Fallback: construct source from memory_id
                category = metadata.get("category", "general") if metadata else "general"
                source = f"{self.source_prefix}.{category}.{memory_id}"

            effective_user = self._effective_user_id(user_id)
            effective_org = self._effective_org_id(user_id)
            
            # Delete old content
            try:
                self.client.v1.context.delete(
                    source=source,
                    user_id=effective_user,
                    by_doc=True,
                )
            except Exception:
                # If delete fails, just continue with add
                pass

            # Add new content with same source (upsert pattern)
            doc_metadata = dict(metadata or {})
            doc_metadata["benchmark_memory_id"] = memory_id
            doc_metadata["timestamp"] = doc_metadata.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ")
            doc_metadata["updated"] = True
            
            context_type = metadata.get("context_type", self.default_context_type) if metadata else self.default_context_type

            result = self.client.v1.context.add(
                context_type=context_type,
                documents=[{"content": new_content}],
                metadata=doc_metadata,
                scope=self.default_scope,
                source=source,
                user_id=effective_user,
                organization_id=effective_org,
            )

            # Update tracking
            self.memory_content[memory_id] = new_content
            self.memory_mapping[memory_id] = {
                "source": source,
                "user_id": effective_user,
                "org_id": effective_org,
                "context_type": context_type
            }

            return {
                "success": True,
                "memory_id": memory_id,
                "result": result.model_dump() if hasattr(result, "model_dump") else str(result),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
            # Get the mapping info
            mapping = self.memory_mapping.get(memory_id, {})
            source = mapping.get("source")
            
            if not source:
                # Fallback: try to construct source
                source = f"{self.source_prefix}.*.{memory_id}"

            effective_user = self._effective_user_id(user_id)

            # Delete by source
            self.client.v1.context.delete(
                source=source,
                user_id=effective_user,
                by_doc=True,
            )

            # Remove from tracking
            if memory_id in self.memory_mapping:
                del self.memory_mapping[memory_id]
            if memory_id in self.memory_content:
                del self.memory_content[memory_id]

            return {"success": True, "memory_id": memory_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_all(self, user_id: str = "test_user") -> Dict[str, Any]:
        """
        Delete all memories for a user

        Args:
            user_id: User ID

        Returns:
            Dict with deletion status
        """
        try:
            effective_user = self._effective_user_id(user_id)

            # Delete all context for this user
            # Alchemyst allows deletion by user_id
            self.client.v1.context.delete(
                user_id=effective_user,
                by_doc=True,
            )

            # Clear tracking
            self.memory_mapping.clear()
            self.memory_content.clear()

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_all(self, user_id: str = "test_user") -> Dict[str, Any]:
        """
        List all memories for a user

        Args:
            user_id: User ID

        Returns:
            Dict with all memories
        """
        try:
            effective_user = self._effective_user_id(user_id)

            # View all context for user
            results = self.client.v1.context.view.docs(
                user_id=effective_user,
            )

            memories = []
            
            # Extract documents
            documents = []
            if hasattr(results, "documents"):
                documents = results.documents
            elif isinstance(results, dict) and "documents" in results:
                documents = results["documents"]

            for doc in documents:
                # Extract data
                if hasattr(doc, "content"):
                    doc_content = doc.content
                    doc_metadata = getattr(doc, "metadata", {}) or {}
                    doc_source = getattr(doc, "source", None)
                    doc_id = getattr(doc, "id", None)
                    created_at = getattr(doc, "created_at", None)
                else:
                    doc_content = doc.get("content", "")
                    doc_metadata = doc.get("metadata", {})
                    doc_source = doc.get("source")
                    doc_id = doc.get("id")
                    created_at = doc.get("created_at")

                # Get benchmark memory ID
                benchmark_id = doc_metadata.get("benchmark_memory_id") or doc_source or doc_id

                memories.append(
                    {
                        "id": benchmark_id,
                        "content": doc_content,
                        "metadata": doc_metadata,
                        "created_at": created_at,
                        "updated_at": None,
                        "alchemyst_source": doc_source,
                        "alchemyst_id": doc_id,
                    }
                )

            return {"success": True, "memories": memories, "count": len(memories)}
        except Exception as e:
            return {"success": False, "error": str(e), "memories": []}

    def reset(self, user_id: str = "test_user") -> Dict[str, Any]:
        """
        Reset/clear all memories (for testing)

        Args:
            user_id: User ID

        Returns:
            Dict with reset status
        """
        # Delete all memories for the current user namespace
        previous_effective = self._effective_user_id(user_id)
        try:
            self.client.v1.context.delete(
                user_id=previous_effective,
                by_doc=True,
            )
        except Exception:
            pass

        # Create new isolated user namespace
        isolated_user_id = f"{user_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        isolated_org_id = f"org_{isolated_user_id}"
        
        self.user_namespace[user_id] = isolated_user_id
        self.org_namespace[user_id] = isolated_org_id

        # Clear tracking
        self.memory_mapping.clear()
        self.memory_content.clear()

        return {
            "success": True,
            "user_id": isolated_user_id,
            "organization_id": isolated_org_id
        }