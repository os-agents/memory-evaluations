"""
Supermemory Provider Implementation
Implements the MemoryProvider interface for Supermemory
"""

from typing import Dict, Any, Optional
import os
import time
import uuid
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
ENV_PATH = find_dotenv(usecwd=True)
if ENV_PATH:
    load_dotenv(ENV_PATH)


class SupermemoryProvider:
    """
    Supermemory memory provider implementation
    Docs: https://supermemory.ai/docs/intro
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Supermemory provider

        Args:
            api_key: Supermemory API key (or from SUPERMEMORY_API_KEY env var)
            config: Additional configuration for Supermemory
        """
        self.api_key = api_key or os.getenv("SUPERMEMORY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Supermemory API key is required. Set SUPERMEMORY_API_KEY env var or pass api_key"
            )

        self.config = config or {}
        self.timing_config = {
            "post_reset_sleep_s": float(self.config.get("post_reset_sleep_s", 60.0)),
            "post_setup_sleep_s": float(self.config.get("post_setup_sleep_s", 60.0)),
        }

        try:
            from supermemory import Supermemory
        except ImportError as e:
            raise ImportError(
                "supermemory is not installed. Install with: pip install supermemory"
            ) from e

        # Initialize Supermemory client
        self.client = Supermemory(api_key=self.api_key)

        # Track stored memories for evaluation purposes
        # Maps benchmark memory_id -> supermemory document id
        self.memory_mapping = {}
        # Maps benchmark memory_id -> original content
        self.memory_content = {}
        # Track user namespaces for isolation
        self.user_namespace = {}

    def get_timing_config(self) -> Dict[str, float]:
        """Provider-specific timing for evaluator sleeps."""
        return self.timing_config

    def _effective_user_id(self, user_id: str) -> str:
        """Return isolated user namespace to avoid cross-scenario bleed-through."""
        return self.user_namespace.get(user_id, user_id)

    def _container_tags(self, user_id: str) -> list:
        """Generate container tags for a user."""
        effective_user = self._effective_user_id(user_id)
        return [effective_user]

    @staticmethod
    def _obj_get(obj: Any, *keys: str, default: Any = None) -> Any:
        """Safely read fields from SDK objects or dicts using multiple key aliases."""
        for key in keys:
            if isinstance(obj, dict):
                if key in obj:
                    return obj.get(key)
            else:
                if hasattr(obj, key):
                    return getattr(obj, key)
        return default

    def _delete_document_with_retry(self, document_id: str, attempts: int = 30, delay_s: float = 2.0) -> None:
        """
        Delete a document with retry for transient processing states.
        Supermemory may return 409 while docs are still indexing.
        """
        last_err = None
        for _ in range(max(1, attempts)):
            try:
                self.client.documents.delete(id=document_id)
                return
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "still processing" in msg or "409" in msg:
                    time.sleep(delay_s)
                    continue
                raise
        if last_err:
            raise last_err

    def _list_documents_for_tags(self, container_tags: list, page_size: int = 200) -> list:
        """List all documents for container tags with pagination support."""
        all_docs = []
        page = 1
        while True:
            resp = self.client.documents.list(
                container_tags=container_tags,
                limit=page_size,
                page=page,
            )

            docs = []
            if hasattr(resp, "memories"):
                docs = resp.memories or []
            elif isinstance(resp, dict) and "memories" in resp:
                docs = resp["memories"] or []
            elif hasattr(resp, "documents"):
                docs = resp.documents or []
            elif isinstance(resp, dict) and "documents" in resp:
                docs = resp["documents"] or []
            elif isinstance(resp, list):
                docs = resp

            all_docs.extend(docs)

            pagination = self._obj_get(resp, "pagination")
            if not pagination:
                break

            current_page = self._obj_get(pagination, "current_page", "currentPage", default=page)
            total_pages = self._obj_get(pagination, "total_pages", "totalPages", default=page)
            if float(current_page) >= float(total_pages):
                break
            page += 1
        return all_docs

    def get_name(self) -> str:
        """Get provider name"""
        return "supermemory"

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
            "ttl_expiration": False,
            "summarization": True,
            "conflict_detection": False,
        }

    def store(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        user_id: str = "test_user",
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store a memory in Supermemory

        Args:
            content: The memory content/text
            metadata: Optional metadata dict
            user_id: User ID for the memory
            memory_id: Optional custom memory ID (for tracking)

        Returns:
            Dict with memory_id and status
        """
        try:
            container_tags = self._container_tags(user_id)
            custom_id = memory_id or f"mem_{uuid.uuid4().hex[:12]}"

            # Prepare metadata
            doc_metadata = dict(metadata or {})
            doc_metadata["benchmark_memory_id"] = custom_id

            # Add document to Supermemory
            result = self.client.add(
                content=content,
                container_tags=container_tags,
                custom_id=custom_id,
                metadata=doc_metadata,
            )

            # Extract document ID from result
            doc_id = self._obj_get(result, "id")

            # Store mapping
            if memory_id:
                self.memory_mapping[memory_id] = doc_id or custom_id
                self.memory_content[memory_id] = content

            return {
                "success": True,
                "memory_id": memory_id or custom_id,
                "supermemory_id": doc_id,
                "result": result if isinstance(result, dict) else (result.__dict__ if hasattr(result, '__dict__') else str(result)),
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
        Retrieve memories from Supermemory

        Args:
            query: Search query
            user_id: User ID to search for
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Dict with retrieved memories
        """
        try:
            container_tags = self._container_tags(user_id)

            # Prepare search filters
            search_filters = None
            if filters:
                # Convert filters to Supermemory format
                # Supermemory uses AND/OR filter format
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append({"key": key, "value": value})

                if filter_conditions:
                    search_filters = {"AND": filter_conditions}

            # Search documents - only include filters if not None
            search_params = {
                "q": query,
                "container_tags": container_tags,
                "limit": k,
            }
            if search_filters is not None:
                search_params["filters"] = search_filters
            
            # include_full_docs ensures `content` is present on Result objects
            search_params["include_full_docs"] = True
            results = self.client.search.documents(**search_params)

            # Format results
            memories = []
            seen_ids = set()

            # Extract results from response
            result_list = []
            if hasattr(results, "results"):
                result_list = results.results
            elif isinstance(results, dict) and "results" in results:
                result_list = results["results"]
            elif isinstance(results, list):
                result_list = results

            for result in result_list:
                # Supermemory search results use document_id instead of id.
                doc_id = self._obj_get(result, "document_id", "documentId", "id")
                content = self._obj_get(result, "content", "text", default="") or ""
                result_metadata = self._obj_get(result, "metadata", default={}) or {}
                score = self._obj_get(result, "score", default=0.0) or 0.0
                created_at = self._obj_get(result, "created_at", "createdAt")
                updated_at = self._obj_get(result, "updated_at", "updatedAt")
                custom_id = self._obj_get(result, "custom_id", "customId")

                # If full content is absent, use top-matching chunk content.
                if not content:
                    chunks = self._obj_get(result, "chunks", default=[]) or []
                    if chunks:
                        first_chunk = chunks[0]
                        content = self._obj_get(first_chunk, "content", default="") or ""

                # Get benchmark memory ID
                benchmark_id = result_metadata.get("benchmark_memory_id")
                if not benchmark_id and custom_id:
                    # Try to use custom_id as benchmark_id
                    benchmark_id = custom_id

                # Find reverse mapping if needed
                if not benchmark_id:
                    for mem_id, stored_id in self.memory_mapping.items():
                        if stored_id == doc_id:
                            benchmark_id = mem_id
                            break

                # Use doc_id as fallback
                if not benchmark_id:
                    benchmark_id = doc_id

                # Avoid duplicates
                if benchmark_id in seen_ids:
                    continue
                seen_ids.add(benchmark_id)

                memories.append(
                    {
                        "id": benchmark_id,
                        "content": content,
                        "score": score or 0.0,
                        "metadata": result_metadata,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "supermemory_id": doc_id,
                    }
                )

                if len(memories) >= k:
                    break

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

        Args:
            memory_id: The memory ID to update
            new_content: New content for the memory
            user_id: User ID
            metadata: Optional new metadata

        Returns:
            Dict with update status
        """
        try:
            container_tags = self._container_tags(user_id)

            # Prepare metadata
            doc_metadata = dict(metadata or {})
            doc_metadata["benchmark_memory_id"] = memory_id

            # Use customId for upsert - this is the recommended approach
            # Same customId will update existing memory
            result = self.client.add(
                content=new_content,
                container_tags=container_tags,
                custom_id=memory_id,
                metadata=doc_metadata,
            )

            # Update tracking
            self.memory_content[memory_id] = new_content

            # Extract document ID from result
            doc_id = None
            if hasattr(result, "id"):
                doc_id = result.id
            elif isinstance(result, dict):
                doc_id = result.get("id")

            if doc_id:
                self.memory_mapping[memory_id] = doc_id

            return {
                "success": True,
                "memory_id": memory_id,
                "result": result if isinstance(result, dict) else (result.__dict__ if hasattr(result, '__dict__') else str(result)),
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
            # Get the actual supermemory document ID
            doc_id = self.memory_mapping.get(memory_id)

            if doc_id:
                # Delete by document ID
                self._delete_document_with_retry(doc_id)
            else:
                # Try deleting by custom_id
                # Note: Supermemory API supports deletion by customId through the endpoint
                # DELETE /v3/documents/{id} where id can be customId
                self._delete_document_with_retry(memory_id)

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
            container_tags = self._container_tags(user_id)

            # List all documents for this user
            doc_list = self._list_documents_for_tags(container_tags=container_tags)

            # Delete each document
            for doc in doc_list:
                doc_id = self._obj_get(doc, "id")

                if doc_id:
                    try:
                        self._delete_document_with_retry(doc_id)
                    except Exception:
                        pass

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
            container_tags = self._container_tags(user_id)

            # List documents
            results = self.client.documents.list(
                container_tags=container_tags, limit=1000
            )

            # Extract document list
            doc_list = []
            if hasattr(results, "memories"):
                doc_list = results.memories
            elif isinstance(results, dict) and "memories" in results:
                doc_list = results["memories"]
            elif hasattr(results, "documents"):
                doc_list = results.documents
            elif isinstance(results, dict) and "documents" in results:
                doc_list = results["documents"]
            elif isinstance(results, list):
                doc_list = results

            memories = []
            for doc in doc_list:
                doc_id = self._obj_get(doc, "id")
                content = self._obj_get(doc, "content", "text", default="") or ""
                doc_metadata = self._obj_get(doc, "metadata", default={}) or {}
                created_at = self._obj_get(doc, "created_at", "createdAt")
                updated_at = self._obj_get(doc, "updated_at", "updatedAt")
                custom_id = self._obj_get(doc, "custom_id", "customId")

                # Get benchmark memory ID
                benchmark_id = doc_metadata.get("benchmark_memory_id")
                if not benchmark_id and custom_id:
                    benchmark_id = custom_id
                if not benchmark_id:
                    benchmark_id = doc_id

                memories.append(
                    {
                        "id": benchmark_id,
                        "content": content,
                        "metadata": doc_metadata,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "supermemory_id": doc_id,
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
            container_tags = [previous_effective]
            doc_list = self._list_documents_for_tags(container_tags=container_tags)

            for doc in doc_list:
                doc_id = self._obj_get(doc, "id")
                if doc_id:
                    try:
                        self._delete_document_with_retry(doc_id)
                    except Exception:
                        pass
        except Exception:
            pass

        # Create new isolated user namespace
        isolated_user_id = f"{user_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.user_namespace[user_id] = isolated_user_id

        # Clear tracking
        self.memory_mapping.clear()
        self.memory_content.clear()

        return {"success": True, "user_id": isolated_user_id}
