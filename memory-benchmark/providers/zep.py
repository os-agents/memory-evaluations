"""
Zep Provider Implementation
Implements the memory provider interface for Zep Cloud.
"""

from typing import Dict, Any, Optional
import os
import time
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
ENV_PATH = find_dotenv(usecwd=True)
if ENV_PATH:
    load_dotenv(ENV_PATH)


class ZepProvider:
    """
    Zep memory provider implementation.
    Docs: https://help.getzep.com/overview
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        self.api_key = api_key or os.getenv("ZEP_API_KEY")
        if not self.api_key:
            raise ValueError("ZEP_API_KEY is required. Set it in .env or pass api_key.")

        self.config = config or {}
        self.thread_prefix = self.config.get("thread_prefix", "memory-benchmark")
        self.search_scope = self.config.get("search_scope", "episodes")
        self.search_limit_multiplier = int(self.config.get("search_limit_multiplier", 8))
        self.search_min_score = float(self.config.get("search_min_score", 0.0))
        self.search_retries = int(self.config.get("search_retries", 3))
        self.search_retry_delay_ms = int(self.config.get("search_retry_delay_ms", 350))
        self.task_wait_timeout_s = float(self.config.get("task_wait_timeout_s", 20.0))
        self.task_poll_interval_s = float(self.config.get("task_poll_interval_s", 0.25))
        self.strict_uuid_match = bool(self.config.get("strict_uuid_match", False))
        self.use_metadata_timestamp_as_created_at = bool(
            self.config.get("use_metadata_timestamp_as_created_at", True)
        )

        try:
            from zep_cloud import Zep
            from zep_cloud.types import Message
        except ImportError as e:
            raise ImportError(
                "zep-cloud is not installed. Install with: pip install zep-cloud"
            ) from e

        base_url = self.config.get("base_url") or os.getenv("ZEP_BASE_URL")
        if base_url:
            self.client = Zep(api_key=self.api_key, base_url=base_url)
        else:
            self.client = Zep(api_key=self.api_key)

        self.Message = Message

        # Track benchmark local IDs and state per user.
        self._states_by_user: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def get_name(self) -> str:
        return "zep"

    def get_capabilities(self) -> Dict[str, bool]:
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

    def _thread_id(self, user_id: str) -> str:
        return f"{self.thread_prefix}-{user_id}"

    def _state(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        if user_id not in self._states_by_user:
            self._states_by_user[user_id] = {}
        return self._states_by_user[user_id]

    def _ensure_user_and_thread(self, user_id: str) -> str:
        thread_id = self._thread_id(user_id)

        # Best effort create user/thread. If they already exist, continue.
        try:
            self.client.user.add(user_id=user_id)
        except Exception:
            pass

        try:
            self.client.thread.create(thread_id=thread_id, user_id=user_id)
        except Exception as e:
            msg = str(e).lower()
            if "already" not in msg and "exists" not in msg and "conflict" not in msg:
                # If not an "already exists" style error, validate by attempting to fetch.
                self.client.thread.get(thread_id=thread_id, lastn=1)

        return thread_id

    def _mark_message_metadata(self, message_uuid: Optional[str], metadata: Dict[str, Any]) -> None:
        if not message_uuid:
            return
        try:
            self.client.thread.message.update(message_uuid=message_uuid, metadata=metadata)
        except Exception:
            # Metadata update is best effort for benchmark bookkeeping.
            pass

    def _wait_for_task(self, task_id: Optional[str]) -> None:
        """Wait for async Zep ingestion tasks to complete."""
        if not task_id:
            return

        deadline = time.time() + self.task_wait_timeout_s
        while time.time() < deadline:
            task = self.client.task.get(task_id=task_id)
            status = (getattr(task, "status", "") or "").lower()

            if status in ("completed", "succeeded", "success", "done"):
                return
            if status in ("failed", "error", "cancelled"):
                err = getattr(task, "error", None)
                raise RuntimeError(f"Zep task {task_id} failed: {err}")

            time.sleep(self.task_poll_interval_s)

        raise TimeoutError(f"Timed out waiting for Zep task {task_id} to complete")

    def store(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        user_id: str = "test_user",
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            thread_id = self._ensure_user_and_thread(user_id)
            local_id = memory_id or f"zep_{abs(hash(content))}"

            msg_metadata = dict(metadata or {})
            msg_metadata["benchmark_memory_id"] = local_id
            msg_metadata["deleted"] = False
            msg_metadata["superseded"] = False
            created_at = None
            if self.use_metadata_timestamp_as_created_at:
                created_at = (metadata or {}).get("timestamp")

            message = self.Message(
                role="user",
                content=content,
                metadata=msg_metadata,
                created_at=created_at,
            )
            response = self.client.thread.add_messages(
                thread_id=thread_id,
                messages=[message],
                return_context=False,
            )
            self._wait_for_task(getattr(response, "task_id", None))

            message_uuid = None
            if getattr(response, "message_uuids", None):
                message_uuid = response.message_uuids[0]

            self._state(user_id)[local_id] = {
                "content": content,
                "metadata": dict(metadata or {}),
                "deleted": False,
                "message_uuid": message_uuid,
            }

            return {
                "success": True,
                "memory_id": local_id,
                "zep_message_uuid": message_uuid,
                "result": response.model_dump() if hasattr(response, "model_dump") else {},
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
        try:
            self._ensure_user_and_thread(user_id)
            state = self._state(user_id)

            search_limit = max(k * max(self.search_limit_multiplier, 1), 20)
            memories = []

            for attempt in range(max(self.search_retries, 1)):
                results = self.client.graph.search(
                    query=query,
                    user_id=user_id,
                    scope=self.search_scope,
                    limit=search_limit,
                )

                episodes = getattr(results, "episodes", None) or []
                memories = []
                seen = set()

                for episode in episodes:
                    ep_meta = dict(getattr(episode, "metadata", {}) or {})
                    ep_content = getattr(episode, "content", "")
                    ep_id = getattr(episode, "uuid_", None)
                    ep_score = getattr(episode, "score", 0.0) or 0.0
                    if ep_score < self.search_min_score:
                        continue

                    local_id = ep_meta.get("benchmark_memory_id")
                    if not local_id:
                        # Fallback: exact content match to active local state.
                        matches = [
                            lid
                            for lid, entry in state.items()
                            if not entry.get("deleted") and entry.get("content") == ep_content
                        ]
                        if len(matches) == 1:
                            local_id = matches[0]

                    if local_id and local_id in state:
                        entry = state[local_id]
                        if entry.get("deleted"):
                            continue
                        current_uuid = entry.get("message_uuid")
                        # Optionally enforce strict latest-id matching.
                        if self.strict_uuid_match and current_uuid and ep_id and current_uuid != ep_id:
                            continue
                    else:
                        # Ignore untracked episodes so benchmark IDs remain stable.
                        continue

                    if local_id in seen:
                        continue
                    seen.add(local_id)

                    memories.append(
                        {
                            "id": local_id,
                            "content": ep_content,
                            "score": ep_score,
                            "metadata": ep_meta,
                            "created_at": getattr(episode, "created_at", None),
                            "updated_at": None,
                            "zep_episode_uuid": ep_id,
                        }
                    )

                    if len(memories) >= k:
                        break

                if memories or attempt == max(self.search_retries, 1) - 1:
                    break
                time.sleep(max(self.search_retry_delay_ms, 0) / 1000.0)

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
        try:
            thread_id = self._ensure_user_and_thread(user_id)
            state = self._state(user_id)
            previous = state.get(memory_id, {})
            prev_uuid = previous.get("message_uuid")

            if prev_uuid:
                old_meta = dict(previous.get("metadata", {}))
                old_meta["benchmark_memory_id"] = memory_id
                old_meta["deleted"] = False
                old_meta["superseded"] = True
                self._mark_message_metadata(prev_uuid, old_meta)

            new_meta = dict(metadata or {})
            new_meta["benchmark_memory_id"] = memory_id
            new_meta["deleted"] = False
            new_meta["superseded"] = False
            created_at = None
            if self.use_metadata_timestamp_as_created_at:
                created_at = (metadata or {}).get("timestamp")

            message = self.Message(
                role="user",
                content=new_content,
                metadata=new_meta,
                created_at=created_at,
            )
            response = self.client.thread.add_messages(
                thread_id=thread_id,
                messages=[message],
                return_context=False,
            )
            self._wait_for_task(getattr(response, "task_id", None))

            new_uuid = None
            if getattr(response, "message_uuids", None):
                new_uuid = response.message_uuids[0]

            state[memory_id] = {
                "content": new_content,
                "metadata": dict(metadata or {}),
                "deleted": False,
                "message_uuid": new_uuid,
            }

            return {
                "success": True,
                "memory_id": memory_id,
                "result": response.model_dump() if hasattr(response, "model_dump") else {},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete(self, memory_id: str, user_id: str = "test_user") -> Dict[str, Any]:
        try:
            state = self._state(user_id)
            if memory_id in state:
                entry = state[memory_id]
                msg_uuid = entry.get("message_uuid")
                delete_meta = dict(entry.get("metadata", {}))
                delete_meta["benchmark_memory_id"] = memory_id
                delete_meta["deleted"] = True
                delete_meta["superseded"] = False
                self._mark_message_metadata(msg_uuid, delete_meta)
                entry["deleted"] = True

            return {"success": True, "memory_id": memory_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_all(self, user_id: str = "test_user") -> Dict[str, Any]:
        try:
            thread_id = self._thread_id(user_id)
            try:
                self.client.thread.delete(thread_id=thread_id)
            except Exception:
                pass

            if user_id in self._states_by_user:
                self._states_by_user[user_id] = {}

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_all(self, user_id: str = "test_user") -> Dict[str, Any]:
        try:
            state = self._state(user_id)
            memories = []
            for local_id, entry in state.items():
                if entry.get("deleted"):
                    continue
                memories.append(
                    {
                        "id": local_id,
                        "content": entry.get("content"),
                        "metadata": entry.get("metadata", {}),
                        "created_at": None,
                        "updated_at": None,
                        "zep_message_uuid": entry.get("message_uuid"),
                    }
                )
            return {"success": True, "memories": memories, "count": len(memories)}
        except Exception as e:
            return {"success": False, "error": str(e), "memories": []}

    def reset(self, user_id: str = "test_user") -> Dict[str, Any]:
        return self.delete_all(user_id=user_id)
