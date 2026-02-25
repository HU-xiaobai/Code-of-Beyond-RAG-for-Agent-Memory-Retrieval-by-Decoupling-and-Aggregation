"""High-level facade providing a simplified memory interface."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from src.config import MemoryConfig
from src.core.memory_system import MemorySystem
from src.utils import LLMClient, EmbeddingClient



class xMemory:
    """Minimal public API for working with the memory system.

    Designed for "one-line" usage while still allowing custom dependency
    injection for tests and advanced scenarios.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        *,
        memory_system: Optional[MemorySystem] = None,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self._memory_system = memory_system or MemorySystem(
            config=self.config,
            language=self.config.language,
            llm_client=llm_client,
            embedding_client=embedding_client,
        )

    # ------------------------------------------------------------------
    # Basic lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release all resources held by the underlying memory system."""
        self._memory_system.__exit__(None, None, None)

    def __enter__(self) -> "xMemory":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------
    def add_messages(self, user_id: str, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Append messages to the memory buffer for a given user."""
        return self._memory_system.add_messages(user_id, list(messages))

    def flush(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Force creation of an episode from the current buffer."""
        return self._memory_system.force_episode_creation(user_id)

    def wait_for_semantic(self, user_id: str, timeout: float = 30.0) -> bool:
        """Block until all semantic memory tasks for the user complete."""
        return self._memory_system.wait_for_semantic_generation(user_id, timeout=timeout)

    def search(
        self,
        user_id: str,
        query: str,
        *,
        top_k_episodes: Optional[int] = None,
        top_k_semantic: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search episodic + semantic memory in a single call."""
        return self._memory_system.search_all(
            user_id=user_id,
            query=query,
            top_k_episodes=top_k_episodes,
            top_k_semantic=top_k_semantic,
            search_method=search_method,
        )

    def stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Expose system statistics for diagnostics."""
        return self._memory_system.get_stats(user_id)

    ##CAM
    def fetch_incremental_batches(self, user_id: str):
        """
        拿到本轮新产生的 episode / semantic（不清空）。
        返回 (new_eps, new_sems)
        """
        new_eps  = self._memory_system._fetch_new_episodes_for_user(user_id)
        new_sems = self._memory_system._fetch_new_semantics_for_user(user_id)
        return new_eps, new_sems

    def update_themes(self, user_id: str, new_semantics):
        """
        用这轮 new_semantics 同步/拆分 theme 层，并在 memory_system 内部
        缓一份 ThemeManager 给后续建图用。
        """
        return self._memory_system.update_themes(user_id, new_semantics)

    def update_hierarchy_graph(self, user_id: str, new_eps, new_sems):
        """
        用本轮 new_eps/new_sems + 最新 themes 去增量更新层级图。
        """
        return self._memory_system.update_hierarchy_graph(user_id, new_eps, new_sems)

    def mark_incremental_consumed(self, user_id: str):
        """
        这一轮增量已经被 themes 和 graph 都吸收了，可以安全清空缓存。
        """
        return self._memory_system._mark_incremental_data_consumed(user_id)

    async def asearch(
        self,
        user_id: str,
        query: str,
        *,
        top_k_episodes: Optional[int] = None,
        top_k_semantic: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> Dict[str, List[Dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(
                user_id,
                query,
                top_k_episodes=top_k_episodes,
                top_k_semantic=top_k_semantic,
                search_method=search_method,
            ),
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, **kwargs: Any) -> "xMemory":
        """Create an instance using configuration sourced from env vars."""
        return cls(MemoryConfig(), **kwargs)


__all__ = ["xMemory"]
