"""Redis-backed session state. Three keys per session:

  session:{id}:stack      LIST of image bytes, newest at head (index 0 = current image)
  session:{id}:messages   pydantic-ai ModelMessage history (JSON)
  session:{id}:selection  PNG bytes of the active click-selection mask (optional)

The image stack is what makes undo actually restore pixels (v1's undo only trimmed
messages and re-saved the already-edited image). The selection key holds the mask a
click produced (via the /select route), so the next edit turn applies only within it —
the API equivalent of the Gradio UI's gr.State selection.
"""
from typing import Optional

import redis.asyncio as aioredis
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage

MessageListAdapter = TypeAdapter(list[ModelMessage])


class RedisSessionManager:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = 3600):
        self.redis = aioredis.from_url(redis_url, decode_responses=False)
        self.ttl = ttl_seconds

    def _stack(self, sid: str) -> str:
        return f"session:{sid}:stack"

    def _messages(self, sid: str) -> str:
        return f"session:{sid}:messages"

    def _selection(self, sid: str) -> str:
        return f"session:{sid}:selection"

    async def create_session(self, session_id: str, initial_image_bytes: bytes) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(self._stack(session_id), self._messages(session_id), self._selection(session_id))
            pipe.lpush(self._stack(session_id), initial_image_bytes)
            pipe.set(self._messages(session_id), MessageListAdapter.dump_json([]))
            pipe.expire(self._stack(session_id), self.ttl)
            pipe.expire(self._messages(session_id), self.ttl)
            await pipe.execute()

    async def get_current(self, session_id: str) -> tuple[Optional[bytes], list[ModelMessage]]:
        current = await self.redis.lindex(self._stack(session_id), 0)
        msg_json = await self.redis.get(self._messages(session_id))
        messages = MessageListAdapter.validate_json(msg_json) if msg_json else []
        return current, messages

    async def push_edit(self, session_id: str, new_image_bytes: bytes, messages: list[ModelMessage]) -> None:
        """Push a new image onto the stack and persist the updated message history."""
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.lpush(self._stack(session_id), new_image_bytes)
            pipe.set(self._messages(session_id), MessageListAdapter.dump_json(messages))
            pipe.expire(self._stack(session_id), self.ttl)
            pipe.expire(self._messages(session_id), self.ttl)
            await pipe.execute()

    async def undo(self, session_id: str) -> bool:
        """Pop the latest image (keeping at least the original) and drop the last message
        turn. Returns False if there is nothing to undo."""
        depth = await self.redis.llen(self._stack(session_id))
        if depth < 2:
            return False
        _, messages = await self.get_current(session_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.lpop(self._stack(session_id))
            pipe.set(self._messages(session_id), MessageListAdapter.dump_json(messages[:-2]))
            pipe.expire(self._stack(session_id), self.ttl)
            pipe.expire(self._messages(session_id), self.ttl)
            await pipe.execute()
        return True

    async def revert_to(self, session_id: str, step: int) -> bool:
        """Trim the stack back to `step` (0 = original), dropping later images + message turns.
        Reuses undo so message history stays consistent. Returns False if step is out of range."""
        depth = await self.redis.llen(self._stack(session_id))
        if step < 0 or step >= depth:
            return False
        for _ in range(depth - 1 - step):
            await self.undo(session_id)
        return True

    async def set_selection(self, session_id: str, mask_png: bytes) -> None:
        """Store the active click-selection mask (PNG bytes); the next edit turn applies within it."""
        await self.redis.set(self._selection(session_id), mask_png, ex=self.ttl)

    async def get_selection(self, session_id: str) -> Optional[bytes]:
        return await self.redis.get(self._selection(session_id))

    async def clear_selection(self, session_id: str) -> None:
        await self.redis.delete(self._selection(session_id))
