"""Redis-backed session state. Five keys per session:

  session:{id}:stack        LIST of image bytes, RPUSH order (index 0 = original)
  session:{id}:cursor       INT index into stack of the currently active version
  session:{id}:boundaries   LIST of ints, boundaries[i] = len(messages) after step i's turn
  session:{id}:messages     pydantic-ai ModelMessage history for the full stack
  session:{id}:selection    PNG bytes of the active click-selection mask (optional)

Undo and revert only move the cursor — they never delete stack entries, so stepping
back and then forward again still shows the same pixels. Only issuing a *new* edit
while the cursor isn't at the tip drops the steps ahead of it (branch-off, same as
undo/redo semantics in most editors): push_edit trims the stack to the cursor first.

Message counts per turn aren't fixed at 2 (user + assistant) — a turn that calls the
find_region tool adds extra request/response messages for the tool round-trip. Slicing
messages by a flat `2 * cursor` can then split a tool call from its result, which
pydantic-ai rejects on the next run ("unprocessed tool calls"). `boundaries` records the
exact cut point after each real turn instead of assuming a fixed message count.
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

    def _cursor(self, sid: str) -> str:
        return f"session:{sid}:cursor"

    def _boundaries(self, sid: str) -> str:
        return f"session:{sid}:boundaries"

    def _messages(self, sid: str) -> str:
        return f"session:{sid}:messages"

    def _selection(self, sid: str) -> str:
        return f"session:{sid}:selection"

    async def create_session(self, session_id: str, initial_image_bytes: bytes) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(
                self._stack(session_id), self._cursor(session_id), self._boundaries(session_id),
                self._messages(session_id), self._selection(session_id),
            )
            pipe.rpush(self._stack(session_id), initial_image_bytes)
            pipe.set(self._cursor(session_id), 0)
            pipe.rpush(self._boundaries(session_id), 0)  # step 0 (original) has no messages yet
            pipe.set(self._messages(session_id), MessageListAdapter.dump_json([]))
            pipe.expire(self._stack(session_id), self.ttl)
            pipe.expire(self._cursor(session_id), self.ttl)
            pipe.expire(self._boundaries(session_id), self.ttl)
            pipe.expire(self._messages(session_id), self.ttl)
            await pipe.execute()

    async def get_current(self, session_id: str) -> tuple[Optional[bytes], list[ModelMessage]]:
        cursor_raw = await self.redis.get(self._cursor(session_id))
        if cursor_raw is None:
            return None, []
        cursor = int(cursor_raw)
        current = await self.redis.lindex(self._stack(session_id), cursor)
        boundary = int(await self.redis.lindex(self._boundaries(session_id), cursor))
        msg_json = await self.redis.get(self._messages(session_id))
        messages = MessageListAdapter.validate_json(msg_json) if msg_json else []
        return current, messages[:boundary]  # only the turns up to the active step

    async def push_edit(self, session_id: str, new_image_bytes: bytes, messages: list[ModelMessage]) -> None:
        """Append a new image right after the cursor, dropping any steps ahead of it."""
        cursor = int(await self.redis.get(self._cursor(session_id)))
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.ltrim(self._stack(session_id), 0, cursor)
            pipe.rpush(self._stack(session_id), new_image_bytes)
            pipe.ltrim(self._boundaries(session_id), 0, cursor)
            pipe.rpush(self._boundaries(session_id), len(messages))
            pipe.set(self._cursor(session_id), cursor + 1)
            pipe.set(self._messages(session_id), MessageListAdapter.dump_json(messages))
            pipe.expire(self._stack(session_id), self.ttl)
            pipe.expire(self._cursor(session_id), self.ttl)
            pipe.expire(self._boundaries(session_id), self.ttl)
            pipe.expire(self._messages(session_id), self.ttl)
            await pipe.execute()

    async def undo(self, session_id: str) -> bool:
        """Move the cursor back one step. Returns False if already at the original."""
        cursor_raw = await self.redis.get(self._cursor(session_id))
        if cursor_raw is None:
            return False
        return await self.revert_to(session_id, int(cursor_raw) - 1)

    async def revert_to(self, session_id: str, step: int) -> bool:
        """Move the cursor to `step` (0 = original), in either direction. Non-destructive.
        Returns False if step is out of range."""
        depth = await self.redis.llen(self._stack(session_id))
        if step < 0 or step >= depth:
            return False
        await self.redis.set(self._cursor(session_id), step, ex=self.ttl)
        return True

    async def set_selection(self, session_id: str, mask_png: bytes) -> None:
        """Store the active click-selection mask (PNG bytes); the next edit turn applies within it."""
        await self.redis.set(self._selection(session_id), mask_png, ex=self.ttl)

    async def get_selection(self, session_id: str) -> Optional[bytes]:
        return await self.redis.get(self._selection(session_id))

    async def clear_selection(self, session_id: str) -> None:
        await self.redis.delete(self._selection(session_id))
