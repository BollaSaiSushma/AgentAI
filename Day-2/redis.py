import os
import json
import time
import uuid
import hashlib
from typing import List, Dict, Optional

from dotenv import load_dotenv

import redis


# ...

# self.r = Redis.from_url(Url, decode_responses=True)  # <-- removed erroneous line

# -----------------------------
# Minimal LLM wrapper (OpenAI)
# -----------------------------
class LLM:
    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        raise NotImplementedError

class OpenAILLM(LLM):
    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

# -----------------------------
# Redis helpers
# -----------------------------
class RedisStore:
    """
    - Chat history list:  key = chat:{session_id}:messages (RPUSH JSON, LTRIM)
    - Answer cache string: key = cache:{hash} (SETEX)
    """
    def __init__(self, url: str):
        self.url = url
        self.r = redis.from_url(url, decode_responses=True)
        # health check
        try:
            self.r.ping()
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Redis at {url}: {e}")

    # ---- Chat History ----
    def append_message(self, session_id: str, msg: Dict[str, str], max_messages: int = 40) -> None:
        key = f"chat:{session_id}:messages"
        self.r.rpush(key, json.dumps(msg))
        # keep only the last N
        self.r.ltrim(key, max(0, -max_messages), -1)

    def load_history(self, session_id: str, max_messages: int = 40) -> List[Dict[str, str]]:
        key = f"chat:{session_id}:messages"
        raw = self.r.lrange(key, max(0, -max_messages), -1)
        msgs = []
        for s in raw:
            try:
                msgs.append(json.loads(s))
            except Exception:
                pass
        return msgs

    def clear_history(self, session_id: str) -> None:
        key = f"chat:{session_id}:messages"
        self.r.delete(key)

    # ---- Answer Cache ----
    def get_cached(self, key_hash: str) -> Optional[str]:
        val = self.r.get(f"cache:{key_hash}")
        return val

    def set_cached(self, key_hash: str, answer: str, ttl: int) -> None:
        self.r.setex(f"cache:{key_hash}", ttl, answer)

# -----------------------------
# Chatbot core
# -----------------------------
class RedisChatbot:
    def __init__(self, llm: LLM, store: RedisStore,
                 system_prompt: str = "You are a helpful assistant.",
                 history_max_messages: int = 40,
                 cache_ttl: int = 3600):
        self.llm = llm
        self.store = store
        self.system_prompt = system_prompt
        self.history_max = history_max_messages
        self.cache_ttl = cache_ttl

    def _make_cache_key(self, session_id: str, user_text: str, history: List[Dict[str, str]]) -> str:
        """
        Build a stable cache key from:
          - system prompt (shortened),
          - session id,
          - last ~10 messages (truncated for size),
          - the new user_text.
        """
        tail = history[-10:] if history else []
        def norm(s: str) -> str:
            s = (s or "").strip()
            return s[:500]  # cap to reduce key size while keeping main signal

        blob = {
            "sid": session_id,
            "sys": self.system_prompt[:200],
            "history": [{"r": m.get("role"), "c": norm(m.get("content", ""))} for m in tail],
            "user": norm(user_text),
        }
        j = json.dumps(blob, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(j.encode("utf-8")).hexdigest()

    def ask(self, session_id: str, user_text: str, temperature: float = 0.2) -> str:
        # 1) Load history
        history = self.store.load_history(session_id, self.history_max)

        # 2) Check cache
        key_hash = self._make_cache_key(session_id, user_text, history)
        cached = self.store.get_cached(key_hash)
        if cached:
            # still append messages to history so conversation is coherent
            self.store.append_message(session_id, {"role": "user", "content": user_text}, self.history_max)
            self.store.append_message(session_id, {"role": "assistant", "content": cached}, self.history_max)
            return cached + "  \n_(cached)_"

        # 3) Build prompt (system + history + new user)
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        # 4) Query LLM
        answer = self.llm.complete(messages, temperature=temperature)

        # 5) Save to cache & history
        self.store.set_cached(key_hash, answer, ttl=self.cache_ttl)
        self.store.append_message(session_id, {"role": "user", "content": user_text}, self.history_max)
        self.store.append_message(session_id, {"role": "assistant", "content": answer}, self.history_max)

        return answer

# -----------------------------
# CLI runner
# -----------------------------
def main():
    load_dotenv()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_ttl = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    history_max = int(os.getenv("HISTORY_MAX_MESSAGES", "40"))

    # Init dependencies
    store = RedisStore(redis_url)
    llm = OpenAILLM()
    bot = RedisChatbot(llm, store, history_max_messages=history_max, cache_ttl=cache_ttl)

    # New session id each run (or set your own stable id per user)
    session_id = os.getenv("SESSION_ID") or str(uuid.uuid4())
    print(f"🗣️  Redis Chatbot ready (session={session_id}). Type 'clear' to clear history, 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if user.lower() == "clear":
            store.clear_history(session_id)
            print("(history cleared)")
            continue

        t0 = time.time()
        try:
            reply = bot.ask(session_id, user)
        except Exception as e:
            reply = f"Error: {e}"
        dt = (time.time() - t0) * 1000
        print(f"Bot ({dt:.0f} ms): {reply}\n")

if __name__ == "__main__":
    main()
