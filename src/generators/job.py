from dataclasses import dataclass, field
import time
import threading
from typing import Any, Dict, Optional

@dataclass
class llmJob:
    id: str
    prompt: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    api: str = "completion"  # or "chat-completion"
    messages: Optional[list] = None  # For chat-completion
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None
    error: Optional[str] = None
    created_ts: float = field(default_factory=time.time)
