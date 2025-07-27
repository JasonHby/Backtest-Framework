from __future__ import annotations
from collections import deque
from typing import Deque, List

from core.events import Event
from core.fsm import Agent

# ────────────────────────────────
# 1. Global event queue & `emit`
# ────────────────────────────────
EVENT_Q: Deque[Event] = deque()

def emit(evt: Event) -> None:
    """
    Any agent can call `emit(...)` to push a new Event
    into the shared global queue.
    """
    EVENT_Q.appendleft(evt)

# ────────────────────────────────Z
# 2. Run-loop / simulation driver
# ────────────────────────────────
def run_simulation(agents: List[Agent], events: List[Event]) -> None:
    """
    Core orchestrator: seed the queue with initial events,
    then repeatedly pop one event and hand it to every agent.
    Agents may emit more events, keeping the loop going until
    nothing remains.
    """
    # initialize
    EVENT_Q.extend(events)

    # dispatch loop
    while EVENT_Q:
        evt = EVENT_Q.popleft()
        for agent in agents:
            agent.consume(evt)