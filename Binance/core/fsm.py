from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, List
from core.events import Event

@dataclass
class Transition:
    """
    Encapsulates one state transition in an FSM agent.
    - initial: name of the starting state
    - final:   name of the next state
    - sensor:    function to extract a value from an Event
    - predicate: tests the sensor output to arm the transition
    - actuator:  executes side-effects when firing
    """
    initial:   str
    final:     str
    sensor:    Callable[[Event], Any]
    predicate: Callable[[Any], bool]
    actuator:  Callable[[Any], None]

    def armed(self, evt: Event) -> bool:
        """Is this transition ready to fire on evt?"""
        return self.predicate(self.sensor(evt))

    def fire(self, evt: Event) -> None:
        """Execute the transition’s actuator."""
        self.actuator(self.sensor(evt))


class Agent:
    """
    Base FSM agent class.
    Subclass and override observe, preprocess, main, postprocess to implement logic.
    """
    name:        str
    states:      List[str]
    current:     str
    transitions: List[Transition]

    def __init__(self, name: str = "agent"):
        self.name = name
        self.states = []
        self.current = ""
        self.transitions = []

    def observe(self, e: Event) -> bool:
        """
        Event filter — return True to accept this event, False to ignore.
        Default: accept all events.
        """
        return True

    def preprocess(self, e: Event) -> None:
        """
        Optional hook before main().
        Use for updating internal state, indicators, positions, etc.
        """
        pass

    def main(self, e: Event) -> None:
        """
        Core FSM logic.
        Often: loop through self.transitions, fire if predicate holds,
        update self.current, emit side-effects.
        """
        pass

    def postprocess(self, e: Event) -> None:
        """
        Optional hook after main().
        Use for cleanup, logging, recording timestamps, etc.
        """
        pass

    def consume(self, e: Event) -> None:
        """
        Full agent dispatch for a single Event:
          1. filter
          2. preprocess
          3. main
          4. postprocess
        """
        if self.observe(e):
            self.preprocess(e)
            self.main(e)
            self.postprocess(e)