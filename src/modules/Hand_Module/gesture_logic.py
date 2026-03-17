# In src/modules/gesture_logic.py
import time
from typing import Dict, List, Optional, Tuple

class GestureProcessor:
    def __init__(self):
        # Your Original Config
        self.config = {
            (("FOUR_FINGERS", "THREE_FINGERS", "TWO_FINGERS", "ONE_FINGER"), 1.0): ("esc",),
            # (("Open_Palm", "Closed_Fist"), 1.0): ("alt", "tab"),
            # (("Thumb_Up", "Thumb_Down"), 2.0): ("alt", "tab"),
            (("Victory", "ILoveYou"), 2.0): ("alt", "tab"),
        }
        self.history = []
        self.prev_active = set()
        self.max_window = 4.0

    def process_frame(self, canned: Optional[str], custom: List[str]) -> Optional[Tuple[str, ...]]:
        now = time.time()
        active = set(custom)
        if canned and canned != "None": active.add(canned)

        # Transition logic
        new_gestures = active - self.prev_active
        for g in new_gestures:
            self.history.append((now, g))
        self.prev_active = active

        # Clean old history
        self.history = [(t, g) for (t, g) in self.history if t >= (now - self.max_window)]

        # Sequence Matching (Your Subsequence Logic)
        for (pattern, window_sec), binding in self.config.items():
            n = len(pattern)
            if len(self.history) < n: continue

            names = [g for (_, g) in self.history]
            idx = len(names) - 1
            matches = []
            for p in reversed(pattern):
                while idx >= 0 and names[idx] != p: idx -= 1
                if idx >= 0:
                    matches.append(idx)
                    idx -= 1
            
            if len(matches) == n:
                matches.reverse()
                if (self.history[matches[-1]][0] - self.history[matches[0]][0]) <= window_sec:
                    self.history.clear()
                    return binding # Return the keyboard shortcut
        return None