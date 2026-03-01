from __future__ import annotations

import multiprocessing as mp
import pickle

import numpy as np

MAX_OBS_BYTES = 10 * 1024 * 1024  # 10MB per slot, enough for a few 224x224 images


class RobotSlot:
    def __init__(self):
        self._buf = mp.RawArray("B", MAX_OBS_BYTES)  # 'B' = uint8
        self._size = mp.RawValue("i", 0)
        self._lock = mp.Lock()
        self._np_buf = np.frombuffer(self._buf, dtype=np.uint8)  # zero-copy view

    def write_obs(self, obs: dict) -> None:
        data = pickle.dumps(obs)
        n = len(data)
        with self._lock:
            self._np_buf[:n] = np.frombuffer(data, dtype=np.uint8)
            self._size.value = n

    def read_obs(self) -> dict:
        with self._lock:
            data = bytes(self._np_buf[: self._size.value])  # copy under lock
        return pickle.loads(data)  # unpickle outside lock


class RobotSlots:
    """Pre-allocated slots shared across fork. Created in main process before forking."""

    def __init__(self, max_robots: int):
        self._slots = [RobotSlot() for _ in range(max_robots)]
        self._free: list[int] = list(range(max_robots))
        # robot_id→slot_index mapping lives only in WS main process — scheduler never accesses slot assignments
        self._robot_to_slot: dict[str, int] = {}

    def register(self, robot_id: str) -> int:
        idx = self._free.pop()
        self._robot_to_slot[robot_id] = idx
        return idx

    def slot_for(self, robot_id: str) -> int:
        return self._robot_to_slot[robot_id]

    def has_robot(self, robot_id: str) -> bool:
        return robot_id in self._robot_to_slot

    def write_obs(self, slot_idx: int, obs: dict) -> None:
        self._slots[slot_idx].write_obs(obs)

    def read_obs(self, slot_idx: int) -> dict:
        return self._slots[slot_idx].read_obs()

    def free(self, robot_id: str) -> None:
        idx = self._robot_to_slot.pop(robot_id)
        self._free.append(idx)
