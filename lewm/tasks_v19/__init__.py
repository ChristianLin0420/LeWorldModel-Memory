"""V19 P1a task suite: certified pixel+action memory tasks (see docs/V19_PROPOSAL.md 4.4).

Registry: ``TASKS`` lists the four confirmation tasks (t1-t4) and the two
development-only variants (t1dev, t2dev); ``make_task`` instantiates by name.
"""

from lewm.tasks_v19.base import EpisodeBatch, V19Task, load_bank, save_bank
from lewm.tasks_v19.tasks import TASKS, make_task

__all__ = ["TASKS", "make_task", "EpisodeBatch", "V19Task", "save_bank", "load_bank"]
