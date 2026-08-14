"""tournament/sandbox.py — run a submitted agent in its own process.

``tournament.match`` runs agents in-process, which is right for the course
setting and for local testing. Two competition rules are only advisory there:

* **the per-move budget is measured after the fact** — an agent that never
  returns stalls the whole run rather than forfeiting one move, and
* **the forward model is a live object** — an agent that reaches past the
  documented API can mutate the real game or read the face-down plantation
  order.

This module removes both by construction, for the public run:

* Each agent runs in a **separate process**. The organizer's process owns the
  environment, so an isolated agent has nothing to corrupt and nothing to peek
  at — it never holds a reference to the real game.
* The deadline is enforced by the **parent**: it waits at most the per-move
  budget for an answer and **kills** the worker if it overruns. A hung or
  runaway agent costs itself one move, not the tournament.
* Planning agents still work. Each move the parent ships a
  :meth:`~puerto_rico.forward_model.ForwardModel.redacted_snapshot` — a
  determinized copy in which the hidden draw order has already been reshuffled
  — and the worker wraps it in a forward model that clones and simulates
  exactly as the in-process one does. Measured cost is ~2 ms per move against a
  1 s budget.

Usage::

    from tournament.sandbox import sandboxed_pool
    from tournament.runner import run_tournament

    pool, close = sandboxed_pool({"TeamA": "submissions/team_a.py:TeamAgent"},
                                 n_seats=3)
    try:
        result = run_tournament({**pool, "ActionValue": ActionValueAgent}, n_seats=3)
    finally:
        close()

Submissions are named the same way as in the web UI: ``module:Class`` or
``path/to/file.py:Class``.

Note: this module deliberately avoids importing numpy (or anything else heavy)
at module level. A spawned worker imports it before it can set the thread-limit
environment variables that keep a shared machine usable.
"""
import importlib
import importlib.util
import itertools
import multiprocessing
import os

DEFAULT_STARTUP_TIMEOUT_S = 30.0   # generous: a worker may import torch
DEFAULT_MEM_LIMIT_MB = 4096        # per-agent address-space cap (None disables)
TIMEOUT_ACTION = -1                # not a legal action: the harness substitutes


class SandboxError(RuntimeError):
    """The worker died, or could not be started at all."""


# ── loading a submission ─────────────────────────────────────────────────────

def load_agent_class(spec: str, base_dir: str = None):
    """Resolve ``"module:Class"`` or ``"path/to/file.py:Class"`` to a class.

    Args:
        spec: an importable ``module:Class``, or a ``file.py:Class`` path.
        base_dir: what a *relative* path is relative to (default: the working
            directory). The sandbox passes the repo root so a submission
            resolves the same way in the worker as it does for the organizer.

    Raises:
        ValueError / TypeError: malformed spec, or the target is not an
            :class:`~agents.base.Agent` subclass.
    """
    from agents.base import Agent

    if ":" not in spec:
        raise ValueError(
            f"agent spec must be 'module:Class' or 'file.py:Class', got {spec!r}")
    mod_part, cls_name = spec.rsplit(":", 1)

    if mod_part.endswith(".py") or "/" in mod_part or os.sep in mod_part:
        path = (mod_part if os.path.isabs(mod_part)
                else os.path.abspath(os.path.join(base_dir or os.getcwd(), mod_part)))
        module_spec = importlib.util.spec_from_file_location("_sandboxed_agent", path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot import an agent from {path!r}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)

    cls = getattr(module, cls_name)
    if not (isinstance(cls, type) and issubclass(cls, Agent)):
        raise TypeError(f"{cls_name} is not an agents.base.Agent subclass")
    return cls


# ── worker process ───────────────────────────────────────────────────────────

def _apply_limits(mem_limit_mb):
    """Best-effort resource caps. Unavailable limits are simply not applied."""
    # Keep a shared machine usable: an agent that spawns a BLAS thread per core
    # is a much more common accident than a malicious one.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if mem_limit_mb is None:
        return
    try:
        import resource
        limit = int(mem_limit_mb) * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except Exception:
        pass                          # not available (e.g. Windows) — isolation
                                      # and the deadline are the real defences


def _worker_main(conn, spec, mem_limit_mb, repo_root):
    """Child entry point: own one agent instance and answer act() requests."""
    _apply_limits(mem_limit_mb)

    import sys
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from agents.base import Agent
        cls = load_agent_class(spec, base_dir=repo_root)
    except BaseException as exc:                       # noqa: BLE001
        conn.send(("fatal", f"{type(exc).__name__}: {exc}"))
        return

    # Only planning agents pay for a snapshot: a reactive agent that never looks
    # at the forward model should not be charged ~2 ms a move to receive one.
    needs_model = cls.on_game_start is not Agent.on_game_start
    conn.send(("ready", (needs_model, getattr(cls, "name", cls.__name__))))

    agent = None
    model = None

    while True:
        try:
            kind, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
            return

        if kind == "close":
            return

        if kind == "new_game":
            try:
                agent = cls()
                model = _make_model(payload) if payload is not None else None
                agent.on_game_start(model)
                conn.send(("ok", None))
            except BaseException as exc:               # noqa: BLE001
                agent = None
                conn.send(("err", f"{type(exc).__name__}: {exc}"))

        elif kind == "act":
            observation, action_mask, snapshot = payload
            try:
                if model is not None and snapshot is not None:
                    model._rebind(snapshot)
                conn.send(("ok", int(agent.act(observation, action_mask))))
            except BaseException as exc:               # noqa: BLE001
                conn.send(("err", f"{type(exc).__name__}: {exc}"))


_REBINDABLE_MODEL = None


def _make_model(env):
    """A forward model whose backing snapshot is replaced each move.

    The agent keeps the reference it was handed in ``on_game_start``; the worker
    swaps the game underneath it as the real match advances. Every snapshot has
    already been determinized by the parent, so the true draw order never enters
    this process.

    The class is built on first use rather than at import time: a spawned worker
    imports this module before it can set the thread-limit environment
    variables, and defining the class eagerly would drag in numpy with it.
    """
    global _REBINDABLE_MODEL

    if _REBINDABLE_MODEL is None:
        from puerto_rico import ForwardModel

        class _RebindableModel(ForwardModel):
            def _rebind(self, new_env):
                self._env = new_env
                self._redacted = None
                self._redacted_step = object()

        _REBINDABLE_MODEL = _RebindableModel

    return _REBINDABLE_MODEL(env, _live=True)


# ── parent-side handle ───────────────────────────────────────────────────────

class AgentWorker:
    """One long-lived worker process, reused across games.

    A fresh agent instance is built inside the worker for every game (the
    competition rule), but the process — and therefore the cost of importing
    numpy/torch — is paid once.
    """

    def __init__(self, spec: str, *, mem_limit_mb=DEFAULT_MEM_LIMIT_MB,
                 startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
                 repo_root: str = None):
        self.spec = spec
        self.mem_limit_mb = mem_limit_mb
        self.startup_timeout_s = startup_timeout_s
        self.repo_root = repo_root or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.agent_name = spec.rsplit(":", 1)[-1]
        self.needs_model = True
        # Bumped on every (re)start. A proxy compares it against the generation
        # it last set up a game on, so it can tell that "its" worker is really a
        # replacement that has never been told a game is in progress.
        self.generation = 0
        self._proc = None
        self._conn = None
        self._ctx = multiprocessing.get_context("spawn")

    # -- lifecycle --
    def ensure_started(self):
        """Start the worker if it is not running. Idempotent."""
        if self._proc is not None and self._proc.is_alive():
            return
        self._spawn()

    def _spawn(self):
        self._teardown()
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self.spec, self.mem_limit_mb, self.repo_root),
            daemon=True)
        proc.start()
        child_conn.close()
        self._proc, self._conn = proc, parent_conn
        self.generation += 1

        if not parent_conn.poll(self.startup_timeout_s):
            self._teardown()
            raise SandboxError(
                f"{self.spec}: worker did not start within {self.startup_timeout_s}s")
        kind, payload = parent_conn.recv()
        if kind == "fatal":
            self._teardown()
            raise SandboxError(f"{self.spec}: {payload}")
        self.needs_model, self.agent_name = payload

    def _teardown(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.kill()
            self._proc.join(timeout=5)
            self._proc = None

    def close(self):
        """Ask the worker to exit, then make sure it did."""
        if self._proc is not None and self._proc.is_alive() and self._conn is not None:
            try:
                self._conn.send(("close", None))
                self._proc.join(timeout=2)
            except Exception:
                pass
        self._teardown()

    # -- request/response --
    def request(self, kind, payload, deadline_s):
        """Send one message and wait at most ``deadline_s`` for the reply.

        Returns ``(True, value)`` on success and ``(False, message)`` if the
        agent raised. Raises :class:`TimeoutError` when the budget expires (the
        worker is killed, and restarts on the next request) and
        :class:`SandboxError` if the process died on its own.
        """
        self.ensure_started()
        try:
            self._conn.send((kind, payload))
        except (BrokenPipeError, OSError) as exc:
            self._teardown()
            raise SandboxError(f"{self.spec}: worker pipe closed ({exc})") from exc

        if not self._conn.poll(deadline_s):
            # Over budget: kill it now so a runaway agent cannot keep burning
            # CPU while the rest of the tournament waits behind it.
            self._teardown()
            raise TimeoutError(f"{self.spec}: no answer within {deadline_s:.3f}s")

        try:
            reply_kind, value = self._conn.recv()
        except (EOFError, OSError) as exc:
            self._teardown()
            raise SandboxError(f"{self.spec}: worker died mid-move ({exc})") from exc

        if reply_kind == "ok":
            return True, value
        return False, value


# ── the Agent-facing proxy ───────────────────────────────────────────────────

class SandboxedAgent:
    """Stand-in for a submitted agent that actually runs in ``worker``.

    Implements the plain agent interface (``on_game_start`` / ``act``), so the
    tournament harness cannot tell the difference — except that this one can be
    trusted with the game.
    """

    def __init__(self, spec: str, *, worker: AgentWorker = None, name: str = None,
                 time_limit_s: float = 1.0, mem_limit_mb=DEFAULT_MEM_LIMIT_MB,
                 max_consecutive_timeouts: int = 3):
        self.spec = spec
        self.time_limit_s = time_limit_s
        # Recovering from a timeout means starting a process again. An agent
        # that overruns *every* move would otherwise make the organizer pay that
        # cost hundreds of times per game (measured: ~227 s for one game) and
        # hold up everyone else's matches. After this many in a row we stop
        # restarting it and it simply forfeits the rest of the game.
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self._owns_worker = worker is None
        self.worker = worker or AgentWorker(spec, mem_limit_mb=mem_limit_mb)
        self._explicit_name = name
        self._model = None
        self._generation = None
        self._consecutive_timeouts = 0
        self.forfeited = False
        self.timeouts = 0
        self.violations = 0

    @property
    def name(self):
        return self._explicit_name or self.worker.agent_name

    def on_game_start(self, forward_model=None):
        self._model = forward_model
        self.worker.ensure_started()
        self._start_game()

    def _start_game(self):
        """Build a fresh agent instance inside the worker for this game."""
        snapshot = None
        if self._model is not None and self.worker.needs_model:
            snapshot = self._model.redacted_snapshot()
        generation = self.worker.generation
        ok, message = self.worker.request("new_game", snapshot,
                                          self.worker.startup_timeout_s)
        if not ok:
            self.violations += 1
            raise SandboxError(f"{self.spec}: on_game_start failed — {message}")
        self._generation = generation

    def act(self, observation, action_mask):
        if self.forfeited:
            self.violations += 1
            raise SandboxError(
                f"{self.spec}: forfeited after {self.max_consecutive_timeouts} "
                f"consecutive timeouts; not restarted again")

        # A worker that was killed (timed out, crashed, hit its memory cap) comes
        # back with no agent in it. Set the game up again before asking it to
        # move, or every remaining move of the match would be a violation
        # against an agent that no longer exists.
        if self.worker.generation != self._generation:
            self._recover()

        snapshot = None
        if self._model is not None and self.worker.needs_model:
            snapshot = self._model.redacted_snapshot()

        # Give the worker the full budget plus a hair, so that when it does
        # overrun, the elapsed time the harness measures is unambiguously over
        # the limit and the move is booked as a timeout rather than a violation.
        try:
            ok, value = self.worker.request(
                "act", (observation, action_mask, snapshot),
                self.time_limit_s + 0.05)
        except TimeoutError:
            self.timeouts += 1
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts > self.max_consecutive_timeouts:
                self.forfeited = True
            else:
                # Restart now rather than on the next move: this move is already
                # over budget, so charging the restart to it keeps a single slow
                # move from cascading into a timeout on every move after it.
                self._recover()
            return TIMEOUT_ACTION
        except SandboxError:
            self.violations += 1
            raise

        if not ok:
            self.violations += 1
            raise RuntimeError(f"{self.spec}: {value}")
        self._consecutive_timeouts = 0
        return int(value)

    def _recover(self):
        """Bring a replacement worker back into the current game, best-effort.

        The agent's own accumulated state is gone — that is part of the cost of
        overrunning the budget — but it can go on playing the rest of the match.
        """
        try:
            self.worker.ensure_started()
            self._start_game()
        except Exception:
            self._generation = None       # try again before the next move

    def close(self):
        if self._owns_worker:
            self.worker.close()


# ── building a tournament pool ───────────────────────────────────────────────

def sandboxed_pool(specs: dict, n_seats: int = 3, *, time_limit_s: float = 1.0,
                   mem_limit_mb=DEFAULT_MEM_LIMIT_MB):
    """Turn ``{name: spec}`` into a ``{name: factory}`` pool of isolated agents.

    Args:
        specs: ``{leaderboard_name: "module:Class" | "file.py:Class"}``.
        n_seats: seats per game. Each name gets this many worker processes and
            hands them out in rotation, so a game that seats the same agent
            more than once (the round-robin does play ``[A, A, B]``) still gives
            every seat its own process and its own agent instance.
        time_limit_s: per-move budget enforced by the parent.
        mem_limit_mb: address-space cap per worker; ``None`` disables it.

    Returns:
        ``(pool, close)`` — pass ``pool`` to
        :func:`tournament.runner.run_tournament`, and **call ``close()`` when
        the run finishes** (a ``try/finally`` is the tidy way) to reap the
        worker processes.
    """
    workers = {}
    cycles = {}

    for name, spec in specs.items():
        workers[name] = [AgentWorker(spec, mem_limit_mb=mem_limit_mb)
                         for _ in range(max(1, n_seats))]
        cycles[name] = itertools.cycle(workers[name])

    def make_factory(name, spec):
        def factory():
            return SandboxedAgent(spec, worker=next(cycles[name]), name=name,
                                  time_limit_s=time_limit_s)
        return factory

    pool = {name: make_factory(name, spec) for name, spec in specs.items()}

    def close():
        for group in workers.values():
            for worker in group:
                worker.close()
        workers.clear()

    return pool, close
