import asyncio

import pytest

from robocell.clock import SimClock


async def _run_actors(
    n_actors: int, ticks: int, pace: float | None = None
) -> list[tuple[int, str]]:
    clock = SimClock(0.01)
    log: list[tuple[int, str]] = []

    async def actor(name: str) -> None:
        while True:
            tick = await clock.wait_tick(name)
            log.append((tick, name))
            await asyncio.sleep(0)  # yielding mid-tick must not let the clock advance

    names = [f"a{i}" for i in range(n_actors)]
    for name in names:
        clock.register(name)
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(actor(n)) for n in names]
        finished = await clock.run(lambda: clock.tick >= ticks, max_ticks=10_000, pace=pace)
        for t in tasks:
            t.cancel()
    assert finished
    return log


def test_lock_step_in_registration_order() -> None:
    log = asyncio.run(_run_actors(3, 5))
    assert log == [(t, f"a{i}") for t in range(1, 6) for i in range(3)]


def test_runs_are_deterministic() -> None:
    assert asyncio.run(_run_actors(4, 50)) == asyncio.run(_run_actors(4, 50))


def test_pacing_does_not_change_results() -> None:
    assert asyncio.run(_run_actors(2, 5, pace=1000.0)) == asyncio.run(_run_actors(2, 5))


def test_max_ticks_reported() -> None:
    async def main() -> tuple[bool, int]:
        clock = SimClock(0.01)
        return await clock.run(lambda: False, max_ticks=7), clock.tick

    assert asyncio.run(main()) == (False, 7)


def test_time_and_tick_conversion() -> None:
    clock = SimClock(0.01)
    assert clock.ticks(1.5) == 150
    assert clock.ticks(0.001) == 1
    assert clock.ticks(0.0) == 0
    clock.tick = 250
    assert clock.time == pytest.approx(2.5)


def test_registration_errors() -> None:
    clock = SimClock()
    clock.register("x")
    with pytest.raises(ValueError, match="already"):
        clock.register("x")

    async def main() -> None:
        await clock.wait_tick("nobody")

    with pytest.raises(RuntimeError, match="not registered"):
        asyncio.run(main())


def test_actor_crash_propagates_instead_of_hanging() -> None:
    async def main() -> None:
        clock = SimClock()
        clock.register("bad")
        clock.register("good")

        async def bad() -> None:
            while True:
                if await clock.wait_tick("bad") == 3:
                    raise RuntimeError("boom")

        async def good() -> None:
            while True:
                await clock.wait_tick("good")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(bad())
            tg.create_task(good())
            await clock.run(lambda: False, max_ticks=100)

    with pytest.raises(ExceptionGroup) as info:
        asyncio.run(main())
    assert info.group_contains(RuntimeError, match="boom")


def test_unregister_releases_driver() -> None:
    async def main() -> int:
        clock = SimClock()
        clock.register("a")
        clock.register("b")

        async def short() -> None:
            await clock.wait_tick("a")
            clock.unregister("a")

        async def forever() -> None:
            while True:
                await clock.wait_tick("b")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(short())
            t = tg.create_task(forever())
            await clock.run(lambda: clock.tick >= 5, max_ticks=100)
            t.cancel()
        return clock.tick

    assert asyncio.run(main()) == 5
