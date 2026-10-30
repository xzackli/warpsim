"""Bounded diagnostic runs using the existing native compiler/runtime adapter."""


def run_native(
    program,
    *,
    until,
    drives=(),
    initial=None,
    max_events=256,
    device="cpu",
    mode="release",
    tables=None,
    trace_cells=None,
    memory_limit=256 * 1024**2,
    trace_mode="full",
    **execution_options,
):
    from .session import Scenario, Session

    session = Session(
        program,
        (Scenario(drives, initial, tables),),
        device=device,
        mode=mode,
        trace_cells=trace_cells,
        trace_mode=trace_mode,
        memory_limit=memory_limit,
        **execution_options,
    )
    return session.step(until=until, max_events=max_events)[0]


def decode_calendar(l, row):
    from .lower import MAX

    nc, nt = len(l.program.clocks), len(l.program.timers)
    return {
        "clocks": {
            c.name: int(row[12 + j]) if row[12 + j] <= MAX else None
            for j, c in enumerate(l.program.clocks)
        },
        "timers": {
            t.name: (
                None
                if row[12 + nc + 3 * j] > MAX
                else (int(row[12 + nc + 3 * j]), int(row[13 + nc + 3 * j])),
                int(row[14 + nc + 3 * j]),
            )
            for j, t in enumerate(l.program.timers)
        },
        "inputs": {
            v.name: int(row[12 + nc + 3 * nt + j])
            for j, v in enumerate(l.program.inputs)
        },
        "cursor": int(row[8]),
    }
