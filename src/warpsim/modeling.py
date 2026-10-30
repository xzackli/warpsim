"""Typed component elaboration into the closed semantic core.

Connections sample entry values, transactions declare shared rollback domains,
and explicit ordering constraints determine the committed prefix. No component
callback, model-name dispatch, or implicit arbitration survives elaboration.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from .core.ir import (
    BOOL,
    Bits,
    Cancel,
    Cell,
    Clock,
    Expr,
    If,
    Input,
    Let,
    Observe,
    Program,
    Region,
    Repeat,
    Require,
    Schedule,
    Store,
    Table,
    const,
    local,
    op,
)
from .core.lower import dtype, identifier, lower


@dataclass(frozen=True)
class Port:
    name: str
    type: Bits
    clock: str


@dataclass(frozen=True)
class Output:
    name: str
    value: Expr
    clock: str


@dataclass(frozen=True)
class Component:
    name: str
    cells: tuple
    clocks: tuple
    inputs: tuple
    outputs: tuple
    actions: tuple
    captures: tuple = ()
    tables: tuple = ()
    version: str = "1"
    timers: tuple = ()


@dataclass(frozen=True)
class Instance:
    name: str
    component: Component
    clocks: tuple  # (formal clock, model clock) pairs


@dataclass(frozen=True)
class PortRef:
    instance: str
    port: str


@dataclass(frozen=True)
class Binding:
    target: PortRef
    source: PortRef | Expr


@dataclass(frozen=True)
class ActionRef:
    instance: str
    action: str


@dataclass(frozen=True)
class Transaction:
    name: str
    actions: tuple
    repeat: PortRef | Expr | None = None
    limit: int = 1


@dataclass(frozen=True)
class Model:
    name: str
    clocks: tuple
    inputs: tuple
    instances: tuple
    bindings: tuple
    transactions: tuple
    before: tuple = ()  # explicit (earlier, later) transaction constraints
    stop: PortRef | Expr = const(0, BOOL)


@dataclass(frozen=True)
class Origin:
    path: str
    instance: str
    component: str
    version: str
    action: str
    source_path: str


@dataclass(frozen=True)
class ScheduleCertificate:
    order: tuple
    before: tuple
    atomic_groups: tuple
    components: tuple
    port_dependencies: tuple
    effects: tuple
    semantic_hash: str


@dataclass(frozen=True)
class Elaboration:
    program: Program
    certificate: ScheduleCertificate
    origins: tuple

    def origin(self, location):
        return next((x for x in self.origins if x.path == location), None)


def _tuple(value, what):
    if type(value) is not tuple or len(value) > 256:
        raise ValueError(f"{what} must be a tuple of at most 256 entries")


def _unique(items, what):
    result = {}
    for item in items:
        identifier(item.name)
        if item.name in result:
            raise ValueError(f"Duplicate {what}: {item.name}")
        result[item.name] = item
    return result


def _order(names, constraints):
    """Require a unique schedule: even disjoint failures have an ordering effect."""
    edges = set()
    for edge in constraints:
        if (
            type(edge) is not tuple
            or len(edge) != 2
            or any(x not in names for x in edge)
        ):
            raise ValueError("Order constraints must name two transactions")
        if edge[0] == edge[1]:
            raise ValueError("Cyclic transaction order")
        edges.add(edge)
    remaining, result = set(names), []
    while remaining:
        available = [
            n for n in remaining if not any(b == n and a in remaining for a, b in edges)
        ]
        if not available:
            raise ValueError("Cyclic transaction order")
        if len(available) != 1:
            raise ValueError(
                "Ambiguous transaction/failure order; add explicit before constraints"
            )
        result.append(available[0])
        remaining.remove(available[0])
    return tuple(result)


def _effects(l):
    return tuple(
        (
            name,
            tuple(
                (key, tuple(sorted(value)) if isinstance(value, set) else value)
                for key, value in effect.items()
            ),
        )
        for name, effect in l.effects.items()
    )


def verify_schedule(program, certificate):
    """Check the expanded program against its explicit order/effect certificate."""
    if type(certificate) is not ScheduleCertificate:
        raise ValueError("Expected ScheduleCertificate")
    l = lower(program)
    names = tuple(r.name for r in program.schedule)
    if (
        names != certificate.order
        or _order(names, certificate.before) != names
        or l.digest != certificate.semantic_hash
        or _effects(l) != certificate.effects
        or tuple(name for name, _ in certificate.atomic_groups) != names
    ):
        raise ValueError("Schedule certificate does not match the semantic program")
    return True


def verify_elaboration(model, elaboration):
    """Recheck component contracts, expansion, provenance and certificate.

    This is a deterministic structural consistency check, not a compiler proof.
    The smaller verify_schedule check independently checks the executable order,
    effects and semantic hash without needing the component declarations.
    """
    if type(elaboration) is not Elaboration or elaborate(model) != elaboration:
        raise ValueError("Elaboration does not match its component declarations")
    return True


def elaborate(model):
    """Validate contracts and expand components; all ports have group-entry view.

    Binding order defines sampling/failure priority. Component captures follow
    in instance/declaration order. Transaction participants share one snapshot
    and commit: guards are tested left-to-right, stopping at the first disabled
    participant. Action bodies execute in declared order within that patch.
    """
    if type(model) is not Model:
        raise ValueError("Expected Model")
    identifier(model.name)
    for key in ("clocks", "inputs", "instances", "bindings", "transactions", "before"):
        _tuple(getattr(model, key), key)
    if not model.instances or not model.transactions:
        raise ValueError("A model requires components and explicit transactions")
    if any(type(i) is not Instance for i in model.instances):
        raise ValueError("Expected Instance declarations")
    instances = _unique(model.instances, "instance")
    global_clocks = _unique(model.clocks, "clock")
    _unique(model.inputs, "input")
    ports, outputs, actions, clock_maps = {}, {}, {}, {}
    component_ids = []
    for instance in model.instances:
        if "__" in instance.name:
            raise ValueError(
                "Instance names cannot contain the namespace separator '__'"
            )
        c = instance.component
        if type(c) is not Component or type(c.version) is not str or not c.version:
            raise ValueError("Expected a versioned Component declaration")
        identifier(c.name)
        for key in (
            "cells",
            "clocks",
            "inputs",
            "outputs",
            "actions",
            "captures",
            "tables",
            "timers",
        ):
            _tuple(getattr(c, key), f"component {key}")
        if len(set(c.clocks)) != len(c.clocks):
            raise ValueError("Duplicate component clock")
        for name in c.clocks:
            identifier(name)
        _tuple(instance.clocks, "clock bindings")
        if any(type(pair) is not tuple or len(pair) != 2 for pair in instance.clocks):
            raise ValueError("Clock bindings must be pairs")
        cm = dict(instance.clocks)
        if (
            len(cm) != len(instance.clocks)
            or set(cm) != set(c.clocks)
            or any(n not in global_clocks for n in cm.values())
        ):
            raise ValueError("Clock bindings must bind every formal clock exactly once")
        clock_maps[instance.name] = cm
        if any(type(p) is not Port for p in c.inputs) or any(
            type(p) is not Output for p in c.outputs
        ):
            raise ValueError("Expected typed Port and Output declarations")
        ip, out = _unique(c.inputs, "input port"), _unique(c.outputs, "output port")
        if set(ip) & set(out):
            raise ValueError("Input and output port names must be distinct")
        for p in (*c.inputs, *c.outputs):
            if p.clock not in cm:
                raise ValueError(f"Unknown port clock: {p.clock}")
        for p in c.inputs:
            dtype(p.type)
            ports[PortRef(instance.name, p.name)] = p
        for p in c.outputs:
            outputs[PortRef(instance.name, p.name)] = p
        if any(type(a) is not Region for a in c.actions):
            raise ValueError("Component actions must be semantic Regions")
        actions[instance.name] = _unique(c.actions, "action")
        # Component-local validation enforces exclusive ownership: a body can
        # only load/store declared local storage, never another instance's cells.
        dummy = Program(
            c.name,
            c.cells,
            tuple(Clock(n) for n in c.clocks),
            c.actions,
            tuple(Input(p.name, p.type) for p in c.inputs),
            c.captures,
            c.tables,
            timers=c.timers,
        )
        lower(dummy)
        for p in c.outputs:
            lower(replace(dummy, schedule=(), captures=(Let("output_probe", p.value),)))
        component_ids.append(
            (
                instance.name,
                c.name,
                c.version,
                hashlib.sha256(
                    json.dumps(asdict(c), sort_keys=True).encode()
                ).hexdigest(),
            )
        )

    bindings = {}
    for b in model.bindings:
        if (
            type(b) is not Binding
            or type(b.target) is not PortRef
            or b.target not in ports
        ):
            raise ValueError("Binding target must be an existing input port")
        if b.target in bindings:
            raise ValueError("Multiple drivers for an input port")
        if type(b.source) is PortRef:
            if b.source not in outputs:
                raise ValueError("Binding source must be an output port")
            out, port = outputs[b.source], ports[b.target]
            if out.value.type != port.type:
                raise ValueError("Connected port types differ")
            if (
                clock_maps[b.source.instance][out.clock]
                != clock_maps[b.target.instance][port.clock]
            ):
                raise ValueError(
                    "Connected port clocks differ; use an explicit crossing component"
                )
        elif type(b.source) is Expr:
            if b.source.type != ports[b.target].type:
                raise ValueError("Binding expression type differs from port type")
        else:
            raise ValueError("Bindings require closed expressions or output references")
        bindings[b.target] = b.source
    if set(bindings) != set(ports):
        raise ValueError("Every input port must have exactly one binding")

    def q(instance, name):
        value = instance + "__" + name
        identifier(value)
        return value

    def port_name(ref):
        return q("__port", q(ref.instance, ref.port))

    def external(e):
        if type(e) is not Expr:
            raise ValueError("Expected closed binding expression")
        if e.op in ("read", "lookup", "local"):
            raise ValueError("External bindings cannot access private component state")
        return replace(
            e, args=tuple(external(a) if type(a) is Expr else a for a in e.args)
        )

    resolving, expanded, dependencies = [], {}, set()

    def source(value):
        return output(value) if type(value) is PortRef else external(value)

    def output(ref):
        if ref not in outputs:
            raise ValueError("Unknown output reference")
        if ref in resolving:
            raise ValueError("Instantaneous connection cycle")
        if ref in expanded:
            return expanded[ref]
        resolving.append(ref)
        result = expression(outputs[ref].value, ref.instance, {}, sampled=False)
        resolving.pop()
        expanded[ref] = result
        return result

    def expression(e, instance, env, *, sampled=True):
        k, a = e.op, e.args
        if k == "input":
            ref = PortRef(instance, a[0])
            if sampled:
                return local(port_name(ref), e.type)
            if resolving:
                parent = resolving[-1]
                if ports[ref].clock != outputs[parent].clock:
                    raise ValueError(
                        "Combinational output crosses component clock domains"
                    )
                origin = bindings[ref]
                if type(origin) is PortRef:
                    dependencies.add(
                        (
                            f"{origin.instance}.{origin.port}",
                            f"{parent.instance}.{parent.port}",
                        )
                    )
            return source(bindings[ref])
        if k in ("read", "lookup"):
            return replace(
                e,
                args=(
                    q(instance, a[0]),
                    expression(a[1], instance, env, sampled=sampled),
                ),
            )
        if k == "due":
            return replace(e, args=(clock_maps[instance].get(a[0], q(instance, a[0])),))
        if k in ("generation", "due_generation"):
            return replace(e, args=(q(instance, a[0]),))
        if k == "local":
            return local(env[a[0]], e.type)
        return replace(
            e,
            args=tuple(
                expression(x, instance, env, sampled=sampled) if type(x) is Expr else x
                for x in a
            ),
        )

    # Check even unused combinational paths for ambiguous cyclic topology; only
    # bound input values are actually sampled/evaluated at runtime.
    for ref in outputs:
        output(ref)
    cells, tables, timers, captures, origins, capture_envs = [], [], [], [], [], {}
    for b in model.bindings:
        i = instances[b.target.instance]
        origins.append(
            Origin(
                f"capture/{len(captures)}",
                i.name,
                i.component.name,
                i.component.version,
                "port",
                b.target.port,
            )
        )
        captures.append(Let(port_name(b.target), source(b.source)))
    for i in model.instances:
        c = i.component
        cells.extend(replace(x, name=q(i.name, x.name)) for x in c.cells)
        tables.extend(replace(x, name=q(i.name, x.name)) for x in c.tables)
        timers.extend(replace(x, name=q(i.name, x.name)) for x in c.timers)
        env = {}
        for s in c.captures:
            value = expression(s.value, i.name, env)
            name = q("__capture", q(i.name, s.name))
            origins.append(
                Origin(
                    f"capture/{len(captures)}",
                    i.name,
                    c.name,
                    c.version,
                    "capture",
                    s.name,
                )
            )
            captures.append(Let(name, value))
            env[s.name] = name
        capture_envs[i.name] = env

    if any(type(t) is not Transaction for t in model.transactions):
        raise ValueError("Expected Transaction declarations")
    transactions = _unique(model.transactions, "transaction")
    order = _order(tuple(transactions), model.before)
    groups, schedule = [], []

    def statements(body, i, action, env, path, original="body"):
        result = []
        for j, s in enumerate(body):
            p, old = f"{path}/{j}", f"{original}/{j}"
            origins.append(
                Origin(p, i.name, i.component.name, i.component.version, action, old)
            )
            ex = lambda e: expression(e, i.name, env)
            if type(s) is Let:
                value = ex(s.value)
                name = q("__local", q(i.name, q(action, s.name)))
                result.append(Let(name, value))
                env[s.name] = name
            elif type(s) is Store:
                result.append(Store(q(i.name, s.target), ex(s.value), ex(s.index)))
            elif type(s) is Schedule:
                result.append(
                    Schedule(q(i.name, s.timer), ex(s.delay), ex(s.microstep))
                )
            elif type(s) is Cancel:
                result.append(Cancel(q(i.name, s.timer)))
            elif type(s) is Require:
                result.append(
                    Require(ex(s.condition), s.message, tuple(ex(x) for x in s.details))
                )
            elif type(s) is Observe:
                result.append(
                    Observe(q(i.name, s.name), tuple(ex(x) for x in s.values))
                )
            elif type(s) is If:
                result.append(
                    If(
                        ex(s.condition),
                        statements(
                            s.then, i, action, env.copy(), p + "/then", old + "/then"
                        ),
                        statements(
                            s.otherwise,
                            i,
                            action,
                            env.copy(),
                            p + "/else",
                            old + "/else",
                        ),
                    )
                )
            else:
                raise ValueError("Unsupported component statement")
        return tuple(result)

    for index, name in enumerate(order):
        transaction = transactions[name]
        if type(transaction) is not Transaction:
            raise ValueError("Expected Transaction")
        _tuple(transaction.actions, "transaction participants")
        if not transaction.actions or len(set(transaction.actions)) != len(
            transaction.actions
        ):
            raise ValueError("Transactions require distinct participant actions")
        guard, body, participants = const(1, BOOL), [], []
        path = f"schedule/{index}"
        if transaction.repeat is not None:
            path += "/region/0"
            origins.append(
                Origin(path, "", model.name, "1", name, f"transactions/{name}")
            )
        origins.append(
            Origin(
                f"schedule/{index}", "", model.name, "1", name, f"transactions/{name}"
            )
        )
        for j, ref in enumerate(transaction.actions):
            if (
                type(ref) is not ActionRef
                or ref.instance not in instances
                or ref.action not in actions[ref.instance]
            ):
                raise ValueError("Unknown participant action")
            i, action = instances[ref.instance], actions[ref.instance][ref.action]
            env = capture_envs[i.name].copy()
            guard = op(
                "select", guard, expression(action.guard, i.name, env), const(0, BOOL)
            )
            p = f"{path}/{j}"
            origins.append(
                Origin(
                    p,
                    i.name,
                    i.component.name,
                    i.component.version,
                    ref.action,
                    "action",
                )
            )
            body.append(
                If(
                    const(1, BOOL),
                    statements(action.body, i, ref.action, env, p + "/then"),
                )
            )
            participants.append(f"{ref.instance}.{ref.action}")
        groups.append((name, tuple(participants)))
        region = Region(name, tuple(body), guard)
        if transaction.repeat is not None:
            region = Repeat(
                name,
                source(transaction.repeat),
                transaction.limit,
                (replace(region, name=q(name, "body")),),
            )
        schedule.append(region)
    p = Program(
        model.name,
        tuple(cells),
        model.clocks,
        tuple(schedule),
        model.inputs,
        tuple(captures),
        tuple(tables),
        source(model.stop),
        tuple(timers),
    )
    l = lower(p)
    certificate = ScheduleCertificate(
        order,
        model.before,
        tuple(groups),
        tuple(component_ids),
        tuple(sorted(dependencies)),
        _effects(l),
        l.digest,
    )
    verify_schedule(p, certificate)
    return Elaboration(p, certificate, tuple(origins))
