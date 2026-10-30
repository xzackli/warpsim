"""Conservative, application-independent recovery of regular candidate maps.

Only execution instructions are inspected. Each expression is proved total
over declared finite types, all addresses are constant/in bounds, destinations
are distinct, and no input/capture/local from another statement is referenced.
An affine family becomes one shared candidate calculation. This is an alternate
lowering of existing statements, not a new semantic operation.
"""

from dataclasses import dataclass

from . import execution_ir as x


def _safe(statement):
    if not isinstance(statement, x.Statement):
        return None
    ranges, names, literals, shape = {}, {}, [], []
    write = None

    def value(v):
        if v.kind == "literal":
            literals.append(v.data)
            return ("literal", v.type)
        if v.kind != "temporary" or v.data not in names:
            raise ValueError("Nonlocal or effectful operand")
        return ("temporary", names[v.data], v.type)

    def interval(v):
        return (v.data, v.data) if v.kind == "literal" else ranges[v.data]

    def result(v, bounds):
        names[v.data] = len(names)
        ranges[v.data] = bounds
        return (names[v.data], v.type)

    try:
        for i in statement.body:
            if isinstance(i, x.CheckIndex):
                if i.index.kind != "literal" or not 0 <= i.index.data < i.size:
                    return None
                shape.append(("index", i.offset, i.size, value(i.index)))
            elif isinstance(i, x.CheckedLoad):
                if i.index.kind != "literal" or not 0 <= i.index.data < i.size:
                    return None
                index = value(i.index)
                shape.append(
                    (
                        "load",
                        i.space,
                        i.offset,
                        i.size,
                        index,
                        result(
                            i.result, (i.result.type.minimum, i.result.type.maximum)
                        ),
                    )
                )
            elif isinstance(i, x.Compute):
                operands = tuple(value(v) for v in i.operands)
                bounds = [interval(v) for v in i.operands]
                if i.op == "cast":
                    lo, hi = bounds[0]
                elif i.op == "add":
                    lo, hi = bounds[0][0] + bounds[1][0], bounds[0][1] + bounds[1][1]
                elif i.op == "sub":
                    lo, hi = bounds[0][0] - bounds[1][1], bounds[0][1] - bounds[1][0]
                elif i.op == "mul":
                    products = [a * b for a in bounds[0] for b in bounds[1]]
                    lo, hi = min(products), max(products)
                elif i.op in ("eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"):
                    lo, hi = 0, 1
                else:
                    return None
                if lo < -(1 << 63) or hi >= 1 << 63:
                    return None
                typ = i.result.type
                if i.policy == "checked" and not (
                    typ.minimum <= lo <= hi <= typ.maximum
                ):
                    return None
                if lo < typ.minimum or hi > typ.maximum:
                    lo, hi = typ.minimum, typ.maximum
                shape.append(
                    ("compute", i.op, i.policy, operands, result(i.result, (lo, hi)))
                )
            elif isinstance(i, x.CandidateWrite):
                if (
                    write is not None
                    or i is not statement.body[-1]
                    or i.index.kind != "literal"
                ):
                    return None
                write = i
                shape.append(("write", i.offset, value(i.index), value(i.value)))
            else:
                return None
    except (ValueError, KeyError):
        return None
    if write is None:
        return None
    return tuple(shape), tuple(literals), write


@dataclass(frozen=True)
class CandidateMap:
    statements: tuple
    literals: tuple
    strides: tuple
    offset: int
    first_index: int
    index_stride: int
    first_location: int
    location_stride: int


def discover(execution, minimum=32):
    maps = []

    def scan(block):
        cursor = 0
        while cursor < len(block):
            first = _safe(block[cursor])
            group = None
            if first and cursor + 1 < len(block):
                second = _safe(block[cursor + 1])
                if second and first[0] == second[0]:
                    strides = tuple(b - a for a, b in zip(first[1], second[1]))
                    index_stride = second[2].index.data - first[2].index.data
                    location_stride = (
                        block[cursor + 1].origin.location
                        - block[cursor].origin.location
                    )
                    end = cursor + 2
                    while end < len(block):
                        item = _safe(block[end])
                        j = end - cursor
                        if (
                            not item
                            or item[0] != first[0]
                            or item[1]
                            != tuple(a + j * d for a, d in zip(first[1], strides))
                            or block[end].origin.location
                            != block[cursor].origin.location + j * location_stride
                        ):
                            break
                        end += 1
                    if end - cursor >= minimum and index_stride != 0:
                        group = CandidateMap(
                            tuple(block[cursor:end]),
                            first[1],
                            strides,
                            first[2].offset,
                            first[2].index.data,
                            index_stride,
                            block[cursor].origin.location,
                            location_stride,
                        )
            if group:
                maps.append(group)
                cursor += len(group.statements)
                continue
            item = block[cursor]
            if isinstance(item, x.Region):
                scan(item.guard)
                scan(item.body)
            elif isinstance(item, x.Repeat):
                scan(item.condition_code)
                scan(item.body)
            elif isinstance(item, x.Branch):
                scan(item.then)
                scan(item.otherwise)
            elif isinstance(item, x.Statement):
                scan(item.body)
            cursor += 1

    scan(execution.captures)
    scan(execution.schedule)
    scan(execution.stop_code)
    return tuple(maps)


def worker_source(group):
    """Emit one affine template; proof above permits omitting dynamic checks."""
    literals = iter(zip(group.literals, group.strides))
    names, lines = {}, []

    def value(v):
        if v.kind == "literal":
            a, d = next(literals)
            return f"({a}LL+{d}LL*item)" if d else f"({a}LL)"
        return names[v.data]

    def result(v):
        names[v.data] = f"c{len(names)}"
        return names[v.data]

    for i in group.statements[0].body:
        if isinstance(i, x.CheckIndex):
            value(i.index)
        elif isinstance(i, x.CheckedLoad):
            index = value(i.index)
            expr = (
                f"states[(DATA+{i.offset}+{index})*n+s]"
                if i.space == "state"
                else f"scales[(1LL+{i.offset}+{index})*n+s]"
            )
            lines.append(f"i64 {result(i.result)}={expr};")
        elif isinstance(i, x.Compute):
            values = [value(v) for v in i.operands]
            name = result(i.result)
            operations = dict(
                add="+",
                sub="-",
                mul="*",
                eq="==",
                ne="!=",
                lt="<",
                le="<=",
                gt=">",
                ge=">=",
                **{"and": "&&", "or": "||"},
            )
            raw = (
                values[0]
                if i.op == "cast"
                else f"!{values[0]}"
                if i.op == "not"
                else f"{values[0]} {operations[i.op]} {values[1]}"
            )
            lines.append(f"i64 {name}=({raw});")
            typ = i.result.type
            if i.policy == "saturate":
                lines.append(
                    f"{name}={name}<{typ.minimum}LL?{typ.minimum}LL:({name}>{typ.maximum}LL?{typ.maximum}LL:{name});"
                )
            elif i.policy == "wrap":
                lines.append(f"u64 bits_{name}=u64({name})&((1ULL<<{typ.width})-1);")
                expr = (
                    f"bits_{name}>u64({typ.maximum}LL)?-i64((1ULL<<{typ.width})-bits_{name}):i64(bits_{name})"
                    if typ.signed
                    else f"i64(bits_{name})"
                )
                lines.append(f"{name}={expr};")
        elif isinstance(i, x.CandidateWrite):
            value(i.index)
            lines.append(f"scratch[item]={value(i.value)};")
    return "\n".join(lines)
