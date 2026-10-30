"""Declaration builders that expand entirely into the closed core language."""

from dataclasses import dataclass, replace

from .ir import BOOL, Bits, Cell, If, Require, Store, const, op, read
from .lower import dtype, identifier, integer


@dataclass(frozen=True)
class Ring:
    """Finite FIFO with no fallthrough and full exchange of the old head.

    The owner supplies captured put/take decisions. This builder merges pointer
    and count writes, so a simultaneous exchange has one writer per scalar.
    """

    name: str
    capacity: int
    fields: tuple

    def __post_init__(self):
        identifier(self.name)
        integer(self.capacity, 1, 65536)
        if type(self.fields) is not tuple or not self.fields:
            raise ValueError("Ring requires a tuple of named fields")
        names = set()
        for name, typ in self.fields:
            identifier(name)
            dtype(typ)
            if name in names or name in ("head", "tail", "count"):
                raise ValueError("Duplicate/reserved ring field")
            names.add(name)

    @property
    def pointer_type(self):
        return Bits(max(1, (self.capacity - 1).bit_length()))

    @property
    def count_type(self):
        return Bits(self.capacity.bit_length())

    @property
    def cells(self):
        return (
            Cell(self.name + "_head", self.pointer_type),
            Cell(self.name + "_tail", self.pointer_type),
            Cell(self.name + "_count", self.count_type),
            *(
                Cell(self.name + "_" + name, typ, self.capacity)
                for name, typ in self.fields
            ),
        )

    @property
    def count(self):
        return read(self.name + "_count", self.count_type)

    def front(self, field):
        return read(
            self.name + "_" + field,
            dict(self.fields)[field],
            read(self.name + "_head", self.pointer_type),
        )

    def at(self, field, offset):
        """Read an old token at a checked logical offset from the current head."""
        from .ir import TIME

        index = op(
            "mod",
            op(
                "add",
                op("cast", read(self.name + "_head", self.pointer_type), type=TIME),
                op("cast", offset, type=TIME),
            ),
            const(self.capacity, TIME),
        )
        return read(self.name + "_" + field, dict(self.fields)[field], index)

    def initialized(self, tokens=()):
        """Explicit initial live-token records; backing storage remains initialized."""
        if len(tokens) > self.capacity or any(
            set(t) != dict(self.fields).keys() for t in tokens
        ):
            raise ValueError("Initial tokens must fit the ring and supply every field")
        values = {
            self.name + "_count": (len(tokens),),
            self.name + "_tail": (len(tokens) % self.capacity,),
        }
        for name, typ in self.fields:
            data = tuple(t[name] for t in tokens)
            for value in data:
                integer(value, typ.minimum, typ.maximum)
            values[self.name + "_" + name] = data + (0,) * (self.capacity - len(tokens))
        return tuple(
            replace(c, initial=values.get(c.name, c.initial)) for c in self.cells
        )

    def discard(self, amount):
        """Remove a bounded number of old tokens in the caller's transaction."""
        from .ir import TIME

        integer(amount, 1, self.capacity)
        head = op("cast", read(self.name + "_head", self.pointer_type), type=TIME)
        return (
            Require(
                op("ge", self.count, const(amount, self.count_type)),
                "ring discard underflow",
            ),
            Store(
                self.name + "_count",
                op("sub", self.count, const(amount, self.count_type)),
            ),
            Store(
                self.name + "_head",
                op(
                    "cast",
                    op("mod", op("add", head, const(amount)), const(self.capacity)),
                    type=self.pointer_type,
                ),
            ),
        )

    def can_take(self, requested):
        return op("and", requested, op("gt", self.count, const(0, self.count_type)))

    def can_put(self, requested, take):
        return op(
            "and",
            requested,
            op("or", op("lt", self.count, const(self.capacity, self.count_type)), take),
        )

    def update(self, put, take, payload):
        if set(payload) != {name for name, _ in self.fields}:
            raise ValueError("Ring payload must supply every field")
        if put.type != BOOL or take.type != BOOL:
            raise ValueError("Ring decisions must be BOOL")

        def advance(name):
            old = read(name, self.pointer_type)
            return Store(
                name,
                op(
                    "select",
                    op("eq", old, const(self.capacity - 1, self.pointer_type)),
                    const(0, self.pointer_type),
                    op("add", old, const(1, self.pointer_type)),
                ),
            )

        count = self.count
        next_count = op(
            "select",
            op("and", put, op("not", take)),
            op("add", count, const(1, self.count_type)),
            op(
                "select",
                op("and", take, op("not", put)),
                op("sub", count, const(1, self.count_type)),
                count,
            ),
        )
        return (
            Require(
                op("le", count, const(self.capacity, self.count_type)),
                "ring count exceeds capacity",
            ),
            Require(
                op(
                    "le",
                    read(self.name + "_head", self.pointer_type),
                    const(self.capacity - 1, self.pointer_type),
                ),
                "ring head out of range",
            ),
            Require(
                op(
                    "le",
                    read(self.name + "_tail", self.pointer_type),
                    const(self.capacity - 1, self.pointer_type),
                ),
                "ring tail out of range",
            ),
            Require(
                op("or", op("not", take), op("gt", count, const(0, self.count_type))),
                "ring take requires an old token",
            ),
            Require(
                op(
                    "or",
                    op("not", put),
                    op(
                        "or",
                        op("lt", count, const(self.capacity, self.count_type)),
                        take,
                    ),
                ),
                "ring put requires capacity or exchange",
            ),
            Store(self.name + "_count", next_count),
            If(take, (advance(self.name + "_head"),)),
            If(
                put,
                (
                    advance(self.name + "_tail"),
                    *(
                        Store(
                            self.name + "_" + field,
                            payload[field],
                            read(self.name + "_tail", self.pointer_type),
                        )
                        for field, _ in self.fields
                    ),
                ),
            ),
        )


def quantize(
    value, *, output, fractional_bits, rounding="nearest_even", overflow="saturate"
):
    """Explicit fixed-point rounding followed by a finite-width conversion."""
    modes = {
        "nearest_even": "round_even",
        "nearest_away": "round_away",
        "floor": "round_floor",
        "toward_zero": "round_zero",
    }
    if rounding not in modes or overflow not in ("checked", "wrap", "saturate"):
        raise ValueError("Unsupported quantizer policy")
    return op(
        "cast",
        op(modes[rounding], value, const(fractional_bits), type=value.type),
        type=output,
        policy=overflow,
    )


def fir(
    samples,
    coefficients,
    *,
    accumulator,
    output,
    fractional_bits,
    rounding="nearest_even",
    overflow="saturate",
):
    """Fully widened signed-coefficient products and explicitly quantized sum."""
    if not samples or len(samples) != len(coefficients):
        raise ValueError("FIR requires matching nonempty samples and coefficients")
    if any(not x.type.signed for x in coefficients):
        raise ValueError("This FIR builder requires signed coefficients")
    value = const(0, accumulator)
    for sample, coefficient in zip(samples, coefficients):
        product = op(
            "mul",
            sample,
            coefficient,
            type=Bits(sample.type.width + coefficient.type.width, True),
        )
        value = op(
            "add", value, op("cast", product, type=accumulator), type=accumulator
        )
    return quantize(
        value,
        output=output,
        fractional_bits=fractional_bits,
        rounding=rounding,
        overflow=overflow,
    )


def priority_select(requests, priorities, *, result_type=Bits(8, True)):
    """Stable minimum-priority arbiter; earlier declaration wins an exact tie.

    Losing payload actions remain in the caller's lazy branch. This returns a
    grant only and cannot manufacture rollback or an implicit fallback policy.
    """
    if not requests or len(requests) != len(priorities):
        raise ValueError("Arbiter needs matching requests and priorities")
    selected = const(-1, result_type)
    best = const(priorities[0].type.maximum, priorities[0].type)
    for i in reversed(range(len(requests))):
        win = op(
            "and",
            requests[i],
            op(
                "or",
                op("eq", selected, const(-1, result_type)),
                op("le", priorities[i], best),
            ),
        )
        selected = op("select", win, const(i, result_type), selected)
        best = op("select", win, priorities[i], best)
    return selected
