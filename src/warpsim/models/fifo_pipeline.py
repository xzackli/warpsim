"""An arbitrary chain of FIFO instances, expressed only through public contracts."""

from ..core.ir import Bits, BOOL, TIME, Input, Clock, inp
from ..components import fifo
from ..modeling import (
    Model,
    Instance,
    Binding,
    PortRef,
    ActionRef,
    Transaction,
    elaborate,
)


def fifo_pipeline(capacities=(2, 3), *, period="1", value_type=Bits(16, True)):
    if type(capacities) is not tuple or not 2 <= len(capacities) <= 8:
        raise ValueError("Pipeline requires two to eight FIFO capacities")
    inputs = (
        Input("valid", BOOL),
        Input("value", value_type),
        Input("id", Bits(32)),
        Input("source_time", TIME),
        Input("ready", BOOL, 1),
    )
    instances = tuple(
        Instance(
            f"stage{i}", fifo(capacity=n, value_type=value_type), (("clock", "tick"),)
        )
        for i, n in enumerate(capacities)
    )
    bindings = []
    for i, instance in enumerate(instances):
        for name, external, typ in (
            ("valid", "valid", BOOL),
            ("value", "value", value_type),
            ("id", "id", Bits(32)),
            ("time", "source_time", TIME),
        ):
            source = (
                inp(external, typ)
                if i == 0
                else PortRef(instances[i - 1].name, "out_" + name)
            )
            bindings.append(Binding(PortRef(instance.name, "in_" + name), source))
        ready = (
            inp("ready", BOOL)
            if i == len(instances) - 1
            else PortRef(instances[i + 1].name, "in_ready")
        )
        bindings.append(Binding(PortRef(instance.name, "out_ready"), ready))
    model = Model(
        "fifo_pipeline",
        (Clock("tick", period),),
        inputs,
        instances,
        tuple(bindings),
        (Transaction("edge", tuple(ActionRef(i.name, "step") for i in instances)),),
    )
    return elaborate(model)
