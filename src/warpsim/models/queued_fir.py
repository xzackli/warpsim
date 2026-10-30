"""Queued two-tap stage with held output and a lower-priority repair port.

This executable fragment follows the supplied research example. All behavior is
declarations/primitives; the compiler and runtime know nothing about this model.
"""

from ..core import (Bits, BOOL, TIME, Cell, Input, Clock, Program, Region,
                    Let, Store, Require, Observe, If, const, read, inp, local, due, op)
from ..core.primitives import Ring, fir


SAMPLE, COEFF, ID, ADDRESS = Bits(10, True), Bits(12, True), Bits(32), Bits(2)


def queued_fir(*, capacity=2, history_size=2, producer_period="1", consumer_period="1", coefficients=(1024, 1024)):
    # Include at least one invalid raw address to exercise selected port faults.
    address_type = Bits(max(1, history_size.bit_length()))
    q = Ring("q", capacity, (("value", SAMPLE), ("id", ID), ("time", TIME), ("address", address_type)))
    history = (Cell("history", SAMPLE, history_size), Cell("history_valid", BOOL, history_size, (1,) * history_size),
               Cell("history_id", ID, history_size))
    output = (Cell("out_valid", BOOL), Cell("out_value", SAMPLE), Cell("out_id", ID),
              Cell("out_time", TIME), Cell("out_history_id", ID))
    repair = (Cell("repair_pending", BOOL), Cell("repair_address", address_type), Cell("repair_value", SAMPLE), Cell("repair_id", ID))
    inputs = (Input("valid", BOOL), Input("value", SAMPLE), Input("id", ID), Input("source_time", TIME),
              Input("address", address_type), Input("ready", BOOL, 1))
    out_valid, ready = read("out_valid", BOOL), inp("ready", BOOL)
    take = local("take", BOOL)
    put = local("put", BOOL)
    retire = local("retire", BOOL)
    grant = local("repair_granted", BOOL)
    captures = (
        Let("retire", op("and", due("consumer"), op("and", out_valid, ready))),
        Let("take", q.can_take(op("and", due("consumer"), op("or", op("not", out_valid), ready)))),
        Let("put", q.can_put(op("and", due("producer"), inp("valid", BOOL)), take)),
        Let("repair_granted", op("and", read("repair_pending", BOOL), op("not", take))),
    )
    address = local("address", address_type)
    value, uid, time = local("value", SAMPLE), local("uid", ID), local("time", TIME)
    old_history = read("history", SAMPLE, address)
    filtered = fir((value, old_history), tuple(const(c, COEFF) for c in coefficients),
                   accumulator=Bits(24, True), output=SAMPLE, fractional_bits=10)
    process = (
        Let("address", q.front("address")), Let("value", q.front("value")),
        Let("uid", q.front("id")), Let("time", q.front("time")),
        Require(read("history_valid", BOOL, address), "history sample must be valid"),
        Let("filtered", filtered),
        Store("history", value, address), Store("history_id", uid, address),
        Store("out_valid", const(1, BOOL)), Store("out_value", local("filtered", SAMPLE)),
        Store("out_id", uid), Store("out_time", time), Store("out_history_id", read("history_id", ID, address)),
        Observe("produced", (uid, local("filtered", SAMPLE), time, read("history_id", ID, address))),
    )
    edge = Region("edge", (
        If(retire, (Observe("consumed", (read("out_id", ID), read("out_value", SAMPLE), read("out_time", TIME), read("out_history_id", ID))),)),
        If(take, process, (If(retire, (Store("out_valid", const(0, BOOL)),)),)),
        *q.update(put, take, {"value": inp("value", SAMPLE), "id": inp("id", ID),
                              "time": inp("source_time", TIME), "address": inp("address", address_type)}),
        If(put, (Observe("accepted", (inp("id", ID), inp("value", SAMPLE), inp("source_time", TIME))),)),
    ))
    repair_region = Region("repair", (
        Store("history", read("repair_value", SAMPLE), read("repair_address", address_type)),
        Store("history_valid", const(1, BOOL), read("repair_address", address_type)),
        Store("history_id", read("repair_id", ID), read("repair_address", address_type)),
        Store("repair_pending", const(0, BOOL)),
        Observe("repaired", (read("repair_address", address_type), read("repair_id", ID))),
    ), grant)
    return Program("queued_fir", (*q.cells, *history, *output, *repair),
                   (Clock("producer", producer_period), Clock("consumer", consumer_period)),
                   (edge, repair_region), inputs, captures)
