"""Specialize the validated core to one body shared by C++ and CUDA entry points.

Region writes use a bounded sparse journal. Loads see physical entry storage
until validation succeeds; arrays are never copied to implement snapshots.
Full state copies below are explicit diagnostic trace output only.
"""

from . import execution_ir as x
from .execution import data_offset, trace_layout
from .lower import MAX, lower
from time import perf_counter


class Emitter:
    def __init__(self, l, trace_cells=None, trace_mode="full", *, strategy="scalar"):
        self.l, self.p = l, l.program
        start = perf_counter()
        self.execution = x.lower_execution(l)
        self.lowering_seconds = perf_counter() - start
        self.lines = []
        self.source_map = {}
        _, self.trace_indices = trace_layout(l, trace_cells)
        if trace_mode not in ("full", "observations", "summary"):
            raise ValueError("Unknown trace mode")
        self.trace_mode = trace_mode
        if strategy not in ("scalar", "warp", "block"):
            raise ValueError("Unknown execution strategy")
        self.strategy = strategy
        from .cooperative import discover

        self.maps = discover(self.execution) if strategy != "scalar" else ()
        self._maps = {id(m.statements[0]): (j, m) for j, m in enumerate(self.maps)}
        self.cooperative_regions = tuple(
            dict(
                path=m.statements[0].origin.path,
                count=len(m.statements),
                source_paths=tuple(s.origin.path for s in m.statements),
                nonfaulting=True,
            )
            for m in self.maps
        )

    def emit(self, line):
        self.lines.extend(line.split("\n"))

    @staticmethod
    def value(value):
        if value.kind == "literal":
            return f"({value.data}LL)"
        if value.kind == "temporary":
            return f"v{value.data}"
        if value.kind == "time":
            return "time"
        if value.kind == "microstep":
            return "microstep"
        if value.kind == "duration":
            return f"({value.data}LL*scale)"
        if value.kind == "generation":
            return f"generations[{value.data}]"
        if value.kind == "due_generation":
            return f"due_generations[{value.data}]"
        if value.kind == "input":
            return f"inputs[{value.data}]"
        if value.kind == "due":
            return f"due[{value.data}]"
        raise ValueError(f"Unknown execution operand: {value.kind}")

    def record(self, kind, location, obs="0", vals="zero"):
        if self.trace_mode == "summary" and kind != 4:
            return
        if self.trace_mode == "observations" and kind not in (3, 4):
            return
        self.emit(
            f"record(states,traces,n,s,records++,time,microstep,iteration,{kind},active,{location},fault,target,bad_index,{obs},{vals});"
        )

    def block(self, instructions):
        cursor = 0
        while cursor < len(instructions):
            instruction = instructions[cursor]
            group = self._maps.get(id(instruction))
            if group:
                self.candidate_map(*group)
                cursor += len(group[1].statements)
            else:
                self.instruction(instruction)
                cursor += 1

    def candidate_map(self, number, group):
        for statement in group.statements:
            self.source_map.setdefault(statement.origin.path, []).append(
                len(self.lines) + 1
            )
        self.emit("#ifdef __CUDACC__")
        self.emit(f"if(!fault) {{ cooperative->command={number}; cooperative_sync();")
        self.emit(
            "cooperative_values(states,scales,n,s,cooperative->command,scratch); cooperative_sync();"
        )
        self.emit(
            f"int before_map=nw; for(int item=0;item<{len(group.statements)} && !fault;++item) {{"
        )
        self.emit(
            f"int index={group.first_index}+{group.index_stride}*item,address={group.offset}+index;"
        )
        self.emit(f"location={group.first_location}+{group.location_stride}*item;")
        self.emit(
            f"for(int j=0;j<before_map;++j) if(address==addresses[j]) {{fault=3;target={group.offset};bad_index=index;break;}}"
        )
        self.emit("if(!fault) {addresses[nw]=address;values[nw++]=scratch[item];} } }")
        self.emit("#else")
        for statement in group.statements:
            self.instruction(statement)
        self.emit("#endif")

    def cooperative_helpers(self):
        from .cooperative import worker_source

        self.emit("#ifdef __CUDACC__")
        self.emit("""
#if __CUDA_ARCH__ < 700
#error Cooperative execution requires independent thread scheduling (sm_70+)
#endif
struct CooperativeContext { int command; };
// Non-aligned PTX barrier: coordinator and workers reach different instructions.
// Every block thread participates once in each publication/completion phase.
INLINE void cooperative_sync() { asm volatile("barrier.sync 0;" ::: "memory"); }
INLINE void cooperative_values(const i64* states,const i64* scales,int n,int s,int command,i64* scratch) {
    switch(command) {
""")
        for j, group in enumerate(self.maps):
            self.emit(
                f"case {j}: for(int item=threadIdx.x;item<{len(group.statements)};item+=blockDim.x) {{"
            )
            self.emit(worker_source(group))
            self.emit("} break;")
        self.emit("} }\n#endif")

    def instruction(self, instruction):
        i = instruction
        self.source_map.setdefault(i.origin.path, []).append(len(self.lines) + 1)
        if isinstance(i, x.Declare):
            self.emit(f"i64 {self.value(i.result)}=0;")
        elif isinstance(i, x.Assign):
            self.emit(f"{self.value(i.result)}={self.value(i.value)};")
        elif isinstance(i, x.CheckedLoad):
            result, index = self.value(i.result), self.value(i.index)
            self.emit(f"i64 {result}=0;")
            if i.space == "state":
                target, load = i.offset, f"S(DATA+{i.offset}+{index})"
            else:
                target, load = -2 - i.offset, f"scales[(1LL+{i.offset}+{index})*n+s]"
            self.emit(
                f"if(!fault) {{ if({index}<0 || {index}>={i.size}) {{fault=1;target={target};bad_index={index};}} else {result}={load}; }}"
            )
        elif isinstance(i, x.Compute):
            result = self.value(i.result)
            values, k = [self.value(v) for v in i.operands], i.op
            self.emit(f"i64 {result}=0;")
            if k in (
                "cast",
                "round_even",
                "round_away",
                "round_floor",
                "round_zero",
                "add",
                "sub",
                "mul",
                "div",
                "mod",
            ):
                if k == "cast":
                    raw = values[0]
                elif k.startswith("round_"):
                    mode = {
                        "round_even": 0,
                        "round_away": 1,
                        "round_floor": 2,
                        "round_zero": 3,
                    }[k]
                    raw = f"rounded({values[0]}, {values[1]}, {mode})"
                else:
                    if k in ("div", "mod"):
                        self.emit(f"if(!fault && {values[1]}==0) fault=6;")
                    raw = f"({values[0]} {dict(add='+', sub='-', mul='*', div='/', mod='%')[k]} {values[1]})"
                self.conversion(result, raw, i.result.type, i.policy)
            else:
                operation = {
                    "eq": "==",
                    "ne": "!=",
                    "lt": "<",
                    "le": "<=",
                    "gt": ">",
                    "ge": ">=",
                    "and": "&&",
                    "or": "||",
                }
                raw = (
                    f"!{values[0]}"
                    if k == "not"
                    else f"{values[0]} {operation[k]} {values[1]}"
                )
                self.emit(f"if(!fault) {result}=({raw});")
        elif isinstance(i, x.CheckIndex):
            index = self.value(i.index)
            self.emit(
                f"if(!fault && ({index}<0 || {index}>={i.size})) {{ fault=1;target={i.offset};bad_index={index}; }}"
            )
        elif isinstance(i, x.CandidateWrite):
            index, value = self.value(i.index), self.value(i.value)
            self.emit(
                f"if(!fault) {{ int address={i.offset}+int({index}); for(int j=0;j<nw;++j) if(address==addresses[j]) {{ fault=3;target={i.offset};bad_index={index};break; }} if(!fault) {{addresses[nw]=address;values[nw++]={value};}} }}"
            )
        elif isinstance(i, x.CandidateObservation):
            values = [self.value(v) for v in i.values]
            self.emit(
                f"if(!fault) {{ observations[no].id={i.observation};observations[no].location=location;"
            )
            for j in range(8):
                self.emit(
                    f"observations[no].values[{j}]={values[j] if j < len(values) else 0};"
                )
            self.emit("++no; }")
        elif isinstance(i, x.CandidateEvent):
            j, target = i.timer, -2 - self.l.table_width - i.timer
            self.emit("if(!fault) {")
            if i.delay is not None:
                delay, step = self.value(i.delay), self.value(i.microstep)
                self.emit(
                    f"if({delay}>MAX_TIME-time) fault=2; else if({delay}==0 && {step}<=microstep) fault=7; else if({step}>{self.p.max_microstep}) fault=8;"
                )
            self.emit(
                f"if(!fault && timer_written[{j}]) fault=3; if(!fault && generations[{j}]==MAX_TIME) fault=2;"
            )
            self.emit(
                f"if(fault) {{target={target};bad_index=0;}} else {{ timer_written[{j}]=true;"
            )
            self.emit(
                f"timer_values[{j}]="
                + (
                    f"time+{delay};timer_steps[{j}]={step};"
                    if i.delay is not None
                    else f"INF;timer_steps[{j}]=0;"
                )
            )
            self.emit("} }")
        elif isinstance(i, x.FailRequirement):
            self.emit("if(!fault) {")
            for j, value in enumerate(i.details):
                self.emit(f"diagnostic[{j}]={self.value(value)};")
            self.emit("fault=4; }")
        elif isinstance(i, x.Branch):
            self.emit(f"if(!fault) {{ if({self.value(i.condition)}) {{")
            self.block(i.then)
            self.emit("} else {")
            self.block(i.otherwise)
            self.emit("} }")
        elif isinstance(i, x.Statement):
            # Bound CUDA debug compilation units without changing evaluation
            # order or captures. Release inlines these scopes: keeping their
            # boundaries there increased local-memory use and execution time.
            self.emit(
                f"if(!fault) {{ [&]() BLOCK_ATTRIBUTE {{ location={i.origin.location};"
            )
            self.block(i.body)
            self.emit("}(); }")
        elif isinstance(i, x.BeginJournal):
            self.emit("int nw=0,no=0;")
            self.emit(
                f"for(int j=0;j<{len(self.p.timers)};++j) timer_written[j]=false;"
            )
        elif isinstance(i, x.Commit):
            self.emit(
                "if(!fault) { for(int j=0;j<nw;++j) S(DATA+addresses[j])=values[j];"
            )
            self.emit(
                f"for(int j=0;j<{len(self.p.timers)};++j) if(timer_written[j]) {{timer_times[j]=timer_values[j];timer_microsteps[j]=timer_steps[j];++generations[j];}}"
            )
            self.block(i.outcomes)
            self.emit("}")
        elif isinstance(i, x.Outcome):
            if i.kind == "observations":
                self.emit("for(int j=0;j<no;++j) {")
                self.record(
                    3,
                    "observations[j].location",
                    "observations[j].id",
                    "observations[j].values",
                )
                self.emit("}")
            else:
                self.record({"commit": 1, "disabled": 2}[i.kind], i.origin.location)
        elif isinstance(i, x.Region):
            self.emit(f"if(!fault) {{ active={i.region};location={i.origin.location};")
            self.block(i.guard)
            self.emit(f"if(!fault) {{ if({self.value(i.enabled)}) {{")
            self.block(i.body)
            self.emit("} else {")
            self.instruction(i.disabled)
            self.emit("} } }")
        elif isinstance(i, x.Repeat):
            self.emit(
                f"for(int repeat_index=0;repeat_index<={i.limit} && !fault;++repeat_index) {{ iteration=repeat_index;active=0;location={i.origin.location};"
            )
            self.block(i.condition_code)
            self.emit(
                f"if(fault || !({self.value(i.condition)})) break; if(iteration=={i.limit}) {{fault=5;break;}}"
            )
            self.block(i.body)
            self.emit("}")
        else:
            raise TypeError(f"Unsupported execution instruction: {type(i)}")

    def conversion(self, result, raw, type, policy):
        # Width and policy are execution-IR constants. Emit their selected
        # operation directly: debug builds otherwise clone all branches and
        # parameter/debug-variable machinery of a forced-inline generic helper.
        lo, hi = f"{type.minimum}LL", f"{type.maximum}LL"
        self.emit(f"if(!fault) {{ {result}={raw};")
        if policy == "checked":
            self.emit(f"if({result}<{lo} || {result}>{hi}) {{fault=2;{result}=0;}}")
        elif policy == "saturate":
            self.emit(f"{result}={result}<{lo}?{lo}:({result}>{hi}?{hi}:{result});")
        elif policy == "wrap":
            self.emit(
                f"u64 modulus=1ULL<<{type.width}, bits=u64({result})&(modulus-1);"
            )
            value = (
                f"bits>u64({hi})?-i64(modulus-bits):i64(bits)"
                if type.signed
                else "i64(bits)"
            )
            self.emit(f"{result}={value};")
        else:
            raise ValueError(f"Unknown conversion policy: {policy}")
        self.emit("}")

    def generate(self):
        nc, ni, nt = len(self.p.clocks), len(self.p.inputs), len(self.p.timers)
        data = data_offset(self.p)
        # Large fully inlined functions make NVRTC's optimizer grow sharply.
        # Partition by compiler IR size, independent of model names/topology.
        partition_release = int(self.execution.temporary_count > 2048)
        spans = []
        for destination, source in enumerate(self.trace_indices):
            if spans and spans[-1][0] + spans[-1][2] == source:
                start, dest, size = spans[-1]
                spans[-1] = (start, dest, size + 1)
            else:
                spans.append((source, destination, 1))
        trace_copy = "\n".join(
            f"for(int j=0;j<{size};++j) traces[base+(18+{dest}+j)*n]=S(DATA+{start}+j);"
            for start, dest, size in spans
        )
        self.emit(f"""// warpsim.core/v2 {self.l.digest}
#define COOPERATIVE_EXECUTION {int(self.strategy != "scalar")}
typedef long long i64;
typedef unsigned long long u64;
static_assert(sizeof(i64)==8,"64-bit carrier required");
#ifdef __CUDACC__
#define INLINE __device__ __forceinline__
#if defined(_DEBUG) || {partition_release}
#define BLOCK_ATTRIBUTE __attribute__((noinline))
#else
#define BLOCK_ATTRIBUTE __attribute__((always_inline))
#endif
#else
#define INLINE inline
#define BLOCK_ATTRIBUTE
#endif
#define S(k) states[(i64(k))*n+s]
constexpr i64 MAX_TIME={MAX}LL, INF=9223372036854775807LL;
constexpr int DATA={data}, CELLS={self.l.width}, WIDTH=DATA+CELLS, RECORD=18+{len(self.trace_indices)};
struct Observation {{ int id,location; i64 values[8]; }};
INLINE i64 rounded(i64 value,int shift,int mode) {{
    i64 divisor=1LL<<shift, q=value/divisor, r=value%divisor;
    if(mode==3) return q;
    if(mode==1) {{
        i64 magnitude=value<0?-value:value, result=magnitude/divisor, remainder=magnitude%divisor;
        result+=i64(remainder>=divisor-remainder);
        return value<0?-result:result;
    }}
    if(r<0) {{ --q;r+=divisor; }}
    if(mode==2) return q;
    return q + i64(r>divisor-r || (r==divisor-r && q%2!=0));
}}
INLINE void record(i64* states,i64* traces,int n,int s,int row,i64 time,i64 microstep,int iteration,int kind,int region,int location,int code,int target,i64 index,int obs,const i64* values) {{
    i64 base=i64(row)*RECORD*n+s;
    traces[base]=time;traces[base+n]=kind;traces[base+2*n]=region;traces[base+3*n]=location;
    traces[base+4*n]=code;traces[base+5*n]=target;traces[base+6*n]=index;traces[base+7*n]=obs;
    for(int j=0;j<8;++j) traces[base+(8+j)*n]=values[j];
    traces[base+16*n]=microstep;traces[base+17*n]=iteration;
    {trace_copy}
}}
""")
        if self.strategy != "scalar":
            self.cooperative_helpers()
        self.emit(f"""
INLINE void simulate(i64* states,const i64* external,const i64* counts,const i64* scales,int n,int s,int external_size,i64 until,int budget,i64* traces,int trace_limit
#if defined(__CUDACC__) && COOPERATIVE_EXECUTION
,CooperativeContext* cooperative,i64* scratch
#endif
) {{
    i64 next[{max(1, nc)}], inputs[{max(1, ni)}];
    bool due[{max(1, nc + nt)}];
    i64 timer_times[{max(1, nt)}],timer_microsteps[{max(1, nt)}],generations[{max(1, nt)}],due_generations[{max(1, nt)}];
    i64 timer_values[{max(1, nt)}],timer_steps[{max(1, nt)}];bool timer_written[{max(1, nt)}];
    int addresses[{self.l.journal_size}];i64 values[{self.l.journal_size}];
    Observation observations[{self.l.observation_size}];
    i64 zero[8]={{}}, diagnostic[8]={{}}, time=S(10)?S(1):-1, microstep=S(9), bad_index=-1;
    int status=0,processed=0,cursor=int(S(8)),records=0,fault=0,target=-1,location=0,active=0,iteration=-1;
    i64 events=S(2);
    if(S(0)==3 || S(0)==4) {{{"" if self.trace_mode == "summary" else "S(3)=0;"}return;}}
    i64 scale=scales[s];
""")
        for j, phase in enumerate(self.l.phases):
            self.emit(f"next[{j}]=S(10)?S({12 + j}):{phase}LL*scale;")
        for j, (timer, initial) in enumerate(zip(self.p.timers, self.l.timer_initials)):
            start = "INF" if initial is None else f"{initial}LL*scale"
            self.emit(
                f"timer_times[{j}]=S(10)?S({12 + nc + 3 * j}):{start};timer_microsteps[{j}]=S(10)?S({13 + nc + 3 * j}):{timer.microstep};generations[{j}]=S(10)?S({14 + nc + 3 * j}):0;"
            )
        for j, item in enumerate(self.p.inputs):
            self.emit(f"inputs[{j}]=S(10)?S({12 + nc + 3 * nt + j}):{item.initial}LL;")
        self.emit("while(true) { i64 candidate=INF,candidate_step=0;")
        for j in range(nc):
            self.emit(
                f"if(next[{j}]<candidate) {{candidate=next[{j}];candidate_step=0;}}"
            )
        for j in range(nt):
            self.emit(
                f"if(timer_times[{j}]<candidate || (timer_times[{j}]==candidate && timer_microsteps[{j}]<candidate_step)) {{candidate=timer_times[{j}];candidate_step=timer_microsteps[{j}];}}"
            )
        self.emit(
            f"if(cursor<counts[s] && external[i64(cursor)*{ni + 1}*n+s]<=candidate) {{candidate=external[i64(cursor)*{ni + 1}*n+s];candidate_step=0;}}"
        )
        self.emit(
            "if(candidate==INF || candidate>until) {status=1;break;} if(processed==budget) {status=2;break;} time=candidate;microstep=candidate_step;++events;++processed;"
        )
        for j, period in enumerate(self.l.periods):
            self.emit(
                f"due[{j}]=next[{j}]==time && microstep==0; if(due[{j}]) next[{j}]=time>MAX_TIME-{period}LL*scale?INF:time+{period}LL*scale;"
            )
        for j in range(nt):
            self.emit(
                f"due[{nc + j}]=timer_times[{j}]==time && timer_microsteps[{j}]==microstep;due_generations[{j}]=generations[{j}];if(due[{nc + j}]) {{timer_times[{j}]=INF;timer_microsteps[{j}]=0;}}"
            )
        self.emit(
            f"if(microstep==0 && cursor<counts[s] && external[i64(cursor)*{ni + 1}*n+s]==time) {{"
        )
        for j in range(ni):
            self.emit(f"inputs[{j}]=external[(i64(cursor)*{ni + 1}+{j + 1})*n+s];")
        self.emit("++cursor; }")
        self.emit("active=0;iteration=-1;target=-1;bad_index=-1;")
        self.block(self.execution.captures)
        for instruction in self.execution.schedule:
            self.emit("if(!fault) iteration=-1;")
            self.instruction(instruction)
        self.emit("if(fault) {")
        self.record(4, "location", vals="diagnostic")
        self.emit("status=3;break; }")
        self.emit(f"active=0;iteration=-1;location={self.l.locations['stop'][0]};")
        self.block(self.execution.stop_code)
        stopped = self.value(self.execution.stop)
        self.emit("if(fault) {")
        self.record(4, "location", vals="diagnostic")
        self.emit(f"status=3;break; }} if({stopped}) {{status=4;break;}} }}")
        self.emit(
            "S(0)=status;S(1)=time;S(2)=events;S(3)=records;S(4)=fault;S(5)=location;S(6)=target;S(7)=bad_index;S(8)=cursor;S(9)=microstep;S(10)=1;"
        )
        for j in range(nc):
            self.emit(f"S({12 + j})=next[{j}];")
        for j in range(nt):
            self.emit(
                f"S({12 + nc + 3 * j})=timer_times[{j}];S({13 + nc + 3 * j})=timer_microsteps[{j}];S({14 + nc + 3 * j})=generations[{j}];"
            )
        for j in range(ni):
            self.emit(f"S({12 + nc + 3 * nt + j})=inputs[{j}];")
        self.emit("}")
        scratch_size = max((len(m.statements) for m in self.maps), default=1)
        if self.strategy != "scalar":
            self.emit(f"""
#ifdef __CUDACC__
INLINE void cooperative_scenario(i64* states,const i64* external,const i64* counts,const i64* scales,int n,int s,int external_size,i64 until,int budget,i64* traces,int trace_limit) {{
    __shared__ CooperativeContext cooperative;
    __shared__ i64 scratch[{scratch_size}];
    if(threadIdx.x==0) {{
        simulate(states,external,counts,scales,n,s,external_size,until,budget,traces,trace_limit,&cooperative,scratch);
        cooperative.command=-1; cooperative_sync();
    }} else {{
        while(true) {{
            cooperative_sync();
            int command=cooperative.command;
            if(command<0) break;
            cooperative_values(states,scales,n,s,command,scratch);
            cooperative_sync();
        }}
    }}
}}
#endif
""")
        self.emit("""
#ifdef __CUDACC__
extern "C" __global__ void initialize(i64* states,const i64* initial,int n) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;if(s<n) for(int j=0;j<WIDTH;++j) S(j)=initial[i64(j)*n+s];
}
extern "C" __global__ void advance(i64* states,const i64* external,const i64* counts,const i64* scales,int n,int external_size,i64 until,int budget,i64* traces,int trace_limit) {
    #if COOPERATIVE_EXECUTION
    int s=blockIdx.x;if(s<n) cooperative_scenario(states,external,counts,scales,n,s,external_size,until,budget,traces,trace_limit);
    #else
    int s=blockIdx.x*blockDim.x+threadIdx.x;if(s<n) simulate(states,external,counts,scales,n,s,external_size,until,budget,traces,trace_limit);
    #endif
}
extern "C" __global__ void advance_indexed(i64* states,const i64* external,const i64* counts,const i64* scales,int n,int external_size,i64 until,int budget,i64* traces,int trace_limit,const int* ids,int active_count) {
    #if COOPERATIVE_EXECUTION
    int slot=blockIdx.x;if(slot<active_count) cooperative_scenario(states,external,counts,scales,n,ids[slot],external_size,until,budget,traces,trace_limit);
    #else
    int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if(slot<active_count) simulate(states,external,counts,scales,n,ids[slot],external_size,until,budget,traces,trace_limit);
    #endif
}
extern "C" __global__ void compact_active(const i64* states,int n,int* ids,int* count) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s<n && states[s]==2) ids[atomicAdd(count,1)]=s;
}
#else
extern "C" void initialize(i64* states,const i64* initial,int n) {
    for(int s=0;s<n;++s) for(int j=0;j<WIDTH;++j) S(j)=initial[i64(j)*n+s];
}
extern "C" void advance(i64* states,const i64* external,const i64* counts,const i64* scales,int n,int external_size,i64 until,int budget,i64* traces,int trace_limit) {
    for(int s=0;s<n;++s) simulate(states,external,counts,scales,n,s,external_size,until,budget,traces,trace_limit);
}
extern "C" void advance_parallel(i64* states,const i64* external,const i64* counts,const i64* scales,int n,int external_size,i64 until,int budget,i64* traces,int trace_limit,const int* ids,int active_count,int threads,int dynamic,int chunk) {
    if(dynamic) {
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic,chunk)
        #endif
        for(int slot=0;slot<active_count;++slot) simulate(states,external,counts,scales,n,ids?ids[slot]:slot,external_size,until,budget,traces,trace_limit);
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(static)
        #endif
        for(int slot=0;slot<active_count;++slot) simulate(states,external,counts,scales,n,ids?ids[slot]:slot,external_size,until,budget,traces,trace_limit);
    }
}
extern "C" void compact_active(const i64* states,int n,int* ids,int* count) {
    *count=0;for(int s=0;s<n;++s) if(states[s]==2) ids[(*count)++]=s;
}
#endif
""")
        return "\n".join(self.lines)


def generate(program, *, trace_cells=None, trace_mode="full"):
    return Emitter(lower(program), trace_cells, trace_mode).generate()
