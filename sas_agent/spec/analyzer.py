"""SAS analyzer: writes the facts layer of each macro's spec from the SAS text. No model is involved.

It recognises the SAS this project needs, not the whole language:
- comments: /* ... */ and statement comments (*...; and %*...;);
- macros: %macro name(params); ... %mend;
- steps: DATA ... RUN; and PROC ... RUN;/QUIT; (an ODS OUTPUT statement before a PROC belongs to it);
- datasets read (SET, MERGE, DATA=, SEED=, SQL FROM) and written (DATA, OUT=, CREATE TABLE, ODS OUTPUT,
  PROC SORT without OUT=);
- macro variables used (&x) and created (%LET, INTO:, CALL SYMPUTX);
- call sites in the programs, with %LET values resolved in program order.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from .schema import Arg, CallSite, Facts, SideEffects, Step, Unit


@dataclass
class Statement:
    text: str        # without the trailing semicolon
    line: int        # line of its first character
    end_line: int    # line of its semicolon


@dataclass
class RawStep:
    kind: str
    statements: list[Statement] = field(default_factory=list)
    pending_ods: list[Statement] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------------
# Text to statements


def blank_comments(text: str) -> str:
    """Replace /* ... */ comments with spaces, keeping newlines so line numbers stay correct."""
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group()), text, flags=re.DOTALL)


def statements(text: str, first_line: int = 1) -> list[Statement]:
    """Split SAS text into statements at semicolons outside quotes; drop statement comments."""
    text = blank_comments(text)
    out, start, quote, line = [], 0, None, first_line
    start_line = line
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == ";":
            raw = text[start:i]
            stripped = raw.strip()
            if stripped and not stripped.startswith(("*", "%*")):
                lead = raw[: len(raw) - len(raw.lstrip())]
                out.append(Statement(stripped, start_line + lead.count("\n"), line))
            start, start_line = i + 1, line
        if ch == "\n":
            line += 1
    return out


def strip_options(text: str) -> str:
    """Remove dataset options in parentheses, e.g. 'dat0 (keep=a b) dat1' -> 'dat0  dat1'."""
    out, depth = [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


_NAME = r"[A-Za-z_&][\w&.]*"


def dataset_names(text: str) -> list[str]:
    """Dataset names in a list like 'a (keep=x) b c nobs=n', skipping option=value tokens."""
    return [t for t in strip_options(text).split() if "=" not in t and re.fullmatch(_NAME, t)]


# ---------------------------------------------------------------------------------------------------
# Statements to steps


def split_steps(stmts: list[Statement]) -> list[RawStep]:
    steps, current, pending_ods = [], None, []
    for st in stmts:
        low = st.text.lower()
        if re.match(r"data\b(?!\s*=)", low) or re.match(r"proc\s+\w+", low):
            if current:
                steps.append(current)
            kind = "DATA" if low.startswith("data") else "PROC " + re.match(r"proc\s+(\w+)", low).group(1).upper()
            current = RawStep(kind, [st])
            if kind.startswith("PROC") and pending_ods:
                current.pending_ods, pending_ods = pending_ods, []
        elif re.match(r"ods\s+output\b", low) and current is None:
            pending_ods.append(st)
        elif current:
            current.statements.append(st)
            if re.fullmatch(r"(run|quit)\b.*", low, flags=re.DOTALL):
                steps.append(current)
                current = None
    if current:
        steps.append(current)
    return steps


def _ods_writes(st: Statement) -> list[str]:
    return [m.group(1) for m in re.finditer(r"\w+\s*=\s*(" + _NAME + r")", strip_options(st.text[len("ods output"):]))]


def step_reads_writes(step: RawStep) -> tuple[list[str], list[str]]:
    reads, writes = [], []
    head = step.statements[0].text
    if step.kind == "DATA":
        writes += [n for n in dataset_names(re.sub(r"(?i)^data\b", "", head)) if n.lower() != "_null_"]
        for st in step.statements[1:]:
            m = re.search(r"(?i)(?:^|\bthen\s+)(set|merge)\s+(.*)", st.text, flags=re.DOTALL)
            if m:
                reads += dataset_names(m.group(2))
    else:
        options = strip_options(head)
        data = re.search(r"(?i)\bdata\s*=\s*(" + _NAME + ")", options)
        out = re.search(r"(?i)\bout\s*=\s*(" + _NAME + ")", options)
        seed = re.search(r"(?i)\bseed\s*=\s*(" + _NAME + ")", options)
        reads += [m.group(1) for m in (data, seed) if m]
        if out:
            writes.append(out.group(1))
        elif step.kind == "PROC SORT" and data:
            writes.append(data.group(1))          # sorts in place
        for st in step.pending_ods + step.statements[1:]:
            low = st.text.lower()
            if low.startswith("ods output"):
                writes += _ods_writes(st)
            elif low.startswith("output"):
                m = re.search(r"(?i)\bout\s*=\s*(" + _NAME + ")", strip_options(st.text))
                if m:
                    writes.append(m.group(1))
            elif step.kind == "PROC SQL":
                m = re.search(r"(?i)create\s+table\s+(" + _NAME + r")\s+as", st.text)
                if m:
                    writes.append(m.group(1))
                f = re.search(r"(?is)\bfrom\s+(.*?)(?:\bwhere\b|\bgroup\b|\border\b|\bhaving\b|$)", st.text)
                if f:
                    reads += [t.split()[0] for t in f.group(1).split(",") if t.split()]
    return _unique(reads), _unique(writes)


def _unique(names: list[str]) -> list[str]:
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


# ---------------------------------------------------------------------------------------------------
# Constructs (fixed vocabulary)

_CONSTRUCT_PATTERNS = [
    ("array", r"\barray\b"),
    ("do_loop", r"\bdo\s+\w+\s*="),
    ("sum_of", r"\bsum\s*\(\s*of\b"),
    ("retain", r"\bretain\b"),
    ("first_last", r"\b(first|last)\.\w"),
    ("set_n_once", r"\bif\s+_n_\s*=\s*1\s+then\s+set\b"),
    ("select_into", r"\binto\s*:"),
    ("call_symputx", r"\bcall\s+symputx?\b"),
    ("macro_if", r"%if\b"),
]


def step_constructs(step: RawStep) -> list[str]:
    text = "\n".join(st.text for st in step.pending_ods + step.statements).lower()
    found = [(text.find(m.group()), name) for name, pat in _CONSTRUCT_PATTERNS if (m := re.search(pat, text))]
    if step.kind.startswith("PROC"):
        proc = "proc_" + step.kind[5:].lower()
        has_ods = bool(step.pending_ods) or "ods output" in text
        found.append((-1, proc + "_ods" if proc == "proc_means" and has_ods else proc))
        if step.kind == "PROC SQL" and re.search(r"(?is)\bfrom\s+[^;]*?,", text):
            found.append((len(text), "sql_cross_join"))
    if step.kind == "DATA" and re.search(r"\bmerge\b", text) and re.search(r"(^|\n)\s*by\b|;\s*by\b", text):
        found.append((len(text) + 1, "merge_by"))
    return [name for _, name in sorted(found)]


# ---------------------------------------------------------------------------------------------------
# Macros


@dataclass
class MacroDef:
    name: str
    params: list[str]
    keyword: bool
    lines: tuple[int, int]
    body: list[Statement]
    source_text: str


def find_macros(path: Path) -> dict[str, MacroDef]:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    stmts = statements(text)
    macros, i = {}, 0
    while i < len(stmts):
        m = re.match(r"(?is)%macro\s+(\w+)\s*(?:\((.*)\))?\s*$", stmts[i].text)
        if not m:
            i += 1
            continue
        j = next(k for k in range(i + 1, len(stmts)) if re.match(r"(?i)%mend\b", stmts[k].text))
        raw_params = [p.strip() for p in (m.group(2) or "").split(",") if p.strip()]
        params = [p.split("=")[0].strip() for p in raw_params]
        start, end = stmts[i].line, stmts[j].end_line
        macros[m.group(1)] = MacroDef(
            m.group(1), params, any("=" in p for p in raw_params), (start, end), stmts[i + 1 : j],
            "\n".join(lines[start - 1 : end]),
        )
        i = j + 1
    return macros


def _normalize(name: str, params: list[str]) -> str:
    """'&indsn.' -> the declared parameter name 'indsn'; other names are kept as written."""
    m = re.fullmatch(r"&(\w+)\.?", name)
    if m:
        for p in params:
            if p.lower() == m.group(1).lower():
                return p
    return name.rstrip(".")


def _refs(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"&(\w+)", text)]


# ---------------------------------------------------------------------------------------------------
# Programs: %LET resolution, call sites, and later reads


def resolve(text: str, lets: dict[str, str]) -> str:
    for _ in range(10):
        new = re.sub(r"&(\w+)\.?", lambda m: lets.get(m.group(1).lower(), m.group(0)), text)
        if new == text:
            break
        text = new
    return text


def split_args(text: str) -> list[str]:
    args, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    args.append("".join(cur).strip())
    return [a for a in args if a]


@dataclass
class ProgramScan:
    call_sites: dict[str, list[CallSite]]           # macro name -> call sites
    step_events: list[tuple[int, int, str, str]]    # (program index, line, "read"/"write", dataset name)


def scan_programs(macros: dict[str, MacroDef]) -> ProgramScan:
    lets: dict[str, str] = {}
    sites: dict[str, list[CallSite]] = {name: [] for name in macros}
    events = []
    for index, program in enumerate(config.PROGRAMS):
        stmts = statements(program.path.read_text(errors="replace"))
        for st in stmts:
            let = re.match(r"(?is)%let\s+(\w+)\s*=\s*(.*)$", st.text)
            if let:
                lets[let.group(1).lower()] = resolve(let.group(2).strip(), lets)
                continue
            call = re.match(r"(?is)%(\w+)\s*\((.*)\)\s*$", st.text)
            if call and call.group(1) in macros:
                mac = macros[call.group(1)]
                args = []
                for pos, raw in enumerate(split_args(call.group(2))):
                    if mac.keyword and "=" in raw:
                        param, value = raw.split("=", 1)
                    else:
                        param, value = (mac.params[pos] if pos < len(mac.params) else f"arg{pos + 1}"), raw
                    args.append(Arg(param=param.strip(), value=resolve(value.strip(), lets)))
                sites[mac.name].append(CallSite(caller=f"{program.id}:L{st.line}", program_id=program.id,
                                                line=st.line, args=args))
        for step in split_steps(stmts):
            reads, writes = step_reads_writes(step)
            line = step.statements[0].line
            events += [(index, line, "read", resolve(n, lets)) for n in reads]
            events += [(index, line, "write", resolve(n, lets)) for n in writes]
    return ProgramScan(sites, sorted(events))


def _read_downstream(names: list[str], sites: list[CallSite], events) -> list[str]:
    """Names a program step reads after a call site, before any program step writes them again."""
    order = {p.id: i for i, p in enumerate(config.PROGRAMS)}
    found = []
    for name in names:
        for site in sites:
            for prog, line, kind, ds in events:
                if (prog, line) <= (order[site.program_id], site.line) or ds.lower() != name.lower():
                    continue
                if kind == "read":
                    found.append(name)
                break
    return _unique(found)


# ---------------------------------------------------------------------------------------------------


def analyze_macros(macro_file: Path | None = None) -> dict[str, tuple[Unit, Facts]]:
    """The unit and facts layer of every macro, keyed by unit id ("M.grp_score")."""
    macro_file = macro_file or config.MACROS_FILE
    macros = find_macros(macro_file)
    scan = scan_programs(macros)
    result = {}
    for name, mac in macros.items():
        raw_steps = split_steps(mac.body)
        steps, constructs = [], []
        for n, raw in enumerate(raw_steps, 1):
            reads, writes = step_reads_writes(raw)
            first = (raw.pending_ods or raw.statements)[0].line
            steps.append(Step(id=f"s{n}", kind=raw.kind, lines=(first, raw.statements[-1].end_line),
                              reads=[_normalize(r, mac.params) for r in reads],
                              writes=[_normalize(w, mac.params) for w in writes]))
            constructs += step_constructs(raw)
        body_text = "\n".join(st.text for st in mac.body)
        referenced = {r.lower() for r in _refs(body_text)}
        param_set = {p.lower() for p in mac.params}
        written = _unique([w for s in steps for w in s.writes])
        output_params = [p for p in mac.params if p.lower() in {w.lower() for w in written}]
        work_left = [w for w in written if w.lower() not in param_set and "." not in w and "&" not in w]
        created = [m.group(1) for m in re.finditer(r"(?i)%let\s+(\w+)", body_text)]
        created += [m.group(1) for m in re.finditer(r"(?i)\binto\s*:\s*(\w+)", body_text)]
        created += [m.group(1) for m in re.finditer(r"(?i)call\s+symputx?\s*\(\s*'(\w+)'", body_text)]
        unit = Unit(id=f"M.{name}", source=macro_file.name, lines=mac.lines,
                    source_hash=hashlib.sha256(mac.source_text.encode()).hexdigest())
        facts = Facts(
            params=mac.params,
            unused_params=[p for p in mac.params if p.lower() not in referenced],
            output_params=output_params,
            steps=steps,
            constructs=_unique(constructs),
            macro_vars_used=_unique([r for r in _refs(body_text) if r.lower() not in param_set]),
            macro_vars_created=_unique(created),
            side_effects=SideEffects(work_datasets_left=work_left,
                                     read_downstream=_read_downstream(work_left, scan.call_sites[name], scan.step_events)),
            call_sites=scan.call_sites[name],
        )
        result[unit.id] = (unit, facts)
    return result


def unit_source(unit: Unit) -> str:
    """The SAS text of a unit, for the prompt."""
    lines = (config.SAS_PROGRAMS_DIR / unit.source).read_text(errors="replace").splitlines()
    return "\n".join(lines[unit.lines[0] - 1 : unit.lines[1]])
