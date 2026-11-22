"""Parse SAS_CSV_Mapping.txt into the section for one program."""


def mapping_heading(program_id: str) -> str:
    return f"Program {program_id.removeprefix('program')}:"


def parse_mapping(text: str, heading: str) -> str:
    """Return the mapping block that starts with `heading` (e.g. 'Program 0:')."""
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(heading):
            start = index
            break
    if start is None:
        raise ValueError(f"No mapping section starting with {heading!r}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("Program ") and index != start:
            end = index
            break
    block = "\n".join(lines[start:end]).strip()
    if not block:
        raise ValueError(f"Empty mapping section for {heading!r}")
    return block
