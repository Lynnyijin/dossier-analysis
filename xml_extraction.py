from __future__ import annotations

import signal
import sys
from pathlib import Path

from pagexml.parser import parse_pagexml_file


def extract_page_text(pagexml_path: Path) -> str:
    scan = parse_pagexml_file(str(pagexml_path))
    regions = scan.get_text_regions_in_reading_order()
    lines: list[str] = []
    for region in regions:
        for line in region.lines:
            if getattr(line, "text", None):
                lines.append(line.text)
    return "\n".join(lines).strip()


def main() -> None:
    # Avoid BrokenPipeError tracebacks when piping output (e.g. `| head`).
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    root = Path("NAMA_digitised_page_files")
    xml_paths = sorted(root.rglob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No .xml files found under: {root.resolve()}")

    out_root = Path("outputs/page_text_by_page")

    try:
        out_root.mkdir(parents=True, exist_ok=True)
        for xml_path in xml_paths:
            page_text = extract_page_text(xml_path)

            rel_xml_path = xml_path.relative_to(root)
            out_path = (out_root / rel_xml_path).with_suffix(".txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(page_text + ("\n" if page_text else ""), encoding="utf-8")

            print(f"WROTE {out_path.as_posix()}")
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)


if __name__ == "__main__":
    main()