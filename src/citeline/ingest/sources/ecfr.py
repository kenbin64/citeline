"""eCFR source client.

Fetches a part of the Code of Federal Regulations from the public eCFR API and
turns it into one document per regulation section.

Why this corpus: it is public domain, it is genuinely hard (dense cross
references, tables, defined terms), and it is the kind of text where a wrong
answer has a real consequence. That makes it a fair test of a system whose
selling point is refusing to guess.

API note: the /full endpoint returns HTTP 406 unless the request permits
compression, which is why Accept-Encoding is set explicitly.
"""

import hashlib
import re
from dataclasses import dataclass

import httpx
from lxml import etree
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

BASE = "https://www.ecfr.gov/api/versioner/v1"

_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class SourceDoc:
    """One regulation section, flattened to text."""

    source: str
    source_ref: str  # '40 CFR 141.62'
    title: str
    url: str
    version: str
    text: str

    @property
    def content_sha(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@retry(
    retry=retry_if_exception_type((httpx.HTTPError,)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def fetch_part_xml(title: int, part: int, version: str, timeout: float = 120.0) -> bytes:
    url = f"{BASE}/full/{version}/title-{title}.xml"
    headers = {
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "citeline/0.1 (+https://butterflyfx.us) public-data ingest",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url, params={"part": str(part)}, headers=headers)
        r.raise_for_status()
        return r.content


def _clean(text: str) -> str:
    text = _WS.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return _BLANK.sub("\n\n", "\n".join(lines)).strip()


def _node_text(node: etree._Element) -> str:
    """Flatten a section element to readable text, one block per paragraph."""
    blocks: list[str] = []
    for el in node.iter():
        if el.tag in {"P", "FP", "HD", "HED", "SUBJECT", "NOTE", "GID"}:
            t = "".join(el.itertext())
            if t and t.strip():
                blocks.append(t.strip())
    return _clean("\n\n".join(blocks))


def parse_sections(xml: bytes, title: int, part: int, version: str) -> list[SourceDoc]:
    """Extract one SourceDoc per DIV8 (section) element."""
    root = etree.fromstring(xml)
    docs: list[SourceDoc] = []
    seen: set[str] = set()

    for div in root.iter("DIV8"):
        if (div.get("TYPE") or "").upper() != "SECTION":
            continue
        num = (div.get("N") or "").strip()
        if not num:
            continue
        # 'N' looks like '§ 141.62' or '141.62'
        section = num.replace("§", "").strip()
        if not section:
            continue

        head_el = div.find("HEAD")
        head = "".join(head_el.itertext()).strip() if head_el is not None else ""
        head = _clean(head)
        # The HEAD repeats the section number; keep only the descriptive part.
        head_title = re.sub(r"^§?\s*[\d.\-a-z]+\s*", "", head).strip() or section

        text = _node_text(div)
        if not text:
            continue

        source_ref = f"{title} CFR {section}"
        if source_ref in seen:
            continue
        seen.add(source_ref)

        docs.append(
            SourceDoc(
                source="ecfr",
                source_ref=source_ref,
                title=head_title,
                url=f"https://www.ecfr.gov/current/title-{title}/part-{part}/section-{section}",
                version=version,
                # Prefix the heading so the retrieval unit carries its own subject.
                text=f"{source_ref} {head_title}\n\n{text}",
            )
        )
    return docs


def load(title: int, part: int, version: str) -> list[SourceDoc]:
    return parse_sections(fetch_part_xml(title, part, version), title, part, version)
