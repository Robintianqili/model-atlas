"""Coded errors (design §4.12). Every failure the SDK raises is an AtlasError with an AKxxx code.

Printed form (the same as the TS CLI, site/scripts/atlas/errors.ts):

    AK102  export: tensor "fc1.weight" is on cuda:0
      where   layer "fc1" (pipeline/demo/export.py:41)
      likely  the model was trained on the GPU and not moved back
      fix     call model.cpu() before export_model(...)
      docs    site/DEVKIT.md#ak102
"""
from __future__ import annotations

from typing import NoReturn, Optional

# Only listed codes exist (design §4.12 "Defined codes"). Keep in sync with site/scripts/atlas/errors.ts.
CODES: dict[str, str] = {
    "AK001": "Node or npm",
    "AK002": "Python or Python packages",
    "AK003": "port busy",
    "AK004": "git refuses the repo",
    "AK101": "graph differs between cases",
    "AK102": "CUDA or TF32 during export",
    "AK103": "describe() differs from module(x)",
    "AK104": "recurrent gate reference differs from the module",
    "AK105": "budget at export",
    "AK106": "op attributes differ from ops.json",
    "AK201": "layer over the parity bar (kind A)",
    "AK202": "explain() differs from the trace",
    "AK203": "twin mismatch (kind B)",
    "AK204": "reference mismatch (kind C, first divergence)",
    "AK205": "near-tie divergence (warning)",
    "AK301": "schema",
    "AK302": "inferShape differs from outShape at a dim extreme",
    "AK303": "size budget",
    "AK401": "placeholder unresolved or non-finite",
    "AK402": "literal multi-digit number in narration (warning)",
    "AK403": "en/zh placeholder sets differ",
    "AK404": "cursor not found on a fixture",
    "AK405": "view not registered",
    "AK406": "missing zh",
    "AK407": "lesson shape",
    "AK408": "claim test failed",
    "AK409": "reviewed lesson made only of auto steps",
    "AK410": "two different numbers print the same (warning)",
    "AK411": "step term in no glossary",
    "AK412": "predict narration gives the answer away (warning)",
    "AK501": "TODO markers in the manifest",
    "AK502": "step without provenance",
    "AK503": "dependency rule or Math.random",
    "AK504": "empty describe()",
    "AK601": "console error",
    "AK602": "horizontal scroll at 360 px",
    "AK603": "long task",
    "AK604": "accessibility violation (axe: serious or critical)",
    "AK701": "slug not in catalog",
    "AK702": "change outside owned paths",
    "AK703": "sync refused",
    "AK704": "credits missing",
}


class AtlasError(Exception):
    """An error with an AK code and the optional explanation lines of the CLI format."""

    def __init__(self, code: str, message: str, *, where: Optional[str] = None, likely: Optional[str] = None,
                 fix: Optional[str] = None, docs: Optional[str] = None):
        if code not in CODES:
            raise ValueError(f"unknown AK code {code!r}")
        self.code = code
        self.message = message
        self.where = where
        self.likely = likely
        self.fix = fix
        self.docs = docs if docs is not None else f"site/DEVKIT.md#{code.lower()}"
        super().__init__(self.format())

    def format(self) -> str:
        lines = [f"{self.code}  {self.message}"]
        for label, value in (("where", self.where), ("likely", self.likely), ("fix", self.fix), ("docs", self.docs)):
            if value:
                lines.append(f"  {label:<7} {value}")
        return "\n".join(lines)

    def __str__(self) -> str:  # tracebacks show the whole block
        return self.format()


def fail(code: str, message: str, **kw) -> NoReturn:
    """Raise an AtlasError."""
    raise AtlasError(code, message, **kw)
