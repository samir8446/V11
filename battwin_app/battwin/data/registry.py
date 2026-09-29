"""Per-cell test conditions transcribed from the NASA README files, and condition groups.

Grouping (edit GROUP_OF to override):
  Reference         B0005-07, B0018, B0036   24 °C, 2 A CC
  High current      B0033, B0034             24 °C, 4 A CC
  Hot               B0029-B0032              43 °C, 4 A CC
  Mixed conditions  B0038-B0040              1/2/4 A and 24/44 °C within each cell
  Cold              B0041-B0048, B0053-B0056 4 °C (41-44 mix 4 A and 1 A)
  Pulsed load       B0025-B0028              4 A, 0.05 Hz square wave, 50 % duty
  Corrupted logging B0049-B0052              control software crashed
Cells that stop at 1.6 Ah (B0033-B0040) cannot reach the 1.4 Ah EOL in the data.
"""
from __future__ import annotations

GROUPS = ["Reference", "High current", "Hot", "Mixed conditions", "Cold", "Pulsed load",
          "Corrupted logging", "Unknown"]

# id: (ambient °C or list, load A or list or "pulsed", cutoff V, stop capacity Ah or None)
_CELLS = {
    "B0005": (24, 2.0, 2.7, 1.4), "B0006": (24, 2.0, 2.5, 1.4), "B0007": (24, 2.0, 2.2, 1.4),
    "B0018": (24, 2.0, 2.5, 1.4),
    "B0025": (24, "pulsed", 2.0, None), "B0026": (24, "pulsed", 2.2, None),
    "B0027": (24, "pulsed", 2.5, None), "B0028": (24, "pulsed", 2.7, None),
    "B0029": (43, 4.0, 2.0, None), "B0030": (43, 4.0, 2.2, None),
    "B0031": (43, 4.0, 2.5, None), "B0032": (43, 4.0, 2.7, None),
    "B0033": (24, 4.0, 2.0, 1.6), "B0034": (24, 4.0, 2.2, 1.6), "B0036": (24, 2.0, 2.7, 1.6),
    "B0038": ([24, 44], [1.0, 2.0, 4.0], 2.2, 1.6), "B0039": ([24, 44], [1.0, 2.0, 4.0], 2.5, 1.6),
    "B0040": ([24, 44], [1.0, 2.0, 4.0], 2.7, 1.6),
    **{f"B00{n}": (4, [4.0, 1.0], c, 1.4) for n, c in zip(range(41, 45), (2.0, 2.2, 2.5, 2.7))},
    **{f"B00{n}": (4, 1.0, c, 1.4) for n, c in zip(range(45, 49), (2.0, 2.2, 2.5, 2.7))},
    **{f"B00{n}": (4, 2.0, c, None) for n, c in zip(range(49, 53), (2.0, 2.2, 2.5, 2.7))},
    **{f"B00{n}": (4, 2.0, c, 1.4) for n, c in zip(range(53, 57), (2.0, 2.2, 2.5, 2.7))},
}

GROUP_OF = {
    **{c: "Reference" for c in ["B0005", "B0006", "B0007", "B0018", "B0036"]},
    "B0033": "High current", "B0034": "High current",
    **{f"B00{n}": "Hot" for n in range(29, 33)},
    **{f"B00{n}": "Mixed conditions" for n in (38, 39, 40)},
    **{f"B00{n}": "Cold" for n in list(range(41, 49)) + list(range(53, 57))},
    **{f"B00{n}": "Pulsed load" for n in range(25, 29)},
    **{f"B00{n}": "Corrupted logging" for n in range(49, 53)},
}

ALL_CELLS = sorted(_CELLS)
_extra: dict[str, dict] = {}   # synthetic / user cells registered at runtime


def normalise_id(cid) -> str:
    s = str(cid).strip().upper()
    if s.startswith("B"):
        s = s[1:]
    try:
        return f"B{int(float(s)):04d}"
    except ValueError:
        return str(cid).strip()


def register(cid: str, group: str, ambient=24, load=2.0, cutoff=2.7, stop=None):
    _extra[cid] = dict(ambient=ambient, load=load, cutoff=cutoff, stop_ah=stop, group=group)


def info(cid: str) -> dict:
    if cid in _extra:
        return {"cell": cid, **_extra[cid]}
    if cid in _CELLS:
        a, l, c, s = _CELLS[cid]
        return dict(cell=cid, ambient=a, load=l, cutoff=c, stop_ah=s, group=GROUP_OF.get(cid, "Unknown"))
    return dict(cell=cid, ambient=None, load=None, cutoff=None, stop_ah=None, group="Unknown")


def group_of(cid: str) -> str:
    return info(cid)["group"]


def is_pulsed(cid: str) -> bool:
    return info(cid)["load"] == "pulsed"


def table(cells=None):
    import pandas as pd
    cells = cells if cells is not None else ALL_CELLS
    rows = []
    for c in cells:
        d = info(c)
        rows.append({**d, "ambient": str(d["ambient"]), "load": str(d["load"])})
    return pd.DataFrame(rows)
