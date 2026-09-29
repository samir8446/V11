"""Export study results as a CSV bundle (zip) and a self-contained HTML report."""
from __future__ import annotations
import html
import io
import zipfile
import pandas as pd
from .. import __version__


def csv_zip(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(f"{name}.csv", df.to_csv(index=False))
    return buf.getvalue()


def html_report(title: str, answers: dict[str, list[str]], tables: dict[str, pd.DataFrame],
                figures_html: dict[str, str] | None = None, meta: dict | None = None) -> str:
    css = ("body{font-family:Inter,Segoe UI,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#1b2430}"
           "h1{border-bottom:3px solid #00a3c4}table{border-collapse:collapse;font-size:12px;margin:8px 0 24px}"
           "td,th{border:1px solid #d0d7de;padding:3px 8px;text-align:right}th{background:#eef2f5}"
           "li{margin:4px 0}@media(prefers-color-scheme:dark){body{background:#0e1117;color:#e6edf3}"
           "th{background:#1f2630}td,th{border-color:#30363d}}")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{css}</style></head><body><h1>{html.escape(title)}</h1>"
             f"<p>Engine {__version__}. {html.escape(str(meta or ''))}</p>"]
    for mission, lines in answers.items():
        parts.append(f"<h2>{html.escape(mission)}</h2><ul>")
        parts += [f"<li>{html.escape(l).replace('**', '')}</li>" for l in lines]
        parts.append("</ul>")
    for name, fig in (figures_html or {}).items():
        parts.append(f"<h3>{html.escape(name)}</h3>{fig}")
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and len(df):
            parts.append(f"<h3>{html.escape(name)}</h3>" + df.to_html(index=False, float_format=lambda x: f"{x:.4g}"))
    parts.append("</body></html>")
    return "".join(parts)
