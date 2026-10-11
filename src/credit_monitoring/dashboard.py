"""Standalone HTML presentation of aggregate monitoring evidence."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import urlsplit


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _table(data: dict) -> str:
    rows = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        rows.append(f"<tr><th scope='row'>{_escape(key)}</th><td>{_escape(value)}</td></tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def _number(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _grid(headers: list[str], rows: list[list], classes: list[str] | None = None) -> str:
    heading = "".join(f"<th scope='col'>{_escape(label)}</th>" for label in headers)
    body = []
    for index, row in enumerate(rows):
        style = classes[index] if classes else ""
        cells = "".join(f"<td>{_escape(_number(cell))}</td>" for cell in row)
        body.append(f"<tr class='{style}'>{cells}</tr>")
    return (
        "<div class='scroll'><table class='grid'><thead><tr>"
        + heading
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _severity(run: dict) -> str:
    if str(run.get("status", "")).upper() in {"BLOCKED", "FAILED", "REJECTED"}:
        return "blocked"
    return "alert" if run.get("alerts") else "healthy"


def _drift_view(drift: dict) -> str:
    features = drift.get("features", [])
    metadata = {key: value for key, value in drift.items() if key != "features"}
    view = _table(metadata)
    if features:
        view += _grid(
            ["Feature", "PSI", "KS", "p-valor (Holm)", "Drift"],
            [
                [
                    item.get("feature", "—"),
                    item.get("psi", "—"),
                    item.get("ks_statistic", "—"),
                    item.get("ks_pvalue_holm", "—"),
                    "Sim" if item.get("drift_detected") else "Não",
                ]
                for item in features
            ],
            ["alert" if item.get("drift_detected") else "" for item in features],
        )
    return view


def _overview(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        metrics = run.get("metrics", {})
        drifted = [
            item.get("feature", "")
            for item in run.get("drift", {}).get("features", [])
            if item.get("drift_detected")
        ]
        rows.append(
            [
                run.get("scenario", "—"),
                run.get("status", "—"),
                run.get("rows", "—"),
                metrics.get("recall", "—"),
                metrics.get("pr_auc", "—"),
                metrics.get("brier", metrics.get("brier_score", "—")),
                ", ".join(drifted) or "—",
                len(run.get("alerts", [])),
            ]
        )
    return _grid(
        [
            "Cenário",
            "Status",
            "Linhas",
            "Recall",
            "PR-AUC",
            "Brier",
            "Features em drift",
            "Alertas",
        ],
        rows,
        [_severity(run) for run in runs],
    )


def _artifact_links(artifacts: dict, output_path: Path) -> str:
    links = []
    for name, value in artifacts.items():
        if not isinstance(value, (str, Path)):
            continue
        raw = str(value).replace("\\", "/")
        if urlsplit(raw).scheme and not Path(value).is_absolute():
            continue
        if raw.startswith("//"):
            continue
        path = Path(value)
        if path.is_absolute() or path.exists():
            import os

            raw = os.path.relpath(path.resolve(), output_path.parent.resolve()).replace("\\", "/")
        links.append(f"<li><a href='{_escape(raw)}'>{_escape(name)}</a></li>")
    return "<ul>" + "".join(links) + "</ul>" if links else "<p>Sem artefatos.</p>"


def render_dashboard(summary: dict, output_path: Path) -> None:
    """Render pre-aggregated data; no CDN, raw client records or network calls."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for run in summary.get("runs", []):
        status = str(run.get("status", "desconhecido"))
        alerts = run.get("alerts", [])
        alert_html = (
            "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in alerts) + "</ul>"
            if alerts
            else "<p>Sem alertas registrados.</p>"
        )
        sections.append(
            f"<article class='{_severity(run)}'>"
            f"<h2>{_escape(run.get('scenario', 'cenário'))} "
            f"<span class='status'>{_escape(status)}</span></h2>"
            + _table(
                {
                    "Volume (linhas)": run.get("rows", 0),
                    "Duração (segundos)": run.get("duration_seconds", "indisponível"),
                    "MLflow run ID": run.get("mlflow_run_id", "indisponível"),
                }
            )
            + "<h3>Alertas</h3>"
            + alert_html
            + "<h3>Modelo: métricas com rótulos</h3>"
            + (
                _table(run.get("metrics", {}))
                if run.get("metrics")
                else "<p>Indisponíveis: lote sem avaliação.</p>"
            )
            + "<h3>Infraestrutura medida</h3>"
            + _table(run.get("resources", {}))
            + "<h3>Drift</h3>"
            + _drift_view(run.get("drift", {}))
            + "<h3>Evidências</h3>"
            + _artifact_links(run.get("artifacts", {}), output_path)
            + "</article>"
        )
    content = """<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Monitoramento de crédito — Tech Challenge 4</title>
<link rel='icon' href='data:,'>
<style>
body{font-family:system-ui,sans-serif;max-width:1120px;margin:2rem auto;padding:0 1rem;
background:#f3f6fa;color:#152335;line-height:1.5}
article,section{background:white;border:1px solid #cad5e2;border-radius:12px;
padding:1.3rem;margin:1rem 0}
h1{font-size:2rem}h2{font-size:1.4rem}h3{font-size:1.1rem;margin-bottom:.4rem}
table{width:100%;border-collapse:collapse;table-layout:fixed}
th,td{text-align:left;vertical-align:top;border-bottom:1px solid #e0e6ee;
padding:.5rem;overflow-wrap:anywhere}th{width:36%}
.status{font-size:.9rem;border:1px solid #61738b;border-radius:5px;
padding:.2rem .5rem;display:inline-block}a{color:#124e9e}li{overflow-wrap:anywhere}
.blocked{background:#fff0f0;border-color:#b52121}.alert{background:#fff7e5;border-color:#93630d}
.healthy .status{background:#e6f7ec;color:#185b2e}.grid{table-layout:auto;min-width:750px}
.grid th{width:auto}.scroll{overflow-x:auto}.grid td,.grid th{font-size:.9rem}
@media(max-width:600px){body{margin:.5rem auto}article,section{padding:.8rem}}
</style></head><body><header><h1>Monitoramento de crédito</h1>
<p>Execução acadêmica com cenários simulados. Relatório agregado;
não representa decisões reais de crédito.</p></header>
"""
    content += "<section><h2>Comparação de cenários</h2>"
    content += "<p>Vermelho: bloqueio/falha. Amarelo: alerta. —: indisponível.</p>"
    content += _overview(summary.get("runs", [])) + "</section>"
    baseline = summary.get("baseline", {})
    content += "<section><h2>Baseline congelado</h2>"
    content += _table({key: value for key, value in baseline.items() if key != "metrics"})
    if baseline.get("metrics"):
        content += "<h3>Métricas de validação</h3>" + _table(baseline["metrics"])
    content += "</section>"
    content += "".join(sections)
    content += (
        "<section><details><summary>Proveniência e contagens</summary>"
        + _table(summary.get("provenance", {}))
        + "</details></section>"
    )
    content += (
        "<footer><p>RSS e CPU são amostrados durante a execução; picos entre amostras "
        "podem não aparecer. CPU do processo pode ultrapassar 100% em múltiplos núcleos. "
        "Espaço livre é o menor valor observado no disco local.</p></footer></body></html>"
    )
    output_path.write_text(content, encoding="utf-8")
