from __future__ import annotations

import json
import sys

from .advisor import build_daily_report
from .models import DailyInput, MacroData, PortfolioState, Position, TechnicalData


def _parse_input(raw: dict) -> DailyInput:
    portfolio_raw = raw["portfolio"]
    positions = [Position(**p) for p in portfolio_raw.get("positions", [])]
    portfolio = PortfolioState(
        portfoy_degeri=portfolio_raw["portfoy_degeri"],
        gunluk_getiri_yuzde=portfolio_raw["gunluk_getiri_yuzde"],
        yillik_getiri_yuzde=portfolio_raw["yillik_getiri_yuzde"],
        borsa_yuzde=portfolio_raw["borsa_yuzde"],
        altin_yuzde=portfolio_raw["altin_yuzde"],
        doviz_yuzde=portfolio_raw["doviz_yuzde"],
        likit_repo_yuzde=portfolio_raw["likit_repo_yuzde"],
        positions=positions,
    )
    return DailyInput(
        tarih=raw["tarih"],
        macro=MacroData(**raw["macro"]),
        technical=TechnicalData(**raw["technical"]),
        portfolio=portfolio,
        sentiment_notlari=raw.get("sentiment_notlari", []),
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("Kullanım: python -m yatirim.cli <girdi.json>", file=sys.stderr)
        return 1

    with open(argv[0], encoding="utf-8") as f:
        raw = json.load(f)

    data = _parse_input(raw)
    report = build_daily_report(data)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
