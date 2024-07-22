"""DQ Report."""

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from user_input.metrics import Metric

import pandas as pd

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report."""
        self.report_ = {
            "title": f"DQ Report for tables {sorted(list(tables.keys()))}",
            "result": pd.DataFrame(columns=["table_name", "metric", "limits", "values", "status", "error"]),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total": len(self.checklist),
            "passed_pct": 0,
            "failed_pct": 0,
            "errors_pct": 0,
        }
        report = self.report_

        # Check if engine supported
        if self.engine != "pandas":
            raise NotImplementedError("Only pandas API currently supported!")

        for table_name, metric, limits in self.checklist:
            if table_name not in tables:
                report["errors"] += 1
                report["result"] = report["result"].append({
                    "table_name": table_name,
                    "metric": str(metric),
                    "limits": str(limits),
                    "values": {},
                    "status": "E",
                    "error": "Table not found in tables dictionary"
                }, ignore_index=True)
                continue
            df = tables[table_name]

            try:
                result = metric(df)
                status = "."
                all_checks = True
                # Check limits
                for key, (lower_limit, upper_limit) in limits.items():

                    value = result.get(key, 0)  # If key not present, default to 0
                    if not (lower_limit <= value <= upper_limit):
                        all_checks = False
                        report["failed"] += 1
                        status = "F"
                        break
                if all_checks:
                    report['passed'] += 1

            except Exception as e:
                report["errors"] += 1
                status = "E"
                result = {}
                report["result"] = report["result"].append({
                    "table_name": table_name,
                    "metric": str(metric),
                    "limits": str(limits),
                    "values": result,
                    "status": status,
                    "error": str(e)
                }, ignore_index=True)
            # Output format
            report["result"] = report["result"].append({
                "table_name": table_name,
                "metric": str(metric),
                "limits": str(limits),
                "values": result,
                "status": status,
                "error": ""
            }, ignore_index=True)


        total_checks = report["total"]
        report["passed_pct"] = (report["passed"] / total_checks) * 100
        report["failed_pct"] = (report["failed"] / total_checks) * 100
        report["errors_pct"] = (report["errors"] / total_checks) * 100


        return report

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
