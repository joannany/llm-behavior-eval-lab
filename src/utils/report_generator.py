"""
Report Generator

Generates comprehensive evaluation reports in various formats
including JSON, HTML, and PDF.
"""

from dataclasses import dataclass
from typing import Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "LLM Evaluation Report"
    include_raw_results: bool = False
    include_visualizations: bool = True
    format: str = "json"


class ReportGenerator:
    """
    Generates evaluation reports from results.
    
    Supports multiple output formats and customizable sections.
    
    Example:
        >>> generator = ReportGenerator(results)
        >>> generator.create_json("report.json")
        >>> generator.create_html("report.html")
    """
    
    def __init__(
        self,
        results: Any,
        config: ReportConfig | None = None,
    ):
        """
        Initialize report generator.
        
        Args:
            results: Evaluation results (EvaluationReport or dict)
            config: Report configuration
        """
        self.results = results
        self.config = config or ReportConfig()
        
        logger.info("Initialized ReportGenerator")
    
    def create_json(self, output_path: str) -> str:
        """
        Create JSON format report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the created report
        """
        report_data = self._prepare_report_data()
        
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON report created: {output_path}")
        return output_path
    
    def create_html(self, output_path: str) -> str:
        """
        Create HTML format report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the created report
        """
        report_data = self._prepare_report_data()
        
        html = self._generate_html(report_data)
        
        with open(output_path, "w") as f:
            f.write(html)
        
        logger.info(f"HTML report created: {output_path}")
        return output_path
    
    def create_pdf(self, output_path: str) -> str:
        """
        Create PDF format report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the created report
        """
        # First create HTML, then convert to PDF
        html_content = self._generate_html(self._prepare_report_data())
        
        # For now, save as HTML with PDF extension note
        html_path = output_path.replace(".pdf", ".html")
        
        with open(html_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Report created: {html_path} (PDF conversion requires additional libraries)")
        return html_path
    
    def _prepare_report_data(self) -> dict:
        """Prepare report data structure."""
        if hasattr(self.results, "to_dict"):
            data = self.results.to_dict()
        elif isinstance(self.results, dict):
            data = self.results
        else:
            data = {"results": str(self.results)}
        
        return {
            "title": self.config.title,
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_summary(data),
            "data": data,
        }
    
    def _generate_summary(self, data: dict) -> dict:
        """Generate executive summary."""
        summary = {
            "total_tests": data.get("total_tests", 0),
            "pass_rate": data.get("pass_rate", 0),
            "mean_score": data.get("mean_score", 0),
        }
        
        # Add any aggregate metrics
        if "aggregate_metrics" in data:
            summary["metrics"] = data["aggregate_metrics"]
        
        return summary
    
    def _generate_html(self, report_data: dict) -> str:
        """Generate HTML report."""
        summary = report_data.get("summary", {})
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_data.get('title', 'Evaluation Report')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric {{
            display: inline-block;
            padding: 15px 25px;
            margin: 5px;
            background: #f0f4f8;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        pre {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_data.get('title', 'Evaluation Report')}</h1>
        <p>Generated: {report_data.get('generated_at', '')}</p>
    </div>
    
    <div class="card">
        <h2>Executive Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{summary.get('total_tests', 0)}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('pass_rate', 0):.1%}</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('mean_score', 0):.3f}</div>
                <div class="metric-label">Mean Score</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Detailed Results</h2>
        <pre>{json.dumps(report_data.get('data', {}), indent=2, default=str)[:5000]}...</pre>
    </div>
</body>
</html>"""
        
        return html
