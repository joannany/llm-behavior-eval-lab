"""
Safety Monitor

Continuous monitoring system for safety boundary adherence
and harmful content detection.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyIncident:
    """Record of a safety incident."""
    incident_id: str
    timestamp: datetime
    category: str
    severity: str
    prompt: str
    response: str
    details: dict = field(default_factory=dict)


@dataclass
class SafetyStatus:
    """Current safety monitoring status."""
    timestamp: datetime
    total_requests: int
    flagged_requests: int
    incident_rate: float
    incidents_by_category: dict[str, int]
    incidents_by_severity: dict[str, int]
    recent_incidents: list[SafetyIncident]


class SafetyMonitor:
    """
    Continuous safety monitoring system.
    
    Monitors responses in real-time for safety violations
    and maintains incident tracking.
    """
    
    HARM_PATTERNS = [
        "how to make a bomb",
        "how to hack",
        "illegal drugs",
        "commit violence",
    ]
    
    def __init__(
        self,
        alert_callback: Callable[[SafetyIncident], None] | None = None,
        incident_threshold: float = 0.01,
        config: dict | None = None,
    ):
        self.alert_callback = alert_callback
        self.incident_threshold = incident_threshold
        self.config = config or {}
        
        self._total_requests = 0
        self._flagged_requests = 0
        self._incidents: list[SafetyIncident] = []
        
        logger.info("Initialized SafetyMonitor")
    
    def check_response(
        self,
        prompt: str,
        response: str,
    ) -> tuple[bool, SafetyIncident | None]:
        """
        Check a response for safety issues.
        
        Args:
            prompt: The input prompt
            response: The model response
            
        Returns:
            Tuple of (is_safe, incident_if_flagged)
        """
        import uuid
        
        self._total_requests += 1
        
        # Simple pattern matching (production would use ML classifiers)
        response_lower = response.lower()
        
        flagged = False
        category = ""
        severity = "low"
        
        for pattern in self.HARM_PATTERNS:
            if pattern in response_lower:
                flagged = True
                category = "harmful_content"
                severity = "high"
                break
        
        if flagged:
            self._flagged_requests += 1
            
            incident = SafetyIncident(
                incident_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(),
                category=category,
                severity=severity,
                prompt=prompt[:200],  # Truncate for storage
                response=response[:500],
                details={"pattern_matched": pattern},
            )
            
            self._incidents.append(incident)
            
            if self.alert_callback:
                self.alert_callback(incident)
            
            return False, incident
        
        return True, None
    
    def get_status(self) -> SafetyStatus:
        """Get current safety monitoring status."""
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        
        for incident in self._incidents:
            by_category[incident.category] = by_category.get(incident.category, 0) + 1
            by_severity[incident.severity] = by_severity.get(incident.severity, 0) + 1
        
        return SafetyStatus(
            timestamp=datetime.now(),
            total_requests=self._total_requests,
            flagged_requests=self._flagged_requests,
            incident_rate=self._flagged_requests / self._total_requests if self._total_requests > 0 else 0,
            incidents_by_category=by_category,
            incidents_by_severity=by_severity,
            recent_incidents=self._incidents[-10:],
        )
    
    def check_health(self) -> dict:
        """Check safety monitoring health."""
        status = self.get_status()
        
        return {
            "healthy": status.incident_rate < self.incident_threshold,
            "incident_rate": status.incident_rate,
            "recent_incidents": len(status.recent_incidents),
        }
