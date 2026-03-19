"""SoulGraph / Intention extraction adapter."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

SOULGRAPH_PATH = Path.home() / "soulgraph"


class SoulGraphAdapter(DetectorAdapter):
    """Adapts SoulEngine.ingest() → DetectorResult.

    Activation: at least one intention/soul item extracted with specificity > 0.3.
    Confidence: average specificity of extracted items.
    """

    dimension = Dimension.SOULGRAPH

    def __init__(self):
        self._engine = None

    def _load(self):
        if self._engine is None:
            if str(SOULGRAPH_PATH) not in sys.path:
                sys.path.insert(0, str(SOULGRAPH_PATH))
            from soulgraph.engine import SoulEngine
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            self._engine = SoulEngine(api_key=api_key)

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        try:
            graph = self._engine.ingest(text)
            items = getattr(graph, "items", [])
            if not items:
                return self._make_result(0.0, detail={"items": 0})
            specificities = [getattr(i, "specificity", 0.5) for i in items]
            avg_spec = sum(specificities) / len(specificities)
            return self._make_result(
                confidence=avg_spec,
                detail={"items": len(items), "avg_specificity": round(avg_spec, 3)},
            )
        except Exception as e:
            return self._make_result(0.0, detail={"error": str(e)})
