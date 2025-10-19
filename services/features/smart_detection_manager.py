# features/smart_detection_manager.py
from typing import Callable, Dict, Any, List

DetectorFunc = Callable[[Dict[str,Any]], Dict[str,Any]]

class SmartDetectionManager:
    def __init__(self):
        self._detectors: Dict[str, DetectorFunc] = {}

    def register(self, name: str, fn: DetectorFunc):
        self._detectors[name] = fn

    def unregister(self, name: str):
        return self._detectors.pop(name, None)

    def run_all(self, input_data: Dict[str,Any]) -> List[Dict[str,Any]]:
        results = []
        for name, fn in self._detectors.items():
            try:
                res = fn(input_data)
            except Exception as e:
                res = {"detector": name, "error": str(e)}
            results.append({"detector": name, "result": res})
        return results

# singleton
_default_manager = SmartDetectionManager()

def get_smart_detection_manager() -> SmartDetectionManager:
    return _default_manager
