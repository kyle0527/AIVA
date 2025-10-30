

from services.aiva_common.schemas import FindingPayload

from .test_result_database import TestResultDatabase


class DataReceptionLayer:
    """Aggregate incoming findings and normalize before persistence."""

    def __init__(self, db: TestResultDatabase) -> None:
        self._db = db

    async def store_finding(self, finding: FindingPayload) -> None:
        await self._db.save_finding(finding)
