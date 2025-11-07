import asyncio
from .worker.postex_worker import run
if __name__ == "__main__":
    asyncio.run(run())
