import asyncio
from .worker.crypto_worker import run
if __name__ == "__main__":
    asyncio.run(run())
