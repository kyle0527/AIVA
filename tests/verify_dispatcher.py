import json
from services.core.aiva_core.task_planning.dispatcher import PlanningDispatcher

def test_exploration_sync():
    dispatcher = PlanningDispatcher()
    print("Testing call_exploration_sync(tool='list')...")
    # This invokes aiva_internal_executor.py
    result = dispatcher.call_exploration_sync(tool="list", timeout=10)
    print(f"Return code: {result.returncode}")

    if result.returncode == 0:
        print("SUCCESS: List command executed.")
    else:
        print(f"FAILURE: {result.stderr}")
        exit(1)

if __name__ == "__main__":
    test_exploration_sync()
