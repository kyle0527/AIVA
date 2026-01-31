
import asyncio
import sys
import os

# Add project root to path
sys.path.append(r"C:\D\fold7\AIVA-git")

from services.core.aiva_core.task_planning.commander.plan_builder import PlanBuilder

class MockEngine:
    async def enhance_attack_plan(self, **kwargs):
        return {"similar_techniques": []}
    def get_high_quality_samples(self, **kwargs):
        return []
    def generate_decision(self, **kwargs):
        return {
            "attack_vector": "reconnaissance", 
            "confidence": 0.9,
            "recommended_tools": ["nmap", "xss_scanner"]
        }

async def main():
    builder = PlanBuilder(
        rag_engine=MockEngine(), 
        decision_engine=MockEngine(),
        experience_manager=MockEngine()
    )
    
    context = {
        "target": "http://test-target.com",
        "objective": "Perform XSS vulnerability scan"
    }
    
    print("Generating plan...")
    result = await builder.plan_attack(context)
    
    if result["success"]:
        plan = result["plan"]
        print("\n=== Generated Phases ===")
        for phase in plan["phases"]:
            print(f"\nPhase: {phase['name']}")
            for step in phase["steps"]:
                print(f"  - {step}")
                if "Command:" in step:
                    print("    ✅ CLI Command Found!")
                else:
                    print("    ⚠️ No exact command match (expected for generic steps)")
    else:
        print("Plan generation failed:", result.get("error"))

if __name__ == "__main__":
    asyncio.run(main())
