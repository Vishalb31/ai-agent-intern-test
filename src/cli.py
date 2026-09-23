import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.core import SupportAgent


def main():
    agent = SupportAgent()
    history = []
    print("=" * 60)
    print(" Aster & Row Support Agent CLI")
    print(" Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Ending session. Goodbye!")
                break

            response = agent.chat(user_input, history=history)

            print("\nAgent:")
            print(response.answer)

            if response.sources:
                print(f"\n[Sources: {', '.join(response.sources)}]")

            if response.handoff_recommended:
                print("\n[Status: ⚠️ Human Support Handoff Recommended]")

            print("-" * 60 + "\n")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response.answer})

        except KeyboardInterrupt:
            print("\nEnding session.")
            break


if __name__ == "__main__":
    main()