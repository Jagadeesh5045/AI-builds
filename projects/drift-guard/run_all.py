"""End-to-end entry point: train -> evaluate -> model card.

Usage:
    python run_all.py
"""

import evaluate
import train


def main() -> None:
    champion_id = train.main()
    evaluate.main(champion_id)
    print("[run_all] done. See artifacts/reports/model_card.md")


if __name__ == "__main__":
    main()
