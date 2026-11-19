"""Run Direct, Normalize and Few-shot on the shared full-size manifest."""
from phase1_runner import run_methods

if __name__ == "__main__":
    run_methods(("direct","normalize","fewshot"))
