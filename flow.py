from pocketflow import Flow
# Import all node classes from nodes.py
from nodes import (
    FetchRepo,
    IdentifyConcepts,
    GeneratePuzzles,
    WritePuzzles
)

def create_puzzle_flow():
    """Creates and returns the codebase exercise generation flow."""

    # Instantiate nodes
    fetch_repo = FetchRepo()
    identify_concepts = IdentifyConcepts(max_retries=1, wait=20)
    generate_puzzles = GeneratePuzzles(max_retries=1, wait=20)
    write_puzzles = WritePuzzles()
    # Connect nodes in sequence based on the design
    fetch_repo >> identify_concepts
    identify_concepts >> generate_puzzles
    generate_puzzles >> write_puzzles
    # Create the flow starting with FetchRepo
    puzzle_flow = Flow(start=fetch_repo)

    return puzzle_flow

