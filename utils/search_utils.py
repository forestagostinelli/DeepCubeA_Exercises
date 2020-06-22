from typing import List
from environments.environment_abstract import Environment, State


def is_valid_soln(state: State, soln: List[int], env: Environment) -> bool:
    soln_state: State = state
    move: int
    for move in soln:
        soln_state = env.next_state([soln_state], move)[0][0]

    return env.is_solved([soln_state])[0]
