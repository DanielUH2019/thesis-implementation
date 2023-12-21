from constants import REACT_MODULE, COT_MODULE, POT_MODULE
import dspy



def instantiate_prompt_module(name: str, signature):
    if name == REACT_MODULE:
       return dspy.ReAct(signature)
    elif name == COT_MODULE:
        return dspy.ChainOfThought(signature)
    elif name == POT_MODULE:
        return dspy.ProgramOfThought(signature)

    raise ValueError(f"Prompt module {name} not supported")

