import outlines

@outlines.prompt
def instruction_to_extract_arguments_to_json(inputs, required_arguments: dict[str, str]):
    """ Given the inputs: {{inputs}}
        And considering the following arguments with their descriptions: {{required_arguments}}
        Extract information from the instruction that match the given arguments to call a function in a json format
    """

@outlines.prompt
def instruction_for_algorithms_compatibility(dataset_description: str, current_signature: str):
    """
    A signature is a declarative specification of input/output behavior of a DSPy module.\
    Instead of investing effort into how to get your LM to do a sub-task, signatures enable you to inform DSPy what the sub-task is. \
    Later, the DSPy compiler will figure out how to build a complex prompt for your large LM (or finetune your small LM) specifically for your signature, \
    on your data, and within your pipeline.
    A signature consists of three simple elements:
    1. A minimal description of the sub-task the LM is supposed to solve.
    2. A description of one or more input fields (e.g., input question) that will we will give to the LM.
    3. A description of one or more output fields (e.g., the question's answer) that we will expect from the LM.

    Given the dataset_description: {{dataset_description}}
    Can the following signature be used directly?: {{current_signature}}
    """