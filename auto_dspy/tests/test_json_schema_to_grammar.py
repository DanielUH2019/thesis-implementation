import pytest
from pydantic import BaseModel
from auto_dspy.json_schema_to_grammar import SchemaConverter


def test_basic_pydantic_model_to_grammar():
    class Owner(BaseModel):
        firstName: str
        lastName: str
        age: int

    with open("auto_dspy/tests/grammars/basic_grammar.gbnf", 'r') as grammar_file:
        target_grammar = grammar_file.read()
        converter = SchemaConverter({})
        converter.visit(Owner.model_json_schema(), '')
        grammar_text = converter.format_grammar()
        assert target_grammar.strip() == grammar_text
 
    
