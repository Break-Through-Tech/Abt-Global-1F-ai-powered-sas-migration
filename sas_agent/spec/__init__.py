"""Spec agent (also called the Instruction agent): writes a checked specification for each translation unit.

Code writes the facts layer (analyzer.py); the model writes the contract and interpretation layers
(agent.py); code checks the spec (checks.py) and validates it against SAS's own results (validation.py).
See FirstAgent.md items B to D and design doc §6.5.
"""
