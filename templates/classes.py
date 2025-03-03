"""
classes.py

This script defines the data models used in the Plan-and-Execute agent system. 
It includes classes for representing the workflow state (`PlanExecute`), plans (`Plan`), 
responses (`Response`), and actions (`Act`). These classes are used to structure 
the data passed between different components of the workflow.

Dependencies:
- typing: For type annotations and unions.
- typing_extensions: For the `TypedDict` class.
- pydantic: For data validation and settings management.

Classes:
- PlanExecute: Represents the state of the workflow, including the input, plan, past steps, executed steps, and response.
- Plan: Represents a plan consisting of a list of steps to follow.
- Response: Represents a response to the user.
- Act: Represents an action, which can be either a response or a new plan.
"""

from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator

class PlanExecute(TypedDict):
    """
    Represents the state of the Plan-and-Execute workflow.

    Attributes:
    - input (str): The user's input or query.
    - plan (List[str]): The list of steps in the current plan.
    - past_steps (List[Tuple[str, str]]): A list of tuples representing past steps and their outcomes.
    - executed_steps (List[str]): A list of steps that have been executed.
    - response (str): The final response to the user (if any).
    """
    input: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    executed_steps: List[str]
    response: str

class Plan(BaseModel):
    """
    Represents a plan consisting of a list of steps to follow.

    Attributes:
    - steps (List[str]): The list of steps to execute, in sorted order.
    """
    steps: List[str] = Field(
        description="Different steps to follow, should be in sorted order."
    )

class Response(BaseModel):
    """
    Represents a response to the user.

    Attributes:
    - response (str): The content of the response.
    """
    response: str

class Act(BaseModel):
    """
    Represents an action to perform, which can be either a response or a new plan.

    Attributes:
    - action (Union[Response, Plan]): The action to perform. Use `Response` to respond to the user 
      or `Plan` to define further steps for the workflow.
    """
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to the user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )



