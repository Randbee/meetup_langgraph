"""
workflow_logic.py

This script defines the core workflow functions for a Plan-and-Execute agent system that retrieves weather data. 
It includes functions for planning, executing steps, replanning, and determining when to end the workflow. 
The system uses LangGraph for agent orchestration and integrates with weather tools.

Dependencies:
- templates.classes: Custom classes for planning and execution (PlanExecute, Plan, Act, Response).
- langchain_openai: For OpenAI chat models.
- langgraph.prebuilt: For creating a React agent.
- templates.prompts: Custom prompt templates for planning, replanning, and execution.
- langgraph.graph: For defining the workflow graph.
- src.tools: For the weather tools (`get_weather_south_america`, `get_weather_north_america`).

Global Variables:
- tools: A list of tools available to the agent (weather retrieval functions).
- llm: An instance of the OpenAI chat model (GPT-4).

Functions:
- get_agent_executor(): Returns an agent executor.
- get_planner(): Returns a planner.
- get_replanner(): Returns a replanner.
- execute_step(state): Executes a step in the plan and updates the state.
- plan_step(state): Generates a plan based on the input and initializes the execution state.
- replan_step(state): Replans based on the current state and updates the plan or returns a response.
- should_end(state): Determines whether the workflow should end.
"""

from templates.classes import PlanExecute, Plan, Act, Response
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from templates.prompts import get_planner_prompt, get_replanner_prompt, get_agent_executor_prompt
from langgraph.graph import END
from src.tools import get_weather_south_america, get_weather_north_america, tavily_search
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os

# List of tools available to the agent
tools = [get_weather_south_america, get_weather_north_america, tavily_search]

# Initialize the OpenAI chat model
llm = ChatOpenAI(model="gpt-4o-2024-08-06", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

def get_agent_executor():
    """
    Returns an agent executor.

    Returns:
        AgentExecutor: An agent executor.
    """
    agent_executor_prompt = get_agent_executor_prompt()
    return create_react_agent(model=llm, tools=tools, messages_modifier=agent_executor_prompt)

def get_planner():
    """
    Returns a planner.

    Returns:
        Chain: A planner chain.
    """
    planner_prompt = get_planner_prompt()

    return planner_prompt | llm.with_structured_output(Plan)

def get_replanner():
    """
    Returns a replanner.

    Returns:
        Chain: A replanner chain.
    """
    replanner_prompt = get_replanner_prompt()
    return replanner_prompt | llm.with_structured_output(Act)

def execute_step(state: PlanExecute):
    """
    Executes a step in the plan and updates the state.

    Args:
        state (PlanExecute): The current state of the workflow, including the plan and executed steps.

    Returns:
        dict: The updated state with the executed step and updated history.
    """
    plan = state.get("plan", [])  # Get the plan (default to empty list if not present)
    executed_steps = state.get("executed_steps", [])  # Get executed steps (default to empty list if not present)

    # Get the next step to execute
    step_to_execute = plan[len(executed_steps)]

    try:
        # Execute the step using the agent executor
        agent_executor = get_agent_executor()
        task_formatted = f"""Execute the following step from the plan: {step_to_execute}."""
        agent_response = agent_executor.invoke(
            {"messages": [("user", task_formatted)]}
        )

        # Add the executed step to the list
        executed_steps.append(step_to_execute)

        # Update the state with the executed step and response
        return {
            "past_steps": state.get("past_steps", []) + [(step_to_execute, agent_response["messages"][-1].content)],
            "executed_steps": executed_steps,
        }
    except Exception as e:
        # Handle the case where the step fails
        return {
            "past_steps": state.get("past_steps", []) + [(step_to_execute, f"Step failed: {str(e)}")],
            "executed_steps": executed_steps,
            "response": f"Step failed: {str(e)}",  # Indicate that the workflow should end
        }

def plan_step(state: PlanExecute):
    """
    Generates a plan based on the input and initializes the execution state.

    Args:
        state (PlanExecute): The current state of the workflow.

    Returns:
        dict: The updated state with the generated plan.
    """
    planner = get_planner()
    plan = planner.invoke({"input": [("user", state["input"]) ]})
    
    return {"plan": plan.steps, "executed_steps": []}

def replan_step(state: PlanExecute):
    """
    Replans based on the current state and updates the plan or returns a response.

    Args:
        state (PlanExecute): The current state of the workflow.

    Returns:
        dict: The updated state with a new plan or a final response.
    """
    replanner = get_replanner()
    output = replanner.invoke(state)
    
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    return {"plan": output.action.steps, "executed_steps": []}

def should_end(state: PlanExecute):
    """
    Determines whether the workflow should end.

    Args:
        state (PlanExecute): The current state of the workflow.

    Returns:
        str or END: Returns `END` if the workflow should terminate, otherwise returns "agent" to continue.
    """
    if state["response"]:
        return END
    return "agent"
