from langgraph.graph import StateGraph, START, END
from templates.classes import PlanExecute
from src.workflow_logic import plan_step, execute_step, replan_step, should_end
import os
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel
from rich.console import Console
from rich.pretty import pprint

# Inicializar consola Rich
console = Console()

# Cargar variables de entorno
load_dotenv()

# Configurar el grafo de flujo de trabajo
workflow = StateGraph(PlanExecute)

# Añadir nodos al grafo
workflow.add_node("planner", plan_step)  # Nodo para planificar
workflow.add_node("agent", execute_step)  # Nodo para ejecutar
workflow.add_node("replan", replan_step)  # Nodo para replanificar

# Definir las conexiones entre nodos
workflow.add_edge(START, "planner")  # Inicia con el planificador
workflow.add_edge("planner", "agent")  # Transición del planificador al agente
workflow.add_edge("agent", "replan")  # Transición del agente al replanteo

# Añadir transiciones condicionales para replanteo
workflow.add_conditional_edges(
    "replan",
    should_end,  # Función para determinar si finalizar o continuar
    ["agent", END],  # Posibles transiciones: volver al agente o terminar
)

# Definir la entrada del usuario
user_input = "For Paris, tell me the actual weather conditions, include too the wind and pressure conditions. After that, convert the temperature from Celsius to Fahrenheit and the pressure from mb to Pascal."

# Estado inicial
initial_state = {
    "input": user_input,
    "plan": [],  # Se generará en el primer paso
    "past_steps": [],  # Pasos previos inicializados como lista vacía
    "response": "",  # Se generará en el flujo de trabajo
}

# Compilar el grafo
graph = workflow.compile()

# Definir la configuración
config = {"configurable": {"thread_id": 100}}


result = graph.invoke(input=initial_state, config=config)

console.print(Panel.fit("[bold cyan]🚀 EJECUCIÓN DEL WORKFLOW[/bold cyan]", style="bold magenta"))

console.print(Panel(f"[bold yellow]Entrada del usuario:[/bold yellow]\n[green]{user_input}[/green]", style="blue"))

console.print(Panel.fit("[bold cyan]📌 Resultado del Workflow:[/bold cyan]", style="bold green"))

pprint(result, expand_all=True)
