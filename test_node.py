# test_node.py
from agent.stub_tools import make_stub_tools
from agent.specialists.loss_function_changer import loss_function_changer

# Fake state — simulates what LangGraph will pass in
state = {
    "iteration": 1,
    "current_scores": {"gauc": 0.6674, "ndcg5": 0.5357, "primary": 0.6016},
    "best_scores": {"gauc": 0.6674, "ndcg5": 0.5357, "primary": 0.6016},
    "experiment_history": [],
    "tried_approaches": [],
    "current_code": "",
    "error_message": None,
}

tools = make_stub_tools()

print("🔄 Running loss_function_changer...")
result = loss_function_changer(state, tools)

print("\n✅ Node output:")
print("HYPOTHESIS:  ", result["hypothesis"])
print("REASONING:   ", result["reasoning"])
print("CODE CHANGE: ", result["code_change_instruction"])
print("TRIED:       ", result["tried_approaches"])