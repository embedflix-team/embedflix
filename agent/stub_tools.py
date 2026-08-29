# agent/stub_tools.py
# Fake versions of all 9 MCP tools for testing nodes in isolation

def make_stub_tools():
    return {
        "run_pipeline": lambda kwargs: {
            "gauc": 0.6674,
            "ndcg5": 0.5357,
            "primary": 0.6016,
            "iteration": kwargs.get("iteration", 1)
        },
        "parse_scores": lambda kwargs: {
            "gauc": 0.6674,
            "ndcg5": 0.5357,
            "primary": 0.6016
        },
        "log_iteration": lambda kwargs: {
            "status": "logged",
            "iteration": kwargs.get("iteration", 1)
        },
        "save_checkpoint": lambda kwargs: {
            "status": "saved",
            "path": f"checkpoints/iter_{kwargs.get('iteration', 1)}.json"
        },
        "restore_checkpoint": lambda kwargs: {
            "status": "restored",
            "path": kwargs.get("path", "checkpoints/iter_1.json")
        },
        "read_file": lambda kwargs: {
            "content": "# stub file content\n"
        },
        "edit_file": lambda kwargs: {
            "status": "edited",
            "path": kwargs.get("path", "baseline.py")
        },
        "track_resources": lambda kwargs: {
            "tokens_used": 1500,
            "wall_clock_seconds": 45
        },
        "format_submission": lambda kwargs: {
            "status": "formatted",
            "path": "submission.csv"
        },
        "web_search": lambda kwargs: {
            "results": f"[STUB] Search results for: {kwargs.get('query', '')}",
            "query": kwargs.get("query", ""),
            "search_type": kwargs.get("search_type", "concept")
        }
    }