"""Wraps mcp_server.py's plain functions with a .invoke({...}) call surface,
so agent.py's nodes call tools["x"].invoke({...}) identically whether the
tools come from this direct-import adapter (Day 1 dev/testing) or the real
MultiServerMCPClient-provided LangChain tools (Step 5, over stdio).

This imports mcp_server.py's functions directly rather than reimplementing
them -- there is exactly one copy of the tool logic, so nothing can drift
between "what Person B tests against" and "what actually runs."
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcp_server as _m


class DirectTool:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__

    def invoke(self, kwargs: dict):
        return self.fn(**kwargs)


def build_tools():
    return {
        "run_pipeline": DirectTool(_m.run_pipeline),
        "read_file": DirectTool(_m.read_file),
        "edit_file": DirectTool(_m.edit_file),
        "save_checkpoint": DirectTool(_m.save_checkpoint),
        "restore_checkpoint": DirectTool(_m.restore_checkpoint),
        "log_iteration": DirectTool(_m.log_iteration),
        "track_resources": DirectTool(_m.track_resources),
        "format_submission": DirectTool(_m.format_submission),
        "parse_scores": DirectTool(_m.parse_scores),
    }
