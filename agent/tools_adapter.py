"""Wraps mcp_server.py's plain functions with a call surface agent.py and the
specialists can use interchangeably, so tools come from this direct-import
adapter (Day 1 dev/testing) or the real MultiServerMCPClient-provided
LangChain tools (Step 5, over stdio) without either side caring which.

This imports mcp_server.py's functions directly rather than reimplementing
them -- there is exactly one copy of the tool logic, so nothing can drift
between "what gets tested against" and "what actually runs."

DirectTool supports BOTH calling conventions actually in use in this repo:
  tools["x"].invoke({...})   -- agent.py's nodes (Person B)
  tools["x"]({...})          -- specialists' web_search calls (Person A) --
                                 confirmed from her real committed code, not
                                 the pasted spec (which used .invoke()-style
                                 prose but her specialists never call it that
                                 way) -- supporting both means neither side
                                 has to change working, tested code.
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

    def __call__(self, kwargs: dict):
        return self.invoke(kwargs)


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
        "web_search": DirectTool(_m.web_search),
    }
