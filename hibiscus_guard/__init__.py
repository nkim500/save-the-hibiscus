def __getattr__(name):
    # Lazy: `hibiscus_guard.agent` pulls in the whole LLM stack (google.adk,
    # litellm — tens of seconds), which perception-only consumers like the
    # capture daemon never need. ADK discovery still works: its loader imports
    # the `.agent` submodule directly when the package has no root_agent.
    if name == "agent":
        import importlib

        return importlib.import_module(".agent", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
