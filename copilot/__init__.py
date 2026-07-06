def __getattr__(name):
    # Lazy: `copilot.agent` pulls in the whole LLM stack (google.adk, litellm),
    # which the capture daemon (which only needs copilot.gcs) never uses. ADK
    # discovery still works: its loader imports the `.agent` submodule directly
    # when the package has no root_agent.
    if name == "agent":
        import importlib

        return importlib.import_module(".agent", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
