"""battwin: self-updating digital twin engine for Li-ion diagnostics (NASA Ames 18650 LCO/graphite).

The engine is free of Streamlit. The app checks ``__version__`` against its REQUIRED_ENGINE.
"""
__version__ = "1.0.0"
ENGINE_MAJOR_MINOR = ".".join(__version__.split(".")[:2])
