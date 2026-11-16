"""MINERS → extraction/review → frozen Analysis analysis integration."""
from .contracts import ContractError, VERSION
__version__ = '1.0.0'

def run_request(request, output_dir, *, base_dir='.', env_file=None, resume=False):
    """Python entrypoint for a future Brain or web service; returns the run manifest."""
    from pathlib import Path
    from .contracts import normalize_request
    from .pipeline import Runner,runtime_environment,preflight_runtime
    normalized=normalize_request(request,Path(base_dir))
    environment=runtime_environment(normalized,env_file)
    preflight_runtime(normalized,environment)
    return Runner(normalized,output_dir,environment,resume).run()
