"""Lazy simulation branch of main_rec_v4; never imports the real launcher."""
from pathlib import Path
import runpy
import sys


def run_simulation(*,ipc=False,show_window=True,port=8766,open_browser=True):
    simulator=Path(__file__).resolve().parent.parent/'Lift_FSM_simulator_standalone_20260907'
    if not (simulator/'fsm_worker.py').is_file():
        raise RuntimeError(f'Simulator runtime not found: {simulator}')
    sys.path.insert(0,str(simulator))
    if ipc:
        from fsm_worker import main
        return main([],default_window=show_window,entrypoint='main_rec_v4.py')
    # Standalone launch opens the world UI; its Run/Step starts this launcher
    # again as a separate IPC worker, with the native FSM window enabled.
    previous=sys.argv
    try:
        sys.argv=[str(simulator/'serve_v4.py'),'--port',str(port)]
        if not open_browser:
            sys.argv.append('--no-browser')
        runpy.run_path(str(simulator/'serve_v4.py'),run_name='__main__')
    finally:
        sys.argv=previous
    return 0
