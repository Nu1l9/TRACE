import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True


def require_local_runtime_sources():
    """Fail before hashing source files that macOS has evicted to the cloud.

    The extension lock hashes these exact source groups on every run. Reading a
    dataless file can wait indefinitely inside the kernel, with no Python
    exception or pipeline log to explain the delay.
    """
    root = Path(__file__).resolve().parent
    files = [*root.joinpath('extensions').glob('*.py'),
             *root.joinpath('literature_retrieval').glob('*.py'),
             *root.joinpath('resources/rob').rglob('*.txt')]
    vendor = root / 'vendor/integration'
    files.extend(vendor.rglob('*.py'))
    files.extend(vendor.rglob('*.R'))
    # SF_DATALESS is not exposed by Python's stat module on every macOS build.
    dataless = 0x40000000
    unavailable = [path for path in files
                   if getattr(path.stat(), 'st_flags', 0) & dataless]
    if unavailable:
        examples = '\n'.join(f'  {path.relative_to(root)}' for path in unavailable[:5])
        raise SystemExit(
            f'ERROR: {len(unavailable)} pipeline source/guidance files are macOS cloud placeholders '
            '(dataless). Download the meta_pipeline folder to this Mac before running.\n'
            f'Examples:\n{examples}\n'
            'No pipeline stage or model call has started.'
        )


def dashboard_link(args):
    """Print a directly clickable local dashboard link for a live run."""
    if not args or args[0] not in ("run", "update", "review") or "--plan-only" in args or "--prepare-only" in args or "--output" not in args:
        return
    index = args.index("--output")
    if index + 1 >= len(args):
        return
    dashboard = Path(args[index + 1]).expanduser().resolve() / "dashboard.html"
    url = dashboard.as_uri()
    print(f"\033]8;;{url}\033\\[dashboard] Open live run panel\033]8;;\033\\  {url}", flush=True)


def timestamped_output(argv):
    """Keep fresh pipeline runs separate from existing checkpoints."""
    args = list(argv)
    if not args or args[0] not in ("run", "update", "review") or "--resume" in args or "--plan-only" in args or "--output" not in args:
        return args
    index = args.index("--output")
    if index + 1 >= len(args):
        return args
    base = Path(args[index + 1])
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = base.with_name(base.name + "_" + stamp)
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(base.name + "_" + stamp + f"_{suffix:02d}")
        suffix += 1
    args[index + 1] = str(candidate)
    print(f"[pipeline] New run output: {candidate}", flush=True)
    return args


def main(argv=None):
    """Shared launcher used by the public TRACE CLI and compatible entrypoints."""
    args = list(sys.argv[1:] if argv is None else argv)
    # Accept argparse's --output=value form with the same isolation guarantees.
    args = [part for arg in args for part in
            (arg.split("=", 1) if arg.startswith("--output=") else [arg])]
    if args and args[0] in ('run', 'check', 'update', 'retry-analysis', 'review') and not any(a in args for a in ('-h', '--help')):
        require_local_runtime_sources()
    if args and args[0] == "rob-retry":
        from extensions.rob_resilience import main as command_main
        return command_main(args[1:])
    if args and args[0] == "expert":
        from extensions.expert_cli import main as command_main
        return command_main(args[1:])
    from extensions.__main__ import main as command_main
    if any(a in args for a in ("-h", "--help")):
        return command_main(args)
    arguments = timestamped_output(args)
    dashboard_link(arguments)
    try:
        return command_main(arguments)
    finally:
        # Presentation handoff only; never changes scientific stage state or resume locks.
        try:
            from expert_qa.integration import after_pipeline
            after_pipeline(arguments)
        except Exception as exc:
            print(f'[results] Integration unavailable ({type(exc).__name__}); saved analysis results are unaffected.', flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
