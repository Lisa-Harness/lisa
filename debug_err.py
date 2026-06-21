import traceback, sys, asyncio
from examples.little_lisa.main import _run_single_turn_async
async def main():
    try:
        await _run_single_turn_async('hello', False)
    except Exception as e:
        if hasattr(e, 'exceptions'):
            for ex in e.exceptions:
                traceback.print_exception(type(ex), ex, ex.__traceback__)
asyncio.run(main())
