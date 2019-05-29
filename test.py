import logging
import traceback
from compiler import Compiler, CompileError
from vm import Machine

logger = logging.getLogger(__name__)

class TestCls:
    code = """
cls
    """

    cevents = []
    vevents = [
        ('cls',)
    ]

class TestPrint1:
    code = """
print 10
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
    ]


def run_test_case(name, case):
    events = []
    def event_handler(event):
        events.append(event)

    logger.info(f'Running test case: {name}')

    c = Compiler()
    try:
        module = c.compile(case.code)
    except CompileError:
        event_handler(('error', e.code))
    assert events == case.cevents

    events = []
    machine = Machine.load(module)
    machine.event_handler = event_handler
    machine.launch()
    machine.shutdown()

    assert events == case.vevents


def main():
    failed = []
    success = []

    for name, value in globals().items():
        if name.startswith('Test') and isinstance(value, type):
            try:
                run_test_case(name, value)
            except Exception:
                failed.append(name)
                traceback.print_exc()
            else:
                success.append(name)

    if len(failed) == 0:
        print(f'All {len(success)} test case(s) ran successfully.')
    else:
        total = len(failed) + len(success)
        print(f'{len(failed)} out of {total} test case(s) failed.')


if __name__ == '__main__':
    main()
