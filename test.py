import logging
import traceback
import unittest
from compiler import Compiler, CompileError
from vm import Machine

logger = logging.getLogger(__name__)

# we are going to create this phony test case, so that we can use its
# assert* methods, which are much nicer than the vanilla assert
# statement.
tc = unittest.TestCase()

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
    except CompileError as e:
        event_handler(('error', e.code))
    tc.assertEqual(events, case.cevents)

    events = []
    machine = Machine.load(module)
    machine.event_handler = event_handler
    machine.launch()
    machine.shutdown()

    tc.assertEqual(events, case.vevents)


def main():
    failed = []
    success = []

    for name, value in globals().items():
        if name.startswith('Test') and isinstance(value, type):
            try:
                run_test_case(name, value)
            except Exception as e:
                failed.append((name, e))
                if isinstance(e, AssertionError):
                    print('F', end='')
                else:
                    print('E', end='')
            else:
                success.append(name)
                print('.', end='')

    print()
    if len(failed) == 0:
        print(f'All {len(success)} test case(s) ran successfully.')
    else:
        print('Failures:\n')
        for name, exc in failed:
            print(f'Failed test case: {name}')
            print('Exception:')
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        print('---\n')
        total = len(failed) + len(success)
        print(f'{len(failed)} out of {total} test case(s) failed.')


if __name__ == '__main__':
    main()
