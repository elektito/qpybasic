import logging
import traceback
import unittest
import sys
from compiler import Compiler, CompileError, EC
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


class TestCallSub1:
    code = """
sub foo
   print "foo"
end sub

call foo
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestCallSub2:
    code = """
sub foo
    print "foo"
end sub

foo
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestCallSub3:
    code = """
sub foo(msg as string)
   print msg
end sub

call foo("foo")
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestCallSub4:
    code = """
sub foo(msg as string)
    print msg
end sub

foo "foo"
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestCallSub5:
    code = """
sub foo(msg as string, n&)
   print msg; n&
end sub

call foo("foo", 100)
    """

    cevents = []
    vevents = [
        ('print', 'foo 100 \n'),
    ]


class TestCallSub6:
    code = """
sub foo(msg as string, n&)
    print msg; n&
end sub

foo "foo", 100
    """

    cevents = []
    vevents = [
        ('print', 'foo 100 \n'),
    ]


class TestSubCallMismatch1:
    code = """
sub foo(msg as string, n&)
    print msg; n&
end sub

foo "foo"
    """

    cevents = [
        ('error', EC.ARGUMENT_COUNT_MISMATCH)
    ]
    vevents = []


class TestSubCallMismatch2:
    code = """
sub foo(msg as string, n&)
    print msg; n&
end sub

foo "foo", 1, 2
    """

    cevents = [
        ('error', EC.ARGUMENT_COUNT_MISMATCH)
    ]
    vevents = []


class TestSubCallMismatch3:
    code = """
declare sub foo(msg$, n&)

foo "foo"

sub foo(msg as string, n&)
    print msg; n&
end sub
    """

    cevents = [
        ('error', EC.ARGUMENT_COUNT_MISMATCH)
    ]
    vevents = []


class TestSubCallMismatch4:
    code = """
declare sub foo(msg$, n&)

foo "foo", 1, 2

sub foo(msg as string, n&)
    print msg; n&
end sub
    """

    cevents = [
        ('error', EC.ARGUMENT_COUNT_MISMATCH)
    ]
    vevents = []


class TestFunctionCallMismatch1:
    code = """
function foo(msg as string, n&)
    foo = n& + 100
end function

ret = foo("foo")
    """

    cevents = [
        ('error', EC.ARGUMENT_COUNT_MISMATCH)
    ]
    vevents = []


class TestFunctionCallMismatch2:
    code = """
function foo(msg as string, n&)
    foo = n& + 100
end function

ret = foo("foo", 1, 2)
    """

    cevents = [
        ('error', EC.ARGUMENT_COUNT_MISMATCH)
    ]
    vevents = []


class TestFunctionCallMismatch3:
    code = """
declare function foo(msg as string, n as long)

ret = foo("foo")

function foo(msg as string, n&)
    foo = n& + 100
end function
    """

    cevents = [
        ('error', EC.ARGUMENT_COUNT_MISMATCH)
    ]
    vevents = []


class TestFunctionCallMismatch4:
    code = """
declare function foo(msg as string, n as long)

ret = foo("foo", 1, 2)

function foo(msg as string, n&)
    foo = n& + 100
end function
    """

    cevents = [
        ('error', EC.ARGUMENT_COUNT_MISMATCH)
    ]
    vevents = []


class TestFunctionRecursion1:
    code = """
declare function fib%(n%)

dim i as integer

for i = 1 to 10
   print "fib"; i; fib%(i)
next

function fib%(n%)
   if n% < 3 then
      fib% = 1
   else
      fib% = fib%(n%-2) + fib%(n%-1)
   end if
end function
    """

    cevents = []
    vevents = [
        ('print', 'fib 1  1 \n'),
        ('print', 'fib 2  1 \n'),
        ('print', 'fib 3  2 \n'),
        ('print', 'fib 4  3 \n'),
        ('print', 'fib 5  5 \n'),
        ('print', 'fib 6  8 \n'),
        ('print', 'fib 7  13 \n'),
        ('print', 'fib 8  21 \n'),
        ('print', 'fib 9  34 \n'),
        ('print', 'fib 10  55 \n'),
    ]


class TestSubRecursion1:
    code = """
declare sub fib(n%, r%)

dim i as integer

for i = 1 to 10
   call fib(i, r%)
   print "fib"; i; r
next

sub fib(n%, r%)
   if n% < 3 then
      r% = 1
   else
      fib n% - 1, r1%
      fib n% - 2, r2%
      r% = r1% + r2%
   end if
end sub
    """

    cevents = []
    vevents = [
        ('print', 'fib 1  1 \n'),
        ('print', 'fib 2  1 \n'),
        ('print', 'fib 3  2 \n'),
        ('print', 'fib 4  3 \n'),
        ('print', 'fib 5  5 \n'),
        ('print', 'fib 6  8 \n'),
        ('print', 'fib 7  13 \n'),
        ('print', 'fib 8  21 \n'),
        ('print', 'fib 9  34 \n'),
        ('print', 'fib 10  55 \n'),
    ]


class TestDeclareDefMismatch1:
    code = """
declare sub foo(n, m)

sub foo(n)
end sub
    """

    cevents = [
        ('error', EC.SUB_DEF_NOT_MATCH_DECL),
    ]
    vevents = []


class TestDeclareDefMismatch2:
    code = """
sub foo(n)
end sub

declare sub foo(n, m)
    """

    cevents = [
        ('error', EC.DECL_NOT_MATCH_DEF),
    ]
    vevents = []


class TestDeclareDefMismatch3:
    code = """
function foo(n)
end function

declare function foo(n, m)
    """

    cevents = [
        ('error', EC.DECL_NOT_MATCH_DEF),
    ]
    vevents = []


class TestDeclareDefMismatch4:
    code = """
declare function foo(n, m)

function foo(n)
end function
    """

    cevents = [
        ('error', EC.FUNC_DEF_NOT_MATCH_DECL),
    ]
    vevents = []


class TestConflictingDecl1:
    code = """
declare function foo(n, m)
declare function foo(n)
    """

    cevents = [
        ('error', EC.CONFLICTING_DECL),
    ]
    vevents = []


class TestConflictingDecl2:
    code = """
declare function foo(n)
declare function foo&(n)
    """

    cevents = [
        ('error', EC.CONFLICTING_DECL),
    ]
    vevents = []


class TestConflictingDecl3:
    code = """
declare sub foo(n, m)
declare sub foo(n)
    """

    cevents = [
        ('error', EC.CONFLICTING_DECL),
    ]
    vevents = []


class TestConflictingDecl4:
    code = """
declare sub foo(n)
declare function foo(n)
    """

    cevents = [
        ('error', EC.CONFLICTING_DECL),
    ]
    vevents = []


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
        module = None
    tc.assertEqual(events, case.cevents)

    if module:
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
            sys.stdout.flush()

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
