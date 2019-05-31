#!/usr/bin/env python3

import logging
import traceback
import unittest
import sys
import argparse
from compiler import Compiler, CompileError, EC
from vm import Machine

logger = logging.getLogger(__name__)

# we are going to create this phony test case, so that we can use its
# assert* methods, which are much nicer than the vanilla assert
# statement.
tc = unittest.TestCase()

class TestArith1:
    code = """
x& = 100
let y& = 200

print x& * (y& + 1) / 10
    """

    cevents = []
    vevents = [
        ('print', ' 2010 \n')
    ]


class TestArith2:
    code = """
print 2 + 3 * 4
    """

    cevents = []
    vevents = [
        ('print', ' 14 \n')
    ]


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


class TestPrint2:
    code = """
print -10
    """

    cevents = []
    vevents = [
        ('print', '-10 \n'),
    ]


class TestPrint3:
    code = """
print 0
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
    ]


class TestPrint4:
    code = """
print "foo"
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestPrint5:
    code = """
print 1, 2
    """

    cevents = []
    vevents = [
        ('print', ' 1             2 \n'),
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


class TestArray1:
    code = """
dim x(20) as long

for i% = 1 to 20
    x(i%) = i% * 10 + 1
next

for i% = 1 to 20
    print x(i%)
next
    """

    cevents = []
    vevents = [
        ('print', f' {i * 10 + 1} \n') for i in range(1, 21)
    ]


class TestArray2:
    code = """
dim x(1 TO 3, 3 TO 5) as long

x(1, 3) = 100
x(3, 3) = 200
x(3, 5) = 300

print x(1, 3); x(3, 3); x(3, 5)
    """

    cevents = []
    vevents = [
        ('print', ' 100  200  300 \n'),
    ]


class TestArray3:
    code = """
sub alter(x as long)
    x = x * 2
end sub

dim x(5) as long

for i = 1 to 5
    x(i) = i
next

for i = 1 to 5
    alter x(i)
    print x(i)
next
    """

    cevents = []
    vevents = [
        ('print', ' 2 \n'),
        ('print', ' 4 \n'),
        ('print', ' 6 \n'),
        ('print', ' 8 \n'),
        ('print', ' 10 \n'),
    ]


class TestType1:
    code = """
type foo
    x as integer
    y as long
    z
end type

dim a as foo

a.x = 100
a.y = 200
a.z = 300

print a.x=100; a.y=200; a.z=300
    """

    cevents = []
    vevents = [
        ('print', '-1 -1 -1 \n'),
    ]


class TestType2:
    code = """
type foo
    x as integer
    y as long
    z as integer
end type

type bar
    f as foo
    g as long
end type

dim a as foo
dim b as bar

a.x = 100
a.y = 200
a.z = 300
b.f = a

print b.f.x; b.f.y; b.f.z
    """

    cevents = []
    vevents = [
        ('print', ' 100  200  300 \n'),
    ]

class TestType3:
    code = """
type foo
    x as integer
    y as long
    z as integer
end type

type bar
    f as foo
    g as long
end type

type spam
    eggs as bar
    something as double
end type

dim a as spam
a.eggs.f.y = 100

print a.eggs.f.y
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n'),
    ]


class TestType4:
    code = """
type foo
    x as integer
    y as long
    z as integer
end type

type bar
    f as foo
    g as long
end type

sub alter(x as foo, y as long)
    x.y = y
end sub

dim a as foo
dim b as bar

alter a, 100
alter b.f, 200

print a.y; b.f.y
    """

    cevents = []
    vevents = [
        ('print', ' 100  200 \n'),
    ]


class TestType5:
    code = """
type foo
    x as integer
    y as long
    z as integer
end type

type bar
    f as foo
    g as long
end type

sub alter(x as foo, y as long)
    dim a as foo
    a.y = y
    x = a
end sub

dim a as foo
dim b as bar

alter a, 100
alter b.f, 200

print a.y; b.f.y
    """

    cevents = []
    vevents = [
        ('print', ' 100  200 \n'),
    ]


class TestType6:
    code = """
type foo
    x as integer
    y as long
    z as integer
end type

type bar
    f as foo
    g as long
end type

sub alter(x as foo, y as long)
    dim a as foo
    a.y = y
    x = a
end sub

dim a(10) as foo
dim b(10) as bar

alter a(5), 100
alter b(5).f, 200

print a(5).y; b(5).f.y
    """

    cevents = []
    vevents = [
        ('print', ' 100  200 \n'),
    ]


class TestExitSub1:
    code = """
sub foo(n as integer)
    n = 100
    exit sub
    n = 200
end sub

foo n%
print n%
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n')
    ]


class TestExitSub2:
    code = """
exit sub
    """

    cevents = [
        ('error', EC.EXIT_SUB_INVALID)
    ]
    vevents = []


class TestExitSub3:
    code = """
function foo
    exit sub
end function
    """

    cevents = [
        ('error', EC.EXIT_SUB_INVALID)
    ]
    vevents = []


class TestExitFunc1:
    code = """
function foo%
    foo = 100
    exit function
    foo = 200
end function

n% = foo
print n%
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n')
    ]


class TestExitFunc2:
    code = """
exit function
    """

    cevents = [
        ('error', EC.EXIT_FUNC_INVALID)
    ]
    vevents = []


class TestExitFunc3:
    code = """
sub foo
    exit function
end sub
    """

    cevents = [
        ('error', EC.EXIT_FUNC_INVALID)
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


def get_all_tests():
    return {
        name: value
        for name, value in globals().items()
        if name.startswith('Test') and isinstance(value, type)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run qpybasic tests.')

    parser.add_argument(
        'test_case', nargs='*',
        help='The test case(s) to run. Any number of test cases can be '
        'passed. Defaults to running all tests.')

    args = parser.parse_args()

    test_cases = get_all_tests()
    if args.test_case != []:
        test_cases = {
            name: value
            for name, value in test_cases.items()
            if name in args.test_case
        }

        if len(test_cases) != len(args.test_case):
            not_found = set(args.test_case) - set(get_all_tests())
            print('The following test case(s) not found:')
            for i in not_found:
                print(f'    {i}')
            exit(1)

    failed = []
    success = []

    print(f'Running {len(test_cases)} test case(s).')
    for name, value in test_cases.items():
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
