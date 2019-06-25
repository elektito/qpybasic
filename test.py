#!/usr/bin/env python3

import logging
import traceback
import unittest
import sys
import argparse
from compiler import Compiler, CompileError, EC
from vm import Machine, RE

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


class TestCallSub7:
    code = """
sub foo(msg as string)
   print msg
   msg = "bar"
end sub

m$ = "foo"
call foo(m$)
print m$
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
        ('print', 'bar\n'),
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
   print "fib"; i; r%
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


class TestFunctionCall1:
    code = """
function foo(x as string)
    print x
    x = "bar"
end function

m$ = "foo"
r = foo(m$)
print m$
m$ = "spam"
print m$
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
        ('print', 'bar\n'),
        ('print', 'spam\n'),
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


class TestArray4:
    code = """
dim x(10 to 14) as integer

for i = 10 to 14
    print x(i)
next
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
        ('print', ' 0 \n'),
        ('print', ' 0 \n'),
        ('print', ' 0 \n'),
        ('print', ' 0 \n'),
    ]


class TestArray5:
    code = """
dim x(10) as integer

print x("foo")
    """

    cevents = [
        ('error', EC.TYPE_MISMATCH)
    ]
    vevents = []


class TestArray6:
    code = """
dim x(10) as integer

print x(11)
    """

    cevents = []
    vevents = [
        ('error', RE.SUBSCRIPT_OUT_OF_RANGE)
    ]


class TestArray7:
    code = """
dim x(5 to 10) as integer

x(4) = 1
    """

    cevents = []
    vevents = [
        ('error', RE.SUBSCRIPT_OUT_OF_RANGE)
    ]


class TestArray8:
    code = """
n = 1
m = 3
p = 2
dim x(n TO m, m TO m+p) as long

x(1, 3) = 100
x(3, 3) = 200
x(3, 5) = 300

print x(1, 3); x(3, 3); x(3, 5)
    """

    cevents = []
    vevents = [
        ('print', ' 100  200  300 \n'),
    ]


class TestArray9:
    code = """
n& = 5
dim x(n& * 2) as long

for i% = 1 to n& * 2
    x(i%) = i% * 10 + 1
next

for i% = 1 to n& * 2
    print x(i%)
next
    """

    cevents = []
    vevents = [
        ('print', f' {i * 10 + 1} \n') for i in range(1, 11)
    ]


class TestArray10:
    code = """
n! = 5.0
dim x(n! * 2) as long

for i% = 1 to n! * 2
    x(i%) = i% * 10 + 1
next

for i% = 1 to n! * 2
    print x(i%)
next
    """

    cevents = []
    vevents = [
        ('print', f' {i * 10 + 1} \n') for i in range(1, 11)
    ]


class TestArray11:
    code = """
dim x("foo") as long
    """

    cevents = [
        ('error', EC.TYPE_MISMATCH)
    ]
    vevents = []


class TestArray12:
    code = """
n$ = "foo"
dim x(n$) as long
    """

    cevents = [
        ('error', EC.TYPE_MISMATCH)
    ]
    vevents = []


class TestArray13:
    code = """
dim x(0) as long
    """

    cevents = [
        ('error', EC.INVALID_ARRAY_BOUNDS)
    ]
    vevents = []


class TestArray14:
    code = """
dim x(1 to -1) as long
    """

    cevents = [
        ('error', EC.INVALID_ARRAY_BOUNDS)
    ]
    vevents = []


class TestArray15:
    code = """
defint a-z
dim x(10)
x(1) = 100 + i
    """

    cevents = []
    vevents = []


class TestArray16:
    code = """
dim x(2) as integer, y(3) as integer
x(1) = 10
x(2) = 20
y(1) = 11
y(2) = 22
y(3) = 33

print x(1); x(2); y(1); y(2); y(3)
    """

    cevents = []
    vevents = [
        ('print', ' 10  20  11  22  33 \n'),
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


class TestType6:
    code = """
type foo
    x as integer
    y as long
    z as single
    w as double
end type

dim x as foo

print x.x=0; x.y=0; x.z=0; x.w=0
    """

    cevents = []
    vevents = [
        ('print', '-1 -1 -1 -1 \n'),
    ]


class TestType7:
    code = """
type foo
    x as string
end type
    """

    cevents = [
        ('error', EC.STRING_NOT_ALLOWED_IN_TYPE),
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


class TestExitFor1:
    code = """
for i% = 1 to 10
    print i%
    if i% = 5 then
       exit for
    end if
next
    """

    cevents = []
    vevents = [
        ('print', ' 1 \n'),
        ('print', ' 2 \n'),
        ('print', ' 3 \n'),
        ('print', ' 4 \n'),
        ('print', ' 5 \n'),
    ]


class TestExitFor2:
    code = """
for i% = 1 to 10
    for j% = 1 to 2
        if i% > 3 then
           exit for
        end if
        print i%; j%
    next
    print "foo"
    if i% = 5 then
       exit for
    end if
next
    """

    cevents = []
    vevents = [
        ('print', ' 1  1 \n'),
        ('print', ' 1  2 \n'),
        ('print', 'foo\n'),
        ('print', ' 2  1 \n'),
        ('print', ' 2  2 \n'),
        ('print', 'foo\n'),
        ('print', ' 3  1 \n'),
        ('print', ' 3  2 \n'),
        ('print', 'foo\n'),
        ('print', 'foo\n'),
        ('print', 'foo\n'),
    ]


class TestExitFor3:
    code = """
exit for
    """

    cevents = [
        ('error', EC.EXIT_FOR_INVALID)
    ]
    vevents = []


class TestGosub1:
    code = """
i% = 10
gosub pr
i% = 20
gosub pr
end

pr:
   print i%+1
   return
    """

    cevents = []
    vevents = [
        ('print', ' 11 \n'),
        ('print', ' 21 \n'),
    ]


class TestGosub2:
    code = """
recs% = 0
result% = 0
i% = 10
gosub pr
print result%

end

pr:
   result% = result% + i%
   recs% = recs% + 1
   if recs% = 5 then
      return
   end if
   gosub pr
   return
    """

    cevents = []
    vevents = [
        ('print', ' 50 \n'),
    ]


class TestConst1:
    code = """
const x& = 100
print x&
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n'),
    ]


class TestConst2:
    code = """
const x! = 100 - 20
const y& = (2 * x!) / 4 + 1
print y&
    """

    cevents = []
    vevents = [
        ('print', ' 41 \n'),
    ]


class TestConst3:
    code = """
y = 100
const x = y
    """

    cevents = [
        ('error', EC.INVALID_CONSTANT)
    ]
    vevents = []


class TestConst4:
    code = """
const x = 10
x = 11
    """

    cevents = [
        ('error', EC.CANNOT_ASSIGN_TO_CONST)
    ]
    vevents = []


class TestConst5:
    code = """
const x& = 100.5
print x&
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n'),
    ]


class TestVars1:
    code = """
dim x as string
x = "foo"
print x
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestVars2:
    code = """
dim x as string
x = "foo"
print x$
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestVars3:
    code = """
dim x as string
x$ = "foo"
print x
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestVars4:
    code = """
dim x as string
x$ = "foo"
x& = 100
    """

    cevents = [
        ('error', EC.DUP_DEF)
    ]
    vevents = []


class TestVars5:
    code = """
x$ = "foo"
dim x as string
    """

    cevents = [
        ('error', EC.AS_CLAUSE_REQUIRED_ON_FIRST_DECL)
    ]
    vevents = []


class TestVars6:
    code = """
x$ = "foo"
dim x$
    """

    cevents = [
        ('error', EC.DUP_DEF)
    ]
    vevents = []


class TestVars7:
    code = """
dim x
dim x as single
    """

    cevents = [
        ('error', EC.AS_CLAUSE_REQUIRED_ON_FIRST_DECL)
    ]
    vevents = []


class TestVars8:
    code = """
x$ = "foo"
dim x as single
    """

    cevents = [
        ('error', EC.DUP_DEF)
    ]
    vevents = []


class TestVars9:
    code = """
dim x as string
x = 100
    """

    cevents = [
        ('error', EC.TYPE_MISMATCH)
    ]
    vevents = []


class TestVars10:
    code = """
x$ = "foo"
x& = 100
x% = 200

print x$; x&; x%
    """

    cevents = []
    vevents = [
        ('print', 'foo 100  200 \n')
    ]


class TestVars11:
    code = """
dim x as string, y as string
x = "foo"
y = "bar"
print x; y
    """

    cevents = []
    vevents = [
        ('print', 'foobar\n')
    ]


class TestArrayPass1:
    code = """
sub foo(x() as integer, y)
    print x(5)
    x(5) = y
end sub

dim x(10) as integer
x(5) = 10
foo x(), 100
print x(5)
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
        ('print', ' 100 \n'),
    ]


class TestArrayPass2:
    code = """
function foo(x() as integer, y)
    print x(5)
    x(5) = y
end function

dim x(10) as integer
x(5) = 10
ret = foo(x(), 100)
print x(5)
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
        ('print', ' 100 \n'),
    ]


class TestArrayPass3:
    code = """
declare sub foo(x() as integer, y)

dim x(10) as integer
x(5) = 10
foo x(), 100
print x(5)

sub foo(x() as integer, y)
    print x(5)
    x(5) = y
end sub
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
        ('print', ' 100 \n'),
    ]


class TestArrayPass4:
    code = """
declare function foo(x() as integer, y)

dim x(10) as integer
x(5) = 10
ret = foo(x(), 100)
print x(5)

function foo(x() as integer, y)
    print x(5)
    x(5) = y
end function
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
        ('print', ' 100 \n'),
    ]


class TestArrayPass5:
    code = """
sub foo(x() as integer)
    x(5) = y
end sub

dim x(10) as integer
foo x
    """

    cevents = [
        ('error', EC.TYPE_MISMATCH)
    ]
    vevents = []


class TestLogical1:
    code = """
print not 1800
print 199001 and 2333
print 49 or 87
print 2300 xor 1877
print 199 eqv 44
print 999 imp 777
print 2291 or 4344 and not 1616 eqv 209 imp 7004
print 100 xor 22.29
    """

    cevents = []
    vevents = [
        ('print', '-1801 \n'),
        ('print', ' 2329 \n'),
        ('print', ' 119 \n'),
        ('print', ' 4009 \n'),
        ('print', '-236 \n'),
        ('print', '-231 \n'),
        ('print', ' 7038 \n'),
        ('print', ' 114 \n'),
    ]


class TestComments1:
    code = """
'print 10
REM print 20
REM xyz123;foobar:print&xz!!@'foo
    """

    cevents = []
    vevents = []


class TestComments2:
    code = """
type foo
    x as integer   ' defines the x field
    y as long      ' defines the y field
end type

dim a as foo
a.x = 100
a.y = 200
print a.x; a.y
    """

    cevents = []
    vevents = [
        ('print', ' 100  200 \n'),
    ]


class TestComments3:
    code = """
if 2 > 1 then   ' condition
    print "2>1" ' if part
else            ' else line
    print "2<1" ' else part
end if          ' end of if block!
    """

    cevents = []
    vevents = [
        ('print', '2>1\n'),
    ]


class TestComments4:
    code = """
sub foo(x as long)   ' begin sub
    print x          ' print arg
end sub              ' end the sub

REM call the sub
foo 100
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n'),
    ]


class TestComments5:
    code = """
function foo&(x as long)   ' begin function
    foo& = x + 100         ' set return value
end function               ' end the function

REM call the function
print foo&(100)
    """

    cevents = []
    vevents = [
        ('print', ' 200 \n'),
    ]


class TestComments6:
    code = """
print 10   ' prints 10
print 20: rem print 30 : print 40
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
        ('print', ' 20 \n'),
    ]


class TestShared1:
    code = """
declare sub foo

dim shared a(100) as integer
dim shared x as long

x = 100
foo

sub foo
    print x
end sub
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n'),
    ]


class TestShared2:
    code = """
declare sub foo

dim shared a(5) as integer
dim shared x as long

x = 100
foo

for i = 1 to 5
    print a(i)
next

sub foo
    print x

    for i = 1 to 5
        a(i) = i * 10
    next
end sub
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n'),
        ('print', ' 10 \n'),
        ('print', ' 20 \n'),
        ('print', ' 30 \n'),
        ('print', ' 40 \n'),
        ('print', ' 50 \n'),
    ]


class TestShared3:
    code = """
declare sub foo

dim shared a(5) as integer, x as long

x = 100
foo

for i = 1 to 5
    print a(i)
next

sub foo
    print x

    for i = 1 to 5
        a(i) = i * 10
    next
end sub
    """

    cevents = []
    vevents = [
        ('print', ' 100 \n'),
        ('print', ' 10 \n'),
        ('print', ' 20 \n'),
        ('print', ' 30 \n'),
        ('print', ' 40 \n'),
        ('print', ' 50 \n'),
    ]


class TestDoLoop1:
    code = """
defint x
do while x < 5
    print x
    x = x + 1
loop
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
        ('print', ' 1 \n'),
        ('print', ' 2 \n'),
        ('print', ' 3 \n'),
        ('print', ' 4 \n'),
    ]


class TestDoLoop2:
    code = """
defint x
do while x > 5
    print x
    x = x + 1
loop
    """

    cevents = []
    vevents = []


class TestDoLoop3:
    code = """
defint x
do until x = 5
    print x
    x = x + 1
loop
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
        ('print', ' 1 \n'),
        ('print', ' 2 \n'),
        ('print', ' 3 \n'),
        ('print', ' 4 \n'),
    ]


class TestDoLoop4:
    code = """
defint x
do until x = 0
    print x
    x = x + 1
loop
    """

    cevents = []
    vevents = []


class TestDoLoop5:
    code = """
defint x
do
    print x
    x = x + 1
loop while x < 5
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
        ('print', ' 1 \n'),
        ('print', ' 2 \n'),
        ('print', ' 3 \n'),
        ('print', ' 4 \n'),
    ]


class TestDoLoop6:
    code = """
defint x
do
    print x
    x = x + 1
loop while x = 5
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
    ]


class TestDoLoop7:
    code = """
defint x
do
    print x
    x = x + 1
loop until x = 5
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
        ('print', ' 1 \n'),
        ('print', ' 2 \n'),
        ('print', ' 3 \n'),
        ('print', ' 4 \n'),
    ]


class TestDoLoop8:
    code = """
defint x
do
    print x
    x = x + 1
loop until x < 5
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
    ]


class TestDoLoop9:
    code = """
defint x
do
    do
        print x
        x = x + 1
    loop until x < 5
loop until -1
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
    ]


class TestDoLoop10:
    code = """
defint x
do
    print "foo"
    exit do
    print "bar"
loop
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
    ]


class TestDoLoop11:
    code = """
exit do
    """

    cevents = [
        ('error', EC.EXIT_DO_INVALID)
    ]
    vevents = []


class TestDoLoop12:
    code = """
for i = 1 to 10
    exit do
next
    """

    cevents = [
        ('error', EC.EXIT_DO_INVALID)
    ]
    vevents = []


class TestDoLoop13:
    code = """
defint x
do
    do
        print "foo"
        exit do
        print "bar"
    loop
    print "spam"
loop until x = 0
    """

    cevents = []
    vevents = [
        ('print', 'foo\n'),
        ('print', 'spam\n'),
    ]


class TestNumericLiterals1:
    code = """
x = .1
y = 0.1

print x = y
    """

    cevents = []
    vevents = [
        ('print', '-1 \n'),
    ]


class TestNumericLiterals2:
    code = """
x% = 100%
y& = 1000000&
z! = 1.1!
w# = 2.2#
    """

    cevents = []
    vevents = []


class TestNumericLiterals3:
    code = """
print &hff; &ha0; &h0; &h1; &h002
    """

    cevents = []
    vevents = [
        ('print', ' 255  160  0  1  2 \n'),
    ]


class TestNumericLiterals4:
    code = """
x% = 32768%
    """

    cevents = [
        ('error', EC.ILLEGAL_NUMBER),
    ]
    vevents = []


class TestNumericLiterals5:
    code = """
' Technically this number should (-2^16) be valid, but due to the way
' negative numbers are implemented (as a positive number, negated),
' this won't work. Good thing is, the same is true for QB itself!
x% = -32768%
    """

    cevents = [
        ('error', EC.ILLEGAL_NUMBER),
    ]
    vevents = []


class TestNumericLiterals6:
    code = """
x& = 2147483648&
    """

    cevents = [
        ('error', EC.ILLEGAL_NUMBER),
    ]
    vevents = []


class TestNumericLiterals7:
    code = """
' Technically this number should (-2^31) be valid, but due to the way
' negative numbers are implemented (as a positive number, negated),
' this won't work. Good thing is, the same is true for QB itself!
x& = -2147483648&
    """

    cevents = [
        ('error', EC.ILLEGAL_NUMBER),
    ]
    vevents = []


class TestIf1:
    code = """
if 2 > 1 then
    print 10
else
    print 20
end if
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
    ]


class TestIf2:
    code = """
if 1 > 2 then
    print 10
else
    print 20
end if
    """

    cevents = []
    vevents = [
        ('print', ' 20 \n'),
    ]


class TestIf3:
    code = """
if 2 > 1 then
    print 10
end if
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
    ]


class TestIf4:
    code = """
if 1 > 2 then
    print 10
end if
    """

    cevents = []
    vevents = []


class TestIf5:
    code = """
if 2 > 1 then print 10 else print 20
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
    ]


class TestIf6:
    code = """
if 1 > 2 then print 10 else print 20
    """

    cevents = []
    vevents = [
        ('print', ' 20 \n'),
    ]


class TestIf7:
    code = """
if 2 > 1 then print 10
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
    ]


class TestIf8:
    code = """
if 1 > 2 then print 10
    """

    cevents = []
    vevents = []


class TestIf9:
    code = """
if 2 > 1 then print 10 : print 20
    """

    cevents = []
    vevents = [
        ('print', ' 10 \n'),
        ('print', ' 20 \n'),
    ]


class TestIf10:
    code = """
if 1 > 2 then print 10 : print 20
    """

    cevents = []
    vevents = []


class TestIf11:
    code = """
if 1 > 2 then print 10 : print 20 else print 30 : print 40
    """

    cevents = []
    vevents = [
        ('print', ' 30 \n'),
        ('print', ' 40 \n'),
    ]


class TestIf12:
    code = """
if 1 = 1 then
    print "EQ"
else
    print "NEQ"
end if
    """

    cevents = []
    vevents = [
        ('print', 'EQ\n'),
    ]


class TestIf13:
    code = """
if 1 = 2 then
    print "EQ"
else
    print "NEQ"
end if
    """

    cevents = []
    vevents = [
        ('print', 'NEQ\n'),
    ]


class TestIf14:
    code = """
if 1 = 2 then
    print "EQ1"
elseif 2 = 2 then
    print "EQ2"
else
    print "NEQ"
end if
    """

    cevents = []
    vevents = [
        ('print', 'EQ2\n'),
    ]


class TestIf15:
    code = """
if 1 = 2 then
    print "EQ1"
elseif 2 = 3 then
    print "EQ2"
else
    print "NEQ"
end if
    """

    cevents = []
    vevents = [
        ('print', 'NEQ\n'),
    ]


class TestIf16:
    code = """
if 1 = 2 then
    print "EQ1"
elseif 2 = 3 then
    print "EQ2"
elseif 3 = 3 then
    print "EQ3"
else
    print "NEQ"
end if
    """

    cevents = []
    vevents = [
        ('print', 'EQ3\n'),
    ]


class TestIf17:
    code = """
if 1 = 2 then
    print "EQ1"
elseif 2 = 3 then
    print "EQ2"
elseif 3 = 3 then
    print "EQ3"
else
    print "NEQ"
end if
    """

    cevents = []
    vevents = [
        ('print', 'EQ3\n'),
    ]


class TestViewPrint1:
    code = """
view print
    """

    cevents = []
    vevents = [
        ('view_print', -1, -1),
    ]


class TestViewPrint2:
    code = """
view print 10 to 20
    """

    cevents = []
    vevents = [
        ('view_print', 10, 20),
    ]


class TestMod1:
    code = """
print 20 * 11 mod 4
    """

    cevents = []
    vevents = [
        ('print', ' 0 \n'),
    ]


class TestMod2:
    code = """
print 20 + 11 mod 4
    """

    cevents = []
    vevents = [
        ('print', ' 23 \n'),
    ]


class TestAbs1:
    code = """
x% = 100
y% = -100

print abs(x%); abs(y%)
    """

    cevents = []
    vevents = [
        ('print', ' 100  100 \n'),
    ]


class TestAbs2:
    code = """
x& = 100
y& = -100

print abs(x&); abs(y&)
    """

    cevents = []
    vevents = [
        ('print', ' 100  100 \n'),
    ]


class TestAbs3:
    code = """
x! = 100
y! = -100

print abs(x!); abs(y!)
    """

    cevents = []
    vevents = [
        ('print', ' 100.0  100.0 \n'),
    ]


class TestAbs4:
    code = """
x# = 100
y# = -100

print abs(x#); abs(y#)
    """

    cevents = []
    vevents = [
        ('print', ' 100.0  100.0 \n'),
    ]


class TestAbs5:
    code = """
const x% = 100
const y% = -100

print abs(x%); abs(y%)
    """

    cevents = []
    vevents = [
        ('print', ' 100  100 \n'),
    ]


class TestString1:
    code = """
x$ = "foo"
y$ = x$
y$ = "bar"

print x$; y$
    """

    cevents = []
    vevents = [
        ('print', 'foobar\n'),
    ]


class TestString2:
    code = """
dim x(5) as string

x(1) = "foo1"
x(2) = "foo2"
x(3) = "foo3"
x(4) = "foo4"
x(5) = "foo5"

for i = 1 to 5
    print x(i)
next
    """

    cevents = []
    vevents = [
        ('print', 'foo1\n'),
        ('print', 'foo2\n'),
        ('print', 'foo3\n'),
        ('print', 'foo4\n'),
        ('print', 'foo5\n'),
    ]


class TestInput1:
    code = """
input x$
print x$
    """

    input_lines = [
        'foo',
    ]

    cevents = []
    vevents = [
        ('print', '? '),
        ('input',),
        ('print', 'foo\n'),
    ]


class TestInput2:
    code = """
input x$, y$
print x$; y$
    """

    input_lines = [
        'foo, bar',
    ]

    cevents = []
    vevents = [
        ('print', '? '),
        ('input',),
        ('print', 'foobar\n'),
    ]


class TestInput3:
    code = """
input x$, y$
print x$; y$
    """

    input_lines = [
        'foo',
        'foo,bar,buz',
        'foo,bar',
    ]

    cevents = []
    vevents = [
        ('print', '? '),
        ('input',),
        ('print', 'Redo from start\n'),
        ('print', '? '),
        ('input',),
        ('print', 'Redo from start\n'),
        ('print', '? '),
        ('input',),
        ('print', 'foobar\n'),
    ]


class TestInput4:
    code = """
input n%, x!
print n%; x!
    """

    input_lines = [
        '100,foo',
        'foo,1.1',
        '100000,1.1',
        '100,-1.25',
    ]

    cevents = []
    vevents = [
        ('print', '? '),
        ('input',),
        ('print', 'Redo from start\n'),
        ('print', '? '),
        ('input',),
        ('print', 'Redo from start\n'),
        ('print', '? '),
        ('input',),
        ('print', 'Overflow\n'),
        ('print', 'Redo from start\n'),
        ('print', '? '),
        ('input',),
        ('print', ' 100 -1.25 \n'),
    ]


class TestInput5:
    code = """
input "foo"; x$, y$
print x$; y$
    """

    input_lines = [
        'foo, bar'
    ]

    cevents = []
    vevents = [
        ('print', 'foo? '),
        ('input',),
        ('print', 'foobar\n'),
    ]


class TestInput6:
    code = """
input "foo", x$, y$
print x$; y$
    """

    input_lines = [
        'foo, bar'
    ]

    cevents = []
    vevents = [
        ('print', 'foo'),
        ('input',),
        ('print', 'foobar\n'),
    ]


class TestStaticRoutines1:
    code = """
defint a-z
sub foo static
    n = n + 1
    print n
end sub

for i = 1 to 5
    foo
next
    """

    cevents = []
    vevents = [
        ('print', ' 1 \n'),
        ('print', ' 2 \n'),
        ('print', ' 3 \n'),
        ('print', ' 4 \n'),
        ('print', ' 5 \n'),
    ]


class TestStaticRoutines2:
    code = """
defint a-z
function foo static
    n = n + 1
    foo = n
end function

for i = 1 to 5
    print foo
next
    """

    cevents = []
    vevents = [
        ('print', ' 1 \n'),
        ('print', ' 2 \n'),
        ('print', ' 3 \n'),
        ('print', ' 4 \n'),
        ('print', ' 5 \n'),
    ]


class TestStrCompare1:
    code = """
x$ = "abc"
y$ = "foo"
z$ = "bar"
w$ = "foo"
s$ = "foobar"

print x$ = y$; x$ < y$; x$ > y$
print z$ <= y$; z$ >= y$
print y$ = w$; y$ <> w$; y$ < w$; y$ > w$; y$ <= w$; y$ >= w$
print w$ < s$; w$ <= s$; w$ >= s$
    """

    cevents = []
    vevents = [
        ('print', ' 0 -1  0 \n'),
        ('print', '-1  0 \n'),
        ('print', '-1  0  0  0 -1 -1 \n'),
        ('print', '-1 -1  0 \n'),
    ]


class TestRegression1:
    # This used to crash upon compilation because of a bug in constant
    # folding logic.
    code = """
n = 1 - 0
    """

    cevents = []
    vevents = []


class TestRegression2:
    # This used to print all values of y as zero. The reason for that
    # was that initialization of the y variable was inside the
    # loop. This did not happen to the x variable since it was used in
    # the loop condition and the condition was evaluated before loop
    # code was generated, so the initialization happened outside the
    # loop.
    code = """
defint x-y
do
    print x; y
    x = x + 1
    y = y + 1
loop while x < 5
    """

    cevents = []
    vevents = [
        ('print', ' 0  0 \n'),
        ('print', ' 1  1 \n'),
        ('print', ' 2  2 \n'),
        ('print', ' 3  3 \n'),
        ('print', ' 4  4 \n'),
    ]


class TestSelect1:
    code = """
n = 5
z = 0
select case n + 5
case 5 + z
     print "five"
case 10 + z
     print "ten"
end select
    """

    cevents = []
    vevents = [
        ('print', 'ten\n'),
    ]


class TestSelect2:
    code = """
n = 5
z = 0
select case n + 5
foo:
case 5 + z
     print "five"
case 10 + z
     print "ten"
end select
    """

    cevents = [
        ('error', EC.SYNTAX_ERROR),
    ]
    vevents = []


class TestSelect3:
    code = """
n = 5
z = 0
select case n + 5
print 1
case 5 + z
     print "five"
case 10 + z
     print "ten"
end select
    """

    cevents = [
        ('error', EC.SYNTAX_ERROR),
    ]
    vevents = []


class TestSelect4:
    code = """
n = 5
z = 0
select case n + 5
case 100
     print "five"
case 200
     print "ten"
end select
    """

    cevents = []
    vevents = []


class TestSelect5:
    code = """
n = 5
z = 0
select case n + 5
case 10 + z
     print "ten"
case 10 + z
     print "another ten"
end select
    """

    cevents = []
    vevents = [
        ('print', 'ten\n'),
    ]


def run_test_case(name, case, optimization=0):
    events = []
    input_idx = 0
    def event_handler(event):
        nonlocal input_idx
        events.append(event)
        if event[0] == 'input':
            input_idx += 1
            return case.input_lines[input_idx - 1]

    logger.info(f'Running test case: {name}')

    c = Compiler(optimization=optimization)
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

    for olevel in [0, 1]:
        print(f'Running {len(test_cases)} test case(s) at optimization level {olevel}...')
        for name, value in test_cases.items():
            try:
                run_test_case(name, value, optimization=olevel)
            except Exception as e:
                failed.append((name, e, olevel))
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
        for name, exc, olevel in failed:
            print(f'Failed test case: {name}')
            print(f'Optimization level: {olevel}')
            print('Exception:')
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        print('---\n')
        total = len(failed) + len(success)
        print(f'{len(failed)} out of {total} test case(s) failed.')


if __name__ == '__main__':
    main()
