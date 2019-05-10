from compiler import Compiler


with open('qpybasic.ebnf') as f:
    grammar_text = f.read()

prog = r"""
start:
x = 100:let y = 200
z$ = "foo"
z2$ = "bar"
foo$ = z$ + z2$

if x < y then
    print "less"
else
    print "more"
end if

xyz: cls
print
10 print x; y; -(x + y*2), foo$
print 1, 2, z

for i = 1 to 10
   print "boo"
next

goto 10
end
"""
prog = r"""
sub foo(x as string, y as integer, z&)
   print "sub_foo:"; z&
   y = 190
end sub

sub bar(n as integer, r as long)
   print n, r
end sub

foo "foobar", x%, 12000
print "main:"; x%
"""

prog = r"""
sub fib(n as integer, r as long)
   if n <= 2 then
      r = 1
   else
      fib n-1, r1&
      fib n-2, r2&
      r = r1& + r2&
   end if
end sub

for i% = 1 to 10
   call fib(i%, r&)
   print "fib", i%, r&
next
"""

prog = r"""
if 1 then
   print 100
end if
"""

c = Compiler()
module = c.compile(prog + '\n')
for i in c.instrs:
    print(i)
with open('foo.mod', 'wb') as f:
    f.write(module.dump())
