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
c = Compiler()
module = c.compile(prog)
for i in c.instrs:
    print(i)
with open('foo.mod', 'wb') as f:
    f.write(module.dump())
