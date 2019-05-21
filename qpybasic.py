from compiler import Compiler, CompileError


with open('qpybasic.ebnf') as f:
    grammar_text = f.read()

prog = r"""
'sample code
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

c = Compiler()
try:
    module = c.compile(prog)
except CompileError as e:
    print('COMPILE ERROR:', e)
    exit(1)

for i in c.instrs:
    print(i)

with open('foo.mod', 'wb') as f:
    f.write(module.dump())
