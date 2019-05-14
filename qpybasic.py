from compiler import Compiler


with open('qpybasic.ebnf') as f:
    grammar_text = f.read()

prog = r"""
declare function fib%(n%)

for i% = 1 to 10
   print "fib("; i%; " ) ="; fib%(i%)
next

function fib%(n%)
   if n% < 3 then
      fib% = 1
   else
      x1% = fib%(n%-1)
      x2% = fib%(n%-2)
      fib% = x1% + x2%
   end if
end function

"""

c = Compiler()
module = c.compile(prog + '\n')
for i in c.instrs:
    print(i)
with open('foo.mod', 'wb') as f:
    f.write(module.dump())
