from compiler import Compiler


with open('qpybasic.ebnf') as f:
    grammar_text = f.read()

prog = r"""
declare function fib%(n%)

print fib%(3 )

function fib%(n%)
   if n% < 3 then
      fib% = 1
   else
      fib% = fib%(1) + fib%(1)
   end if
end function

"""

c = Compiler()
module = c.compile(prog + '\n')
for i in c.instrs:
    print(i)
with open('foo.mod', 'wb') as f:
    f.write(module.dump())
