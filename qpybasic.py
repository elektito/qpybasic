from lark import Lark, Transformer

class MyT(Transformer):
    def __init__(self):
        self.variables = {}

    def program(self, items):
        return items

    def line(self, items):
        if items:
            return items[0]

    def stmt(self, items):
        return items[0]

    def cls_stmt(self, items):
        seq =  '\033[2J'    # clear screen
        seq += '\033[1;1H'  # move cursor to screen top-left
        print(seq)
        return items

    def end_stmt(self, items):
        print('END!')

    def let_stmt(self, items):
        if len(items) == 3:
            items = items[1:]
        var, value = items
        print(":: setting {} to {} ({})".format(var, value, type(value).__name__))
        self.variables[var.value] = value
        return 'let {} = {}'.format(var, value)

    def print_stmt(self, items):
        items = items[1:]
        if not items:
            line = '\n'
        else:
            line = ''

        while items:
            i, items = items[0], items[1:]
            if type(i) == int:
                if i < 0:
                    p = '{} '.format(i)
                else:
                    p = ' {} '.format(i)
            else:
                p = i

            if items:
                sep, items = items[0], items[1:]
            else:
                sep = None

            line += p
            if sep == ';':
                pass
            elif sep == ',':
                n = 14 - (len(line) % 14)
                line += n * ' '
            else:
                line += '\n'

        print(line, end='')


    def expr(self, items):
        return items[0]

    def expr_add(self, items):
        x, y = items
        if type(x) == int and type(y) == int:
            return x + y
        elif type(x) == str and type(y) == str:
            return x + y
        else:
            raise RuntimeError('Type mismatch.')

    def expr_sub(self, items):
        x, y = items
        if type(x) == int and type(y) == int:
            return x - y
        else:
            raise RuntimeError('Type mismatch.')

    def mult_expr(self, items):
        return items[0]

    def expr_mult(self, items):
        x, y = items
        if type(x) == int and type(y) == int:
            return x * y
        else:
            raise RuntimeError('Type mismatch.')

    def expr_div(self, items):
        x, y = items
        if type(x) == int and type(y) == int:
            return x // y
        else:
            raise RuntimeError('Type mismatch.')

    def unary_expr(self, items):
        return items[0]

    def value(self, items):
        item, = items
        if type(item) in [int, str]:
            return item
        elif item.type == 'INT_LITERAL':
            return int(item.value)
        elif item.type == 'STRING_LITERAL':
            return item.value[1:-1]

    def var(self, items):
        varname = items[0].value
        if varname in self.variables:
            return self.variables[varname]
        else:
            raise RuntimeError('Undefined variable: {}'.format(varname))

    def negation(self, items):
        x, = items
        if type(x) == int:
            return -items[0]
        else:
            raise RuntimeError('Type mismatch.')

with open('qpybasic.ebnf') as f:
    grammar_text = f.read()
parser = Lark(grammar_text)

prog = r"""
x = 100
let y = 200
z = "foo"
z2 = "bar"
foo = z + z2

cls
print
print x; y; -(x + y*2), foo
print 1, 2, z
end
"""
x = parser.parse(prog)
y=MyT().transform(x)
