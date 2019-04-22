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

    def end_stmt(self, items):
        print('END!')

    def let_stmt(self, items):
        if len(items) == 3:
            items = items[1:]
        var, value = items
        print(":: setting {} to {}".format(var, value))
        self.variables[var.value] = value
        return 'let {} = {}'.format(var, value)

    def print_stmt(self, items):
        items = items[1:]
        if not items:
            print()
        while items:
            i, items = items[0], items[1:]
            if i < 0:
                p = '{} '.format(i)
            else:
                p = ' {} '.format(i)

            if items:
                sep, items = items[0], items[1:]
            else:
                sep = '\n'

            if sep == ';':
                pass
            elif sep == '\n':
                p += sep
            elif sep == ',':
                if len(p) % 14 != 0:
                    n = (len(p) // 14 + 1) * 14 - len(p)
                else:
                    n = 0
                p += n * ' '

            print(p, end='')


    def expr(self, items):
        return items[0]

    def expr_add(self, items):
        x, y = items
        return x + y

    def expr_sub(self, items):
        x, y = items
        return x - y

    def mult_expr(self, items):
        return items[0]

    def expr_mult(self, items):
        x, y = items
        return x * y

    def expr_div(self, items):
        x, y = items
        return x // y

    def unary_expr(self, items):
        return items[0]

    def int_value(self, items):
        item, = items
        if isinstance(item, int):
            return item
        else:
            return int(item.value)

    def var(self, items):
        varname = items[0].value
        if varname in self.variables:
            return self.variables[varname]
        else:
            raise RuntimeError("Undefined variable: {}".format(varname))

    def negation(self, items):
        return -items[0]

with open('qpybasic.ebnf') as f:
    grammar_text = f.read()
parser = Lark(grammar_text)

prog = r"""
x = 100
let y = 200

print
print x; y; -(x + y*2)
end
"""
x = parser.parse(prog)
y=MyT().transform(x)
