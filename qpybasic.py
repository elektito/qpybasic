from lark import Lark, Transformer, Token, Tree

class Instr(tuple):
    def __new__(cls, *args):
        return super(Instr, cls).__new__(cls, tuple(args))

    def __repr__(self):
        return '<Instr {}>'.format(super().__repr__())

    def __str__(self):
        return self[0] + '\t' + ', '.join(str(i) for i in self[1:])

class Label:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '{}:'.format(self.value)

def flatten_tree(t):
    ret = []
    for i in t:
        if isinstance(i, list):
            ret += flatten_tree(i)
        elif isinstance(i, Token) and i.type == 'NEWLINE':
            pass
        elif isinstance(i, Instr):
            ret.append(i)
        elif isinstance(i, Label):
            ret.append(i)
        elif i == None:
            pass
        else:
            raise RuntimeError('Got something unexpected: {}'.format(repr(i)))

    return ret

class MyC(Transformer):
    def start(self, items):
        return items[0]

    def program(self, items):
        return flatten_tree(items)

    def label(self, items):
        return Label('__{}'.format(items[0].value))

    def lineno(self, items):
        return Label('__{}'.format(items[0].value))

    def line(self, items):
        return items

    def stmt(self, items):
        return items[0]

    def cls_stmt(self, items):
        return Instr('call', '__cls', 0)

    def end_stmt(self, items):
        return Instr('end')

    def let_stmt(self, items):
        if len(items) == 3:
            items = items[1:]
        return [items[1],
                Instr('storel', items[0].value)]

    def print_stmt(self, items):
        args = []
        for a in items[1:]:
            if isinstance(a, Instr):
                args.append(a)
            elif isinstance(a, list):
                args += a
            elif isinstance(a, Token):
                args.append(Instr('pushi', '"{}"'.format(a.value)))
            else:
                RuntimeError('Something unexpected sent to PRINT.')
        return args + [Instr('call', '__print', len(items) - 1)]

    def expr(self, items):
        return items[0]

    def expr_add(self, items):
        x, y = items
        if isinstance(x, Instr):
            x = [x]
        if isinstance(y, Instr):
            y = [y]
        return x + y + [Instr('add')]

    def expr_sub(self, items):
        x, y = items
        if isinstance(x, Instr):
            x = [x]
        if isinstance(y, Instr):
            y = [y]
        return x + y + [Instr('sub')]

    def mult_expr(self, items):
        return items[0]

    def expr_mult(self, items):
        x, y = items
        if isinstance(x, Instr):
            x = [x]
        if isinstance(y, Instr):
            y = [y]
        return x + y + [Instr('mul')]

    def expr_div(self, items):
        x, y = items
        if isinstance(x, Instr):
            x = [x]
        if isinstance(y, Instr):
            y = [y]
        return x + y + [Instr('div')]

    def unary_expr(self, items):
        return items[0]

    def value(self, items):
        if isinstance(items[0], Instr):
            return items[0]
        elif isinstance(items[0], Token):
            return Instr('pushi', items[0].value)

    def var(self, items):
        return Instr('pushl', items[0].value)

    def negation(self, items):
        x = items[0]
        if isinstance(x, Instr):
            x = [x]
        return x + [Instr('neg')]


with open('qpybasic.ebnf') as f:
    grammar_text = f.read()
parser = Lark(grammar_text)

prog = r"""
start:
x = 100
let y = 200
z = "foo"
z2 = "bar"
foo = z + z2

xyz: cls
print
10 print x; y; -(x + y*2), foo
print 1, 2, z
end
"""
x = parser.parse(prog)
y=MyC().transform(x)
for x in y:
    print(x)
