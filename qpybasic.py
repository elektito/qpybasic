from lark import Lark, Transformer, Token, Tree

class Instr(tuple):
    def __new__(cls, *args):
        return super(Instr, cls).__new__(cls, tuple(args))

    def __repr__(self):
        return '<Instr {}>'.format(super().__repr__())

    def __str__(self):
        return '\t' + self[0] + '\t' + ', '.join(str(i) for i in self[1:])

class Label:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '{}:'.format(self.value)

class Var:
    def __init__(self, name, typespec):
        assert typespec in '!#%&$'
        self.name = name
        self.typespec = typespec

    def qual_name(self):
        return self.name + self.typespec

    def __eq__(self, other):
        return self.name == other.name and self.typespec == other.typespec

labels = {}
def get_label(prefix):
    if prefix not in labels:
        labels[prefix] = 1
    else:
        labels[prefix] += 1
    return '_{}_{}'.format(prefix, labels[prefix])

variables = {}
def get_variable(prefix, typespec):
    if prefix not in variables:
        variables[prefix] = 1
    else:
        variables[prefix] += 1
    name = '__{}_{}'.format(prefix, variables[prefix])
    return Var(name, typespec)

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

    def stmt_group(self, items):
        return sum((i if isinstance(i, list) else [i] for i in items), [])

    def block(self, items):
        return sum(items, [])

    def block_body(self, items):
        if items:
            return sum(items, [])
        else:
            return []

    def block_body_item(self, items):
        items = items[:-1] # remove final newline
        return sum(items[0], [])

    def for_block(self, items):
        if isinstance(items[-1], Var): # Used "NEXT var"
            if items[-1] != items[1]:
                raise RuntimeError('NEXT variable does not match FOR.')
            items = items[:-1] # remove variable

        if items[5].type == 'STEP_KW':
            _, var, start, _, end, _, step, _, body, _ = items
            step_instrs = step
        else:
            _, var, start, _, end, _, body, _ = items
            step_instrs = [Instr('pushi', 1)]

        start_label = get_label('for_start')
        end_label = get_label('for_end')

        end_var = get_variable('for_end_var', var.typespec)
        step_var = get_variable('for_step_var', var.typespec)

        return \
            start + \
            [Instr('popl', var.qual_name())] + \
            end + \
            [Instr('popl', end_var.qual_name())] + \
            step_instrs + \
            [Instr('popl', step_var.qual_name()),
             Label(start_label),
             Instr('pushl', var.qual_name()),
             Instr('pushl', end_var.qual_name()),
             Instr('gt'),
             Instr('jmpt', end_label)] + \
            body + \
            [Instr('pushl', step_var.qual_name()),
             Instr('pushl', var.qual_name()),
             Instr('add'),
             Instr('popl', var.qual_name()),
             Instr('jmp', start_label),
             Label(end_label)]

    def if_block(self, items):
        # IF, cond, THEN, NEWLINE, then_body, rest, END, IF
        _, cond, _, _, then_body, rest, _, _ = items

        endif_label = get_label('endif')

        if rest:
            label, rest = rest
            rest = [Instr('jmp', endif_label) if r == 'JMP_TO_ENDIF' else r for r in rest]
            return cond + [Instr('jmpf', label)] + then_body + [Instr('jmp', endif_label), Label(label)] + rest + [Label(endif_label)]
        else:
            return cond + [Instr('jmpf', endif_label)] + then_body + [Label(endif_label)]

    def elseif_sub_block(self, items):
        _, cond, _, _, then_body, rest = items

        else_label = get_label('else')

        if rest:
            label, rest = rest

            # JMP_TO_ENDIF is a placeholder that will be later
            # replaced by an actual jmp to the end of the if block.
            instrs = cond + [Instr('jmpf', label)] + then_body + ['JMP_TO_ENDIF', Label(label)] + rest
        else:
            instrs = cond + [Instr('jmpf', endif_label)] + then_body + [Label(endif_label)]

        return else_label, instrs

    def else_sub_block(self, items):
        _, _, body = items
        else_label = get_label('else')
        return else_label, body

    def else_list(self, items):
        # The important cases are handled by separate functions
        # (handling aliases). the only remaining case is no "else" at
        # all.
        return []

    def cls_stmt(self, items):
        return Instr('call', '__cls', 0)

    def end_stmt(self, items):
        return Instr('end')

    def goto_stmt(self, items):
        return Instr('goto', '__' + items[1].value)

    def let_stmt(self, items):
        if len(items) == 3:
            items = items[1:]
        return [items[1],
                Instr('storel', items[0].qual_name())]

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
        assert len(items) == 1
        return items[0]

    def expr_lt(self, items):
        assert len(items) == 2
        return items + [Instr('lt')]

    def expr_gt(self, items):
        assert len(items) == 2
        return items + [Instr('gt')]

    def expr_le(self, items):
        assert len(items) == 2
        return items + [Instr('le')]

    def expr_ge(self, items):
        assert len(items) == 2
        return items + [Instr('ge')]

    def expr_eq(self, items):
        assert len(items) == 2
        return items + [Instr('eq')]

    def expr_ne(self, items):
        assert len(items) == 2
        return items + [Instr('ne')]

    def add_expr(self, items):
        return sum(items, [])

    def expr_add(self, items):
        assert len(items) == 2
        return items + [Instr('add')]

    def expr_sub(self, items):
        assert len(items) == 2
        return items + [Instr('sub')]

    def mult_expr(self, items):
        return sum(items, [])

    def expr_mult(self, items):
        assert len(items) == 2
        return items + [Instr('mul')]

    def expr_div(self, items):
        assert len(items) == 2
        return items + [Instr('div')]

    def unary_expr(self, items):
        assert len(items) == 1
        return items[0]

    def value(self, items):
        assert len(items) == 1
        if isinstance(items[0], Token):
            return [Instr('pushi', items[0].value)]
        elif isinstance(items[0], Var):
            return [Instr('pushl', items[0].qual_name())]
        else:
            return items[0]

    def var(self, items):
        name = items[0].value
        if name[-1] in '!#%&$':
            typespec = name[-1]
            name = name[:-1]
        else:
            typespec = '!'
        return Var(name, typespec)

    def negation(self, items):
        assert len(items) == 1
        return items + [Instr('neg')]


with open('qpybasic.ebnf') as f:
    grammar_text = f.read()

prog = r"""
start:
x = 100:let y = 200
z = "foo"
z2 = "bar"
foo = z + z2

if x < y then
    print "less"
else
    print "more"
end if

xyz: cls
print
10 print x; y; -(x + y*2), foo
print 1, 2, z

for i = 1 to 10
   print "boo"
next

goto 10
end
"""
parser = Lark(grammar_text, propagate_positions=True, start='start')
x = parser.parse(prog)
y=MyC().transform(x)
for x in y:
    print(x)
