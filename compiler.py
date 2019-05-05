import struct
from lark import Lark, Token, Tree


# This is the address programs are loaded, right after where
# conventional memory and upper memory area were back in the bad old
# days.
INITIAL_ADDR = 0x100000


# The address in which string literals will be loaded.
STRING_ADDR = 0x80000000

typespec_chars = '%&!#$'


def get_type_name(typespec):
    return {
        '%': 'INTEGER',
        '&': 'LONG',
        '!': 'SINGLE',
        '#': 'DOUBLE',
        '$': 'STRING',
    }[typespec]


def gen_conv_instrs(t1, t2):
    if t1 == t2:
        return []
    elif t1 in '%&!#' and t2 in '%&!#':
        return [Instr('conv' + t1 + t2)]
    else:
        return None


def get_literal_typespec(literal):
    if literal.startswith('"'): # string literal
        return '$'
    elif literal[-1] in '!#%&': # numeric literal with typespec
        # sanity check first
        if '.' in literal and literal[-1] not in '!#':
            raise RuntimeError('Invalid numeric literal: {}'.format(literal))
        return literal[-1]
    else: # numeric literal without typespec
        if '.' in literal: # single or double
            return '#' # for now, consider all float literals as double
        else: # integer or long
            v = int(literal)
            if -32768 <= v < 32768:
                return '%'
            elif -2**31 <= v < 2**31:
                return '&'
            else:
                raise RuntimeError('Integer value out of possible range.')


class Instr(tuple):
    def __new__(cls, *args):
        return super(Instr, cls).__new__(cls, tuple(args))

    def __repr__(self):
        return '<Instr {}>'.format(super().__repr__())

    def __str__(self):
        return '\t' + self[0] + '\t' + ', '.join(str(i) for i in self[1:])


class Var:
    def __init__(self, source, compiler):
        if isinstance(source, str):
            name = source
        else:
            name = source.children[0].value
        if name[-1] in typespec_chars:
            self.name, self.typespec = name[:-1], name[-1]
        else:
            self.name = name
            self.typespec = self.get_default_typespec(name)

        self.typelen = {
            '%': 2,
            '&': 4,
            '!': 4,
            '#': 8,
            '$': 4,
        }[self.typespec]

        self.routine = compiler.cur_routine
        if self.routine not in compiler.local_vars:
            compiler.local_vars[self.routine] = []

        if self not in compiler.local_vars[self.routine]:
            compiler.local_vars[self.routine].append(self)

        # calculate this variable's index in the stack frame
        self.idx = 0
        for v in compiler.local_vars[self.routine]:
            if v == self:
                break
            else:
                self.idx += v.typelen


    def qual_name(self):
        return f'{self.name}{self.typespec}'


    def get_default_typespec(self, var_name):
        # For now, the default type for all variables is single. This
        # can be changed when we implement DEFINT and friends.
        return '!'


    def gen_read_instructions(self):
        return [Instr(f'readf{self.typelen}', self)]


    def gen_write_instructions(self):
        return [Instr(f'writef{self.typelen}', self)]


    def __repr__(self):
        return '<Var {}{}>'.format(self.name, self.typespec)


    def __eq__(self, other):
        return self.name == other.name and self.typespec == other.typespec


class Expr:
    def __init__(self, ast, parent):
        self.parent = parent
        self.instrs = []
        self.compile_ast(ast)


    def compile_ast(self, ast):
        getattr(self, 'process_' + ast.data)(ast)


    def process_expr_add(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.binary_op('add', left, right)


    def process_expr_sub(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.binary_op('sub', left, right)


    def process_expr_mul(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.binary_op('mul', left, right)


    def process_expr_div(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.binary_op('div', left, right)


    def process_expr_gt(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.compare_op('gt', left, right)


    def process_expr_lt(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.compare_op('lt', left, right)


    def process_expr_ge(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.compare_op('ge', left, right)


    def process_expr_le(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.compare_op('le', left, right)


    def process_expr_eq(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.compare_op('eq', left, right)


    def process_expr_ne(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.compare_op('ne', left, right)


    def process_value(self, ast):
        v = ast.children[0]
        if isinstance(v, Token): # literal
            if v.type == 'STRING_LITERAL':
                self.parent.add_string_literal(v[1:-1])
            t = get_literal_typespec(v.value)
            self.instrs += [Instr('pushi' + t, v.value)]
            self.typespec = t
        elif isinstance(v, Tree): # variable
            var = Var(v, self.parent)
            self.instrs += var.gen_read_instructions()
            self.typespec = var.typespec
        else:
            assert False, 'This should not have happened.'


    def process_negation(self, ast):
        arg = ast.children[0]
        e = Expr(arg, self.parent)
        self.instrs += e.instrs + [Instr(f'neg{e.typespec}')]
        self.typespec = e.typespec


    def binary_op(self, op, left, right):
        ltype, rtype = left.typespec, right.typespec

        if ltype == rtype == '$' and op == 'add':
            self.instrs += [Instr('syscall', '__concat')]
            self.typespec = '$'
            return
        elif ltype == '$' or rtype == '$':
            self.parent.error('Invalid operation.')

        if ltype == rtype:
            t = ltype
        else:
            t = {
                frozenset({'!', '#'}): '#',
                frozenset({'%', '&'}): '&',
                frozenset({'%', '!'}): '!',
                frozenset({'%', '#'}): '#',
                frozenset({'&', '!'}): '!',
                frozenset({'&', '#'}): '#',
            }[frozenset({ltype, rtype})]

        instrs = left.instrs
        instrs += gen_conv_instrs(ltype, t)

        instrs += right.instrs
        instrs += gen_conv_instrs(rtype, t)

        instrs += [Instr(op + t)]

        self.typespec = t
        self.instrs += instrs


    def compare_op(self, op, left, right):
        self.binary_op('sub', left, right)
        self.instrs += gen_conv_instrs(self.typespec, '%')
        self.instrs += [Instr(op)]
        self.typespec = '%'


class Label:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '{}:'.format(self.value)


class Module:
    def __init__(self, bytecode, string_literals):
        self.bytecode = bytecode
        self.string_literals = string_literals

        self.file_magic = b'QB'
        self.version = 1
        self.code_section_type = 1
        self.const_section_type = 2

    def dump(self):
        # file header: magic, version, nsections
        file_hdr = self.file_magic + struct.pack('>BH', self.version, 2)

        # code section header: type, len, load_addr
        code_section_hdr = struct.pack('>BII',
                                       self.code_section_type,
                                       len(self.bytecode),
                                       INITIAL_ADDR)

        # const section header: type, len, load_addr
        total_len = sum(len(i) + 2 for i in self.string_literals)
        const_section_hdr = struct.pack('>BII',
                                        self.const_section_type,
                                        total_len,
                                        STRING_ADDR)

        const_section = b''.join(struct.pack('>h', len(i)) + i.encode('ascii')
                                 for i in self.string_literals)

        return file_hdr + \
            code_section_hdr + \
            self.bytecode + \
            const_section_hdr + \
            const_section


class Compiler:
    def __init__(self):
        with open('qpybasic.ebnf') as f:
            grammar_text = f.read()

        self.parser = Lark(grammar_text,
                           propagate_positions=True,
                           start='program')


    def compile(self, code):
        self.loc = INITIAL_ADDR
        self.labels = {}
        self.instrs = []
        self.cur_routine = '__main'
        self.gen_labels = {}
        self.gen_vars = {}
        self.endif_labels = []
        self.local_vars = {'__main': []}
        self.string_literals = {}
        self.last_string_literal_idx = 0

        # add string literals used by the compiler
        self.add_string_literal(';')
        self.add_string_literal(',')
        self.add_string_literal('%')
        self.add_string_literal('&')
        self.add_string_literal('!')
        self.add_string_literal('#')
        self.add_string_literal('$')

        # Create the main stack frame. The argument to the 'frame'
        # instruction is stack frame size, the actual value of which
        # will be filled in later.
        self.instrs = [Instr('frame', '__main')]

        ast = self.parser.parse(code)
        self.compile_ast(ast)

        self.instrs += [Instr('end')]

        assembler = Assembler(self)
        self.bytecode = assembler.assemble(self.instrs)
        return Module(self.bytecode, self.string_literals)


    def compile_ast(self, ast):
        getattr(self, 'process_' + ast.data)(ast)


    def process_program(self, ast):
        for i in ast.children:
            self.compile_ast(i)


    def process_label(self, t):
        label = t.children[0].value
        self.labels[label] = self.loc
        self.instrs += [Label(label)]


    def process_lineno(self, t):
        label = '__lineno_' + t.children[0].value
        self.labels[label] = self.loc
        self.instrs += [Label(label)]


    def process_block_body(self, ast):
        for i in ast.children:
            self.compile_ast(i)


    def process_cls_stmt(self, ast):
        self.instrs += [Instr('syscall', '__cls')]


    def process_end_stmt(self, ast):
        self.instrs += [Instr('end')]


    def process_for_block(self, ast):
        _, var, start, _, end, step, body, next_stmt = ast.children
        var = Var(var, self)
        if var.typespec == '$':
            self.error('Invalid FOR variable.')

        if len(next_stmt.children) == 2:
            # "NEXT var" is used. Check if NEXT variable matches FOR
            # variable.
            next_var = Var(next_stmt.children[1], self)
            if next_var != var:
                self.error('NEXT variable does not match FOR.')

        step_var = Var(self.gen_var('for_step', var.typespec), self)
        if step.children:
            e = Expr(step.children[1], self)
            self.gen_set_var_code(step_var, e)
        else:
            self.instrs += [Instr(f'pushi{var.typespec}', 1),
                            Instr(f'writef{var.typelen}', step_var)]

        end_var = Var(self.gen_var('for_end', var.typespec), self)
        e = Expr(end, self)
        self.gen_set_var_code(end_var, e)

        e = Expr(start, self)
        self.gen_set_var_code(var, e)

        top_label = self.gen_label('for_top')
        end_label = self.gen_label('for_bottom')
        self.instrs += [Label(top_label),
                        Instr(f'readf{var.typelen}', var),
                        Instr(f'readf{end_var.typelen}', end_var),
                        Instr(f'sub{var.typespec}')]
        self.instrs += gen_conv_instrs(var.typespec, '%')
        self.instrs += [Instr('gt'),
                        Instr('jmpt', end_label)]
        self.compile_ast(body)
        self.instrs += [Instr(f'readf{var.typelen}', var),
                        Instr(f'readf{step_var.typelen}', step_var),
                        Instr(f'add{var.typespec}'),
                        Instr(f'writef{var.typelen}', var),
                        Instr('jmp', top_label),
                        Label(end_label)]


    def process_next_stmt(self, ast):
        self.error('NEXT without FOR.')


    def process_goto_stmt(self, ast):
        _, target = ast.children
        if target.type == 'ID':
            self.instrs += [Instr('jmp', target)]
        else:
            self.instrs += [Instr('jmp', f'__lineno_{target.value}')]


    def process_if_block(self, ast):
        _, cond, _, then_body, rest, _, _ = ast.children
        cond = Expr(cond, self)
        self.instrs += cond.instrs + gen_conv_instrs(cond.typespec, '%')

        self.endif_labels.append(self.gen_label('endif'))

        if rest.children:
            else_label = self.gen_label('else')
            self.instrs += [Instr('jmpf', else_label)]
            self.compile_ast(then_body)
            self.instrs += [Instr('jmp', self.endif_labels[-1])]
            self.instrs += [Label(else_label)]
            self.compile_ast(rest)
        else:
            self.instrs += [Instr('jmpf', self.endif_labels[-1])]
            self.compile_ast(then_body)

        self.instrs += [Label(self.endif_labels[-1])]

        self.endif_labels.pop()


    def process_elseif_sub_block(self, ast):
        _, cond, _, then_body, rest = ast.children
        cond = Expr(cond, self)
        self.instrs += cond.instrs + gen_conv_instrs(cond.typespec, '%')

        if rest.children:
            else_label = self.gen_label('else')
            self.instrs += [Instr('jmpf', else_label)]
            self.compile_ast(then_body)
            self.instrs += [Instr('jmp', self.endif_labels[-1])]
            self.instrs += [Label(else_label)]
            self.compile_ast(rest)
        else:
            self.instrs += [Instr('jmpf', self.endif_labels[-1])]
            self.compile_ast(then_body)


    def process_else_sub_block(self, ast):
        _, body = ast.children
        self.compile_ast(body)


    def process_let_stmt(self, ast):
        if len(ast.children) == 3:
            # cut the LET keyword
            ast.children = ast.children[1:]

        var, expr = ast.children
        var = Var(var, self)
        expr = Expr(expr, self)
        self.gen_set_var_code(var, expr)


    def gen_set_var_code(self, var, expr):
        conv_instrs = gen_conv_instrs(expr.typespec, var.typespec)
        if conv_instrs == None:
            self.error('Cannot convert {} to {}.'
                       .format(get_type_name(expr.typespec),
                               get_type_name(var.typespec)))
        else:
            self.instrs += expr.instrs + conv_instrs
            self.instrs += var.gen_write_instructions()


    def process_print_stmt(self, ast):
        # passing arguments to PRINT:
        #  - the arguments are passed in reverse order
        #  - each argument is preceded by an integer (%) value determining
        #    its type.
        #  - semicolons and commas are also considered arguments.
        #  - for semicolons and commas, only the type is passed.

        type_values = {
            '%': 1,
            '&': 2,
            '!': 3,
            '#': 4,
            '$': 5,
            ';': 6,
            ',': 7,
        }

        parts = []
        for i in ast.children[1:]:
            if isinstance(i, Token):
                parts.append([Instr('pushi%', type_values[i.value])])
            else:
                expr = Expr(i, self)
                parts.append(expr.instrs + \
                             [Instr('pushi%', type_values[expr.typespec])])

        # push the arguments in reverse order, and after that, the
        # number of arguments. this is so that we can read the
        # arguments one at a time, armed with proper information to
        # handle each.
        self.instrs += sum(reversed(parts), []) + \
                       [Instr('pushi%', len(ast.children) - 1)] + \
                       [Instr('syscall', '__print')]


    def error(self, msg):
        raise RuntimeError('COMPILE ERROR: {}'.format(msg))


    def gen_label(self, prefix):
        if prefix not in self.gen_labels:
            self.gen_labels[prefix] = 1
        else:
            self.gen_labels[prefix] += 1
        return '__{}_{}'.format(prefix, self.gen_labels[prefix])


    def gen_var(self, prefix, typespec):
        if prefix not in self.gen_vars:
            self.gen_vars[prefix] = 1
        else:
            self.gen_vars[prefix] += 1
        idx = self.gen_vars[prefix]
        return f'__{prefix}_{idx}{typespec}'


    def add_string_literal(self, literal):
        if literal not in self.string_literals:
            # For each string, we need two extra bytes for the length
            # which will be stored right before the string data in
            # memory.
            self.string_literals[literal] = self.last_string_literal_idx
            self.last_string_literal_idx += len(literal) + 2


class Assembler:
    def __init__(self, compiler):
        self.compiler = compiler


    def assemble(self, instrs):
        all_instrs = {
            'add%': (1, 0x01, self.assemble_one_byte),
            'add&': (1, 0x02, self.assemble_one_byte),
            'add!': (1, 0x03, self.assemble_one_byte),
            'add#': (1, 0x04, self.assemble_one_byte),
            'call': (5, 0x05, self.assemble_call),
            'conv%&': (1, 0x06, self.assemble_one_byte),
            'conv%!': (1, 0x07, self.assemble_one_byte),
            'conv%#': (1, 0x08, self.assemble_one_byte),
            'conv&%': (1, 0x09, self.assemble_one_byte),
            'conv&!': (1, 0x0a, self.assemble_one_byte),
            'conv&#': (1, 0x0b, self.assemble_one_byte),
            'conv!%': (1, 0x0c, self.assemble_one_byte),
            'conv!&': (1, 0x0d, self.assemble_one_byte),
            'conv!#': (1, 0x0e, self.assemble_one_byte),
            'conv#%': (1, 0x0f, self.assemble_one_byte),
            'conv#&': (1, 0x10, self.assemble_one_byte),
            'conv#!': (1, 0x12, self.assemble_one_byte),
            'end': (1, 0x13, self.assemble_one_byte),
            'frame': (3, 0x14, self.assemble_frame),
            'eq': (1, 0x15, self.assemble_one_byte),
            'ge': (1, 0x16, self.assemble_one_byte),
            'gt': (1, 0x17, self.assemble_one_byte),
            'jmp': (5, 0x18, self.assemble_jmp),
            'jmpf': (3, 0x19, self.assemble_jmp_c),
            'jmpt': (3, 0x1a, self.assemble_jmp_c),
            'le': (1, 0x1b, self.assemble_one_byte),
            'lt': (1, 0x1c, self.assemble_one_byte),
            'mul%': (1, 0x1d, self.assemble_one_byte),
            'mul&': (1, 0x1e, self.assemble_one_byte),
            'mul!': (1, 0x1f, self.assemble_one_byte),
            'mul#': (1, 0x20, self.assemble_one_byte),
            'ne': (1, 0x21, self.assemble_one_byte),
            'neg%': (1, 0x22, self.assemble_one_byte),
            'neg&': (1, 0x23, self.assemble_one_byte),
            'neg!': (1, 0x24, self.assemble_one_byte),
            'neg#': (1, 0x25, self.assemble_one_byte),
            'pushi%': (3, 0x26, self.assemble_pushi),
            'pushi&': (5, 0x27, self.assemble_pushi),
            'pushi!': (5, 0x28, self.assemble_pushi),
            'pushi#': (9, 0x29, self.assemble_pushi),
            'pushi$': (5, 0x2a, self.assemble_pushi),
            'readf1': (3, 0x2b, self.assemble_readf),
            'readf2': (3, 0x2c, self.assemble_readf),
            'readf4': (3, 0x2d, self.assemble_readf),
            'sub%': (1, 0x2e, self.assemble_one_byte),
            'sub&': (1, 0x2f, self.assemble_one_byte),
            'sub!': (1, 0x30, self.assemble_one_byte),
            'sub#': (1, 0x31, self.assemble_one_byte),
            'syscall': (3, 0x32, self.assemble_syscall),
            'writef1': (3, 0x33, self.assemble_writef),
            'writef2': (3, 0x34, self.assemble_writef),
            'writef4': (3, 0x35, self.assemble_writef),
            'writef8': (3, 0x36, self.assemble_writef),
        }

        # phase 1: calculate label addresses
        p = 0
        labels = {}
        for i in instrs:
            if isinstance(i, Label):
                labels[i.value] = INITIAL_ADDR + p
            else:
                size, _, _ = all_instrs[i[0]]
                p += size

        # phase 2: assemble
        p = 0
        code = b''
        for i in instrs:
            if not isinstance(i, Instr):
                continue

            size, opcode, assemble = all_instrs[i[0]]
            c = assemble(i, opcode, p, labels)

            assert len(c) == size
            p += len(c)
            code += c

        return code


    def assemble_one_byte(self, instr, opcode, p, _):
        return bytes([opcode])


    def assemble_call(self, instr, opcode, p, labels):
        instr, dest = instr
        dest = struct.pack('>I', labels[dest])
        return bytes([opcode]) + dest


    def assemble_frame(self, instr, opcode, _, __):
        instr, routine = instr
        frame_size = sum(v.typelen for v in self.compiler.local_vars[routine])
        return bytes([opcode]) + struct.pack('>H', frame_size)


    def assemble_jmp(self, instr, opcode, p, labels):
        instr, dest = instr
        dest = struct.pack('>I', labels[dest])
        return bytes([opcode]) + dest


    def assemble_jmp_c(self, instr, opcode, p, labels):
        instr, dest = instr
        diff = labels[dest] - INITIAL_ADDR - p
        if diff < -32768 or diff >32767:
            raise RuntimeError('Conditional jump too long.')
        dest = struct.pack('>h', diff)
        return bytes([opcode]) + dest


    def assemble_pushi(self, instr, opcode, _, __):
        instr, imm = instr
        t = instr[-1]

        if isinstance(imm, str):
            if t != '$' and imm[-1] in '%&!#':
                imm = imm[:-1]

            if t == '$':
                imm = STRING_ADDR + self.compiler.string_literals[imm[1:-1]]
            elif t in '!#':
                imm = float(imm)
            else:
                imm = int(imm)

        if t == '%':
            imm = struct.pack('>h', imm)
        elif t == '&':
            imm = struct.pack('>i', imm)
        elif t == '!':
            imm = struct.pack('>f', imm)
        elif t == '#':
            imm = struct.pack('>d', imm)
        elif t == '$':
            imm = struct.pack('>I', imm)
        else:
            raise RuntimeError('Invalid value.')

        return bytes([opcode]) + imm


    def assemble_readf(self, instr, opcode, _, __):
        instr, imm = instr
        if isinstance(imm, Var):
            imm = imm.idx
        imm = struct.pack('>H', imm)
        return bytes([opcode]) + imm


    def assemble_syscall(self, instr, opcode, _, __):
        instr, imm = instr
        imm = {
            '__cls': 0x02,
            '__concat': 0x03,
            '__print': 0x04,
        }[imm]
        imm = struct.pack('>H', imm)
        return bytes([opcode]) + imm


    def assemble_writef(self, instr, opcode, _, __):
        instr, imm = instr
        if isinstance(imm, Var):
            imm = imm.idx
        imm = struct.pack('>H', imm)
        return bytes([opcode]) + imm


    def get_str_literal_idx(self, literal):
        pass

