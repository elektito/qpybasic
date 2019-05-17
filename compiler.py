import struct
from collections import OrderedDict
from lark import Lark, Token, Tree
from asm import Instr


# This is the address programs are loaded, right after where
# conventional memory and upper memory area were back in the bad old
# days.
INITIAL_ADDR = 0x100000


# The address in which string literals will be loaded.
STRING_ADDR = 0x80000000

typespec_chars = '%&!#$'

typekw_to_typespec = {
    'INTEGER': '%',
    'LONG': '&',
    'SINGLE': '!',
    'DOUBLE': '#',
    'STRING': '$',
}

typespec_to_typename = {
    '%': 'INTEGER',
    '&': 'LONG',
    '!': 'SINGLE',
    '#': 'DOUBLE',
    '$': 'STRING',
}


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


def get_literal_type(literal):
    if literal.startswith('"'): # string literal
        return Type('$')
    elif literal[-1] in '!#%&': # numeric literal with typespec
        # sanity check first
        if '.' in literal and literal[-1] not in '!#':
            raise RuntimeError('Invalid numeric literal: {}'.format(literal))
        return Type(literal[-1])
    else: # numeric literal without typespec
        if '.' in literal: # single or double
            return Type('#') # for now, consider all float literals as double
        else: # integer or long
            v = int(literal)
            if -32768 <= v < 32768:
                return Type('%')
            elif -2**31 <= v < 2**31:
                return Type('&')
            else:
                raise RuntimeError('Integer value out of possible range.')


def get_default_type(var_name):
    # For now, the default type for all variables is single. This can
    # be changed when we implement DEFINT and friends.
    return Type('SINGLE')


def builtin_func_abs(parent, args):
    if len(args) != 1:
        raise RuntimeError('Invalid number of arguments passed to function.')

    arg, = args
    e = Expr(arg, parent)
    instrs = e.instrs
    end_label = parent.gen_label('abs_end')
    instrs += [Instr(f'dup{e.type.get_size()}'),
               Instr(f'sgn{e.type.typespec}'),
               Instr('lt'),
               Instr('jmpf', end_label),
               Instr(f'neg{e.type.typespec}'),
               Label(end_label)]
    return e.type, e.instrs


builtin_functions = {
    'abs': builtin_func_abs,
}


class Type:
    def __init__(self, name, *, is_array=False):
        if name in typespec_chars:
            self.name = typespec_to_typename[name]
        else:
            self.name = name.upper()

        self.is_array = is_array


    @property
    def typespec(self):
        return {
            'INTEGER': '%',
            'LONG': '&',
            'SINGLE': '!',
            'DOUBLE': '#',
            'STRING': '$'
        }.get(self.name, '')


    @property
    def is_basic(self):
        return self.name in [
            'INTEGER',
            'LONG',
            'SINGLE',
            'DOUBLE',
            'STRING'
        ]


    def get_size(self):
        # user-defined types and arrays not supported yet
        assert self.is_basic and not self.is_array
        return {
            'INTEGER': 2,
            'LONG': 4,
            'SINGLE': 4,
            'DOUBLE': 8,
            'STRING': 4,
        }[self.name]


    def __repr__(self):
        if self.typespec == '':
            name = f'User-Defined: {self.name}'
        else:
            name = self.name
        suffix = '()' if self.is_array else ''
        return f'<Type {name}{suffix}>'


    def __eq__(self, other):
        return isinstance(other, Type) and \
            self.name == other.name and \
            self.is_array == other.is_array


    def __hash__(self):
        return hash((self.name, self.is_array))


    @staticmethod
    def from_var_name(var_name):
        if var_name[-1] in typespec_chars:
            return Type(var_name[-1])
        else:
            return get_default_type(var_name)


class Var:
    def __init__(self, used_name, routine, *, status='used', is_param=False, type=None, byref=False):
        assert status in ['used', 'dimmed']
        assert not byref or is_param

        if isinstance(used_name, Tree):
            assert len(used_name.children) == 1 and isinstance(used_name.children[0], Token)
            used_name = used_name.children[0].value

        self.routine = routine
        self.byref = byref

        if status == 'dimmed':
            if used_name in self.routine.no_type_dimmed:
                # already declared with something like "DIM x$"
                raise RuntimeError('Use AS on first DIM.') ##### <------------------------------------- FIX the error to the original

            if type != None and used_name[-1] in typespec_chars:
                raise RuntimeError('Invalid variable name.')
            elif type != None and used_name[-1] not in typespec_chars:
                self.name = used_name
                self.type = type
            elif type == None and used_name[-1] in typespec_chars:
                # DIM x$
                self.name = used_name[:-1]
                self.type = Type(used_name[-1])

                # this is almost the same as using the variable
                # without DIM, except for the fact that you can't DIM
                # it again. so we add the name to a separate list, so
                # that we won't let the variable to be DIM'ed again.
                routine.no_type_dimmed.add(self.name)
            else: # type == None and used_name[-1] not in typespec_chars:
                self.name = used_name
                self.type = get_default_type(used_name)

            if self.name in routine.dimmed_vars:
                raise RuntimeError('Duplicate definition.')

            if self.name not in routine.no_type_dimmed:
                # not properly dimmed (no AS clause), and can't be
                # used without the typespec, so do not add it to the
                # list.
                routine.dimmed_vars[self.name] = self

            if is_param:
                self.routine.params.append(self)
            elif self not in self.routine.local_vars:
                self.routine.local_vars.append(self)
        elif status == 'used':
            assert type == None

            if used_name[-1] in typespec_chars:
                self.name = used_name[:-1]
                self.type = Type(used_name[-1])
            else:
                self.name = used_name
                if used_name in routine.dimmed_vars:
                    self.type = routine.dimmed_vars[used_name].type
                else:
                    self.type = get_default_type(used_name)

            if self not in self.routine.params and self not in self.routine.local_vars:
                self.routine.local_vars.append(self)

            if self in self.routine.params:
                pidx = self.routine.params.index(self)
                self.idx = 8 + sum(v.size for v in self.routine.params[:pidx])
                self.byref = self.routine.params[pidx].byref
            else:
                lidx = self.routine.local_vars.index(self)
                self.idx = -sum(v.size for v in self.routine.local_vars[:lidx+1])
                self.byref = False

        if self not in routine.all_vars:
            routine.all_vars.append(self)

        self.is_param = is_param


    def gen_read_instructions(self):
        if self.byref:
            return [Instr(f'readf4', self),
                    Instr(f'readi{self.type.get_size()}')]
        else:
            return [Instr(f'readf{self.type.get_size()}', self)]


    def gen_write_instructions(self):
        if self.byref:
            return [Instr(f'readf4', self),
                    Instr(f'writei{self.size}')]
        else:
            return [Instr(f'writef{self.size}', self)]


    @property
    def size(self):
        if self.byref:
            return 4
        else:
            return self.type.get_size()


    def __eq__(self, other):
        return isinstance(other, Var) and \
            self.name == other.name and \
            self.type == other.type


    def __hash__(self):
        return hash((self.name, self.type))


    def __repr__(self):
        pass_type = ' BYREF' if self.byref else ''
        if hasattr(self, 'idx'):
            return f'<Var {self.name}{self.type.typespec} idx={self.idx}{pass_type}>'
        else:
            return f'<Var {self.name}{self.type.typespec}{pass_type}>'


class Routine:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.params = []
        self.local_vars = []
        self.dimmed_vars = OrderedDict()
        self.no_type_dimmed = set()
        self.all_vars = []
        self.instrs = []


    def __repr__(self):
        return f'<{self.type.upper()} {self.name} {self.params}>'


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


    def process_function_call(self, ast):
        if len(ast.children) == 2:
            fname, args = ast.children
            args = args.children
        else:
            fname, = ast.children
            args = []
        fname = fname.lower()
        if fname in builtin_functions:
            f = builtin_functions[fname]
            self.type, self.instrs = f(self.parent, args)
        else:
            if fname not in self.parent.routines and \
               fname not in self.parent.declared_functions:
                raise RuntimeError(f'No such function: {fname}')

            i = len(args) - 1
            for a in reversed(args):
                if isinstance(a, Tree) and a.data == 'value' and a.children[0].type == 'TYPED_ID':
                    # just a variable: send byref
                    v = Var(a.children[0].value, self.parent.cur_routine)
                    if fname in self.parent.routines:
                        if v.type != self.parent.routines[fname].params[i].type:
                            raise RuntimeError('Parameter type mismatch.')
                    else:
                        if v.type != self.parent.declared_functions[fname][i]:
                            raise RuntimeError('Parameter type mismatch.')
                    self.instrs += [Instr('pushfp', v)]
                else:
                    e = Expr(a, self.parent)
                    if fname in self.parent.routines:
                        typespec = self.parent.routines[fname].params[i].type.typespec
                    else:
                        typespec = self.parent.declared_functions[fname][i].typespec
                    v = Var(self.parent.gen_var('rvalue', typespec), self.parent.cur_routine)
                    self.parent.gen_set_var_code(v, e)
                    self.instrs += [Instr('pushfp', v)]

                i -= 1

            self.instrs += [Instr('call', f'__function_{fname}')]
            self.type = Type.from_var_name(fname)


    def process_value(self, ast):
        v = ast.children[0]
        if v.type == 'TYPED_ID': # variable or function call
            if self.parent.is_function(v):
                self.process_function_call(ast)
            else:
                var = Var(v, self.parent.cur_routine)
                self.instrs += var.gen_read_instructions()
                self.type = var.type
        elif v.type == 'STRING_LITERAL':
            self.parent.add_string_literal(v[1:-1])
            self.instrs += [Instr('pushi$', v.value)]
            self.type = t
        elif v.type == 'NUMERIC_LITERAL':
            t = get_literal_type(v.value)
            if t.name.lower() in ['integer', 'long']:
                value = int(v.value)
            else:
                value = float(v.value)
            self.instrs += [Instr('pushi' + t.typespec, value)]
            self.type = t
        else:
            assert False, 'This should not have happened.'


    def process_negation(self, ast):
        arg = ast.children[0]
        e = Expr(arg, self.parent)
        self.instrs += e.instrs + [Instr(f'neg{e.type.typespec}')]
        self.type = e.type


    def binary_op(self, op, left, right):
        ltype, rtype = left.type.typespec, right.type.typespec

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

        self.type = Type(t)
        self.instrs += instrs


    def compare_op(self, op, left, right):
        self.binary_op('sub', left, right)
        self.instrs += gen_conv_instrs(self.type.typespec, '%')
        self.instrs += [Instr(op)]
        self.typespec = '%'


class Label:
    def __init__(self, value):
        assert isinstance(value, str)
        self.value = value


    def __repr__(self):
        return f'<Label "{self.value}">'


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


class PostLex:
    def __init__(self, compiler):
        self.compiler = compiler

        self.always_accept = ()


    # This post-lexer converts a single END keyword, to an END_CMD
    # terminal. This makes the keyword available to be properly used
    # in constructs like END IF and END SUB.
    #
    # Notice that this post-lexer only works with Lark's "standard"
    # lexer, because it consumes two tokens and then spits them back
    # (or one END_CMD token in their place).
    def process(self, stream):
        prev_tok = None
        for tok in stream:
            if tok.value.lower() == 'end' and \
               (prev_tok == None or prev_tok.type == '_NEWLINE'):
                try:
                    next_tok = next(stream)
                except StopIteration:
                    yield Token.new_borrow_pos('END_CMD', 'END_CMD', tok)
                else:
                    if next_tok.type == '_NEWLINE':
                        yield Token.new_borrow_pos('END_CMD', 'END_CMD', tok)
                    else:
                        yield tok
                    yield next_tok
            else:
                yield tok
            prev_tok = tok


class Compiler:
    def __init__(self):
        with open('qpybasic.ebnf') as f:
            grammar_text = f.read()

        self.parser = Lark(grammar_text,
                           parser='lalr',
                           lexer='standard',
                           postlex=PostLex(self),
                           propagate_positions=True,
                           start='program')


    def compile(self, code):
        self.instrs = []
        self.cur_routine = Routine('__main', 'sub')
        self.routines = {'__main': self.cur_routine}
        self.gen_labels = {}
        self.gen_vars = {}
        self.endif_labels = []
        self.string_literals = {}
        self.last_string_literal_idx = 0

        # map the names of declared subs/functions to a list of types,
        # which are the types of the parameters each receives.
        self.declared_subs = {}
        self.declared_functions = {}

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
        self.instrs = [Label('__start'),
                       Instr('call', '__main'),
                       Instr('end'),
                       Label('__main'),
                       Instr('frame', self.routines['__main'])]

        ast = self.parser.parse(code)
        self.compile_ast(ast)

        self.instrs += [Instr('unframe', self.routines['__main']),
                        Instr('ret', 0)]
        self.instrs += sum((r.instrs for r in self.routines.values()), [])

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
        self.instrs += [Label(label)]


    def process_lineno(self, t):
        label = '__lineno_' + t.children[0].value
        self.instrs += [Label(label)]


    def process_block_body(self, ast):
        for i in ast.children:
            self.compile_ast(i)


    def process_call_stmt(self, ast):
        if len(ast.children) == 2:
            sub_name, args = ast.children
        else:
            _, sub_name, args = ast.children

        if sub_name not in self.routines:
            raise RuntimeError(f'No such sub-routine: {sub_name}')

        i = len(args.children) - 1
        for a in reversed(args.children):
            if isinstance(a, Tree) and a.data == 'value' and a.children[0].type == 'TYPED_ID':
                # just a variable: send byref
                v = Var(a.children[0].value, self.cur_routine)
                if v.type != self.routines[sub_name].params[i].type:
                    raise RuntimeError('Parameter type mismatch.')
                self.instrs += [Instr('pushfp', v)]
            else:

                e = Expr(a, self)
                typespec = self.routines[sub_name].params[i].type.typespec
                v = Var(self.gen_var('rvalue', typespec), self.cur_routine)
                self.gen_set_var_code(v, e)
                self.instrs += [Instr('pushfp', v)]

            i -= 1

        self.instrs += [Instr('call', f'__sub_{sub_name}')]


    def process_cls_stmt(self, ast):
        self.instrs += [Instr('syscall', '__cls')]


    def process_declare_stmt(self, ast):
        if self.cur_routine.name != '__main':
            raise RuntimeError('DECLARE statement only valid in top-level.')

        _, rtype, name, params = ast.children
        name = name.value
        param_types = []
        for p in params.children:
            if len(p.children) == 3:
                # form: var AS type
                pname, _, ptype = p.children
                pname = pname.value
                ptype = Type(ptype.children[0].value)
            else:
                # form: var$
                pname = p.children[0].value
                ptype = Type.from_var_name(pname)

            param_types.append(ptype)

        if name in self.routines:
            defined_param_types = [v.type for v in self.routines[name].params]
            if param_types != defined_param_types:
                raise RuntimeError('DECLARE statement does not match definition.')

        if rtype == 'sub':
            if name in self.declared_subs:
                if param_types != self.declared_subs[name]:
                    raise RuntimeError('Conflicting DECLARE statements.')
            else:
                self.declared_subs[name] = param_types
        else:
            if name in self.declared_functions:
                if param_types != self.declared_functions[name]:
                    raise RuntimeError('Conflicting DECLARE statements.')
            else:
                self.declared_functions[name] = param_types


    def process_end_stmt(self, ast):
        self.instrs += [Instr('end')]


    def process_for_block(self, ast):
        _, var, start, _, end, step, body, next_stmt = ast.children
        var = Var(var, self.cur_routine)
        if var.type.typespec == '$':
            self.error('Invalid FOR variable.')

        if len(next_stmt.children) == 2:
            # "NEXT var" is used. Check if NEXT variable matches FOR
            # variable.
            next_var = Var(next_stmt.children[1], self.cur_routine)
            if next_var != var:
                self.error('NEXT variable does not match FOR.')

        step_var = Var(self.gen_var('for_step', var.type.typespec), self.cur_routine)
        if step.children:
            e = Expr(step.children[1], self)
            self.gen_set_var_code(step_var, e)
        else:
            self.instrs += [Instr(f'pushi{var.type.typespec}', 1),
                            Instr(f'writef{var.size}', step_var)]

        end_var = Var(self.gen_var('for_end', var.type.typespec), self.cur_routine)
        e = Expr(end, self)
        self.gen_set_var_code(end_var, e)

        e = Expr(start, self)
        self.gen_set_var_code(var, e)

        top_label = self.gen_label('for_top')
        end_label = self.gen_label('for_bottom')
        self.instrs += [Label(top_label),
                        Instr(f'readf{var.size}', var),
                        Instr(f'readf{end_var.size}', end_var),
                        Instr(f'sub{var.type.typespec}')]
        self.instrs += gen_conv_instrs(var.type.typespec, '%')
        self.instrs += [Instr('gt'),
                        Instr('jmpt', end_label)]
        self.compile_ast(body)
        self.instrs += [Instr(f'readf{var.size}', var),
                        Instr(f'readf{step_var.size}', step_var),
                        Instr(f'add{var.type.typespec}'),
                        Instr(f'writef{var.size}', var),
                        Instr('jmp', top_label),
                        Label(end_label)]


    def process_sub_block(self, ast):
        _, name, params, body, _, _ = ast.children

        if name[-1] in typespec_chars:
            raise RuntimeError('Invalid SUB name.')

        saved_instrs = self.instrs
        self.instrs = []
        self.cur_routine = Routine(name.value, 'sub')

        self.instrs += [Label(f'__sub_{name}'),
                        Instr('frame', self.cur_routine)]

        for p in params.children:
            if len(p.children) == 3:
                # form: var AS type
                pname, _, ptype = p.children
                pname = pname.value
                ptype = Type(ptype.children[0].value)
            else:
                # form: var$
                pname = p.children[0].value
                ptype = None

            if any(pname == i.name for i in self.cur_routine.local_vars):
                self.error('Duplicate parameter.')

            var = Var(pname, self.cur_routine, type=ptype, is_param=True, status='dimmed', byref=True)

        if name in self.declared_subs:
            defined_param_types = [v.type for v in self.cur_routine.params]
            if defined_param_types != self.declared_subs[name]:
                raise RuntimeError(
                    'SUB definition does not match previous DECLARE.')

        self.routines[name.value] = self.cur_routine
        self.compile_ast(body)

        arg_size = sum(v.size for v in self.cur_routine.params)
        self.instrs += [Instr('unframe', self.cur_routine.name),
                        Instr('ret', arg_size)]
        self.cur_routine.instrs = self.instrs

        self.cur_routine = self.routines['__main']
        self.instrs = saved_instrs


    def process_function_block(self, ast):
        _, name, params, body, _, _ = ast.children

        ftype = Type.from_var_name(name.value)
        name = name.value

        saved_instrs = self.instrs
        self.instrs = []
        self.cur_routine = Routine(name, 'function')

        self.instrs += [Label(f'__function_{name}'),
                        Instr('frame', self.cur_routine)]

        for p in params.children:
            if len(p.children) == 3:
                # form: var AS type
                pname, _, ptype = p.children
                pname = pname.value
                ptype = Type(ptype.children[0].value)
            else:
                # form: var$
                pname = p.children[0].value
                ptype = None

            if any(pname == i.name for i in self.cur_routine.local_vars):
                self.error('Duplicate parameter.')

            var = Var(pname, self.cur_routine, type=ptype, is_param=True, status='dimmed', byref=True)

        # create a variable with the same name as the function. this
        # will be used for returning values.
        Var(name, self.cur_routine, status='dimmed')

        # mark the variable as used so that an index is allocated to
        # it.
        ret_var = Var(name, self.cur_routine, status='used')

        # initialize the return variable.
        if ftype.name == 'STRING':
            self.instrs += [Instr(f'pushi$'), '""',
                            Instr(f'writef4', ret_var)]
        else:
            self.instrs += [Instr(f'pushi{ftype.typespec}', 0),
                            Instr(f'writef{ftype.get_size()}', ret_var)]

        if name in self.declared_functions:
            defined_param_types = [v.type for v in self.cur_routine.params]
            if defined_param_types != self.declared_functions[name]:
                raise RuntimeError(
                    'FUNCTION definition does not match previous DECLARE.')

        self.routines[name] = self.cur_routine
        self.compile_ast(body)

        arg_size = sum(v.size for v in self.cur_routine.params)
        self.instrs += [Instr('unframe_r', self.cur_routine, ret_var),
                        Instr(f'ret_r', arg_size, ftype.get_size())]
        self.cur_routine.instrs = self.instrs

        self.cur_routine = self.routines['__main']
        self.instrs = saved_instrs



    def process_next_stmt(self, ast):
        # NEXT statements with a matching FOR will be processed by the
        # process_for_block function. This will only be called when
        # there is a NEXT without FOR.
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
        var = Var(var, self.cur_routine)
        expr = Expr(expr, self)
        self.gen_set_var_code(var, expr)


    def gen_set_var_code(self, var, expr):
        conv_instrs = gen_conv_instrs(expr.type.typespec, var.type.typespec)
        if conv_instrs == None:
            self.error('Cannot convert {} to {}.'
                       .format(expr.type.name, var.type.name))
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
                             [Instr('pushi%', type_values[expr.type.typespec])])

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


    def is_function(self, name):
        return \
            name in builtin_functions or \
            name in self.declared_functions or \
            any(name == r.name for r in self.routines.values()
                if r.type == 'function')


class Assembler:
    def __init__(self, compiler):
        self.compiler = compiler


    def assemble(self, instrs):
        labels = {}
        p = 0
        def resolver(instr):
            def fix_label(label):
                if label[-1] in typespec_chars:
                    return label[:-1]
                else:
                    return label

            operands = []
            if instr.name in ['jmpf', 'jmpt']:
                operands = [labels[fix_label(instr.operands[0])] - INITIAL_ADDR - p]
            elif instr.name == 'jmp':
                operands = [labels[fix_label(instr.operands[0])]]
            elif instr.name == 'call':
                operands = [labels[fix_label(instr.operands[0])]]
            elif instr.name == 'unframe_r':
                routine, var = instr.operands
                frame_size = sum(v.size for v in routine.local_vars)
                operands = [frame_size, var.idx, var.type.get_size()]
            elif instr.name in ['frame', 'unframe']:
                frame_size = sum(v.size for v in instr.operands[0].local_vars)
                operands = [frame_size]
            elif instr.name == 'syscall':
                call_code = {
                    '__cls': 0x02,
                    '__concat': 0x03,
                    '__print': 0x04,
                }[instr.operands[0]]
                operands = [call_code]
            else:
                for o in instr.operands:
                    if isinstance(o, Label):
                        o = labels[fix_label(o)]
                    elif isinstance(o, Var):
                        o = o.idx
                    operands.append(o)

            return operands

        # phase 1: calculate label addresses
        for i in instrs:
            if isinstance(i, Label):
                if i.value[-1] in typespec_chars:
                    labels[i.value[:-1]] = INITIAL_ADDR + p
                else:
                    labels[i.value] = INITIAL_ADDR + p
            else:
                p += i.size

        # phase 2: assemble
        code = b''
        p = 0
        for i in instrs:
            if not isinstance(i, Instr):
                continue

            for operand in i.operands:
                if isinstance(operand, Label):
                    if operand.value[-1] in typespec_chars:
                        v = operand.value[:-1]
                    else:
                        v = operand.value
                    operand.address = labels[v]

            c = i.assemble(resolver)
            p += len(c)
            code += c

        return code
