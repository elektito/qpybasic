import struct
from collections import OrderedDict
from enum import IntEnum, unique
from lark import Lark, Token, Tree
from asm import Instr
from module import Module


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

lvalue_aliases = ['var_or_no_arg_func',
                  'func_call_or_idx',
                  'dotted']


def get_type_name(typespec):
    return {
        '%': 'INTEGER',
        '&': 'LONG',
        '!': 'SINGLE',
        '#': 'DOUBLE',
        '$': 'STRING',
    }[typespec]


def gen_conv_instrs(t1, t2):
    assert isinstance(t1, Type)
    assert isinstance(t2, Type)

    if t1 == t2:
        return []
    elif not t1.is_basic or not t2.is_basic:
        return None
    elif not t1.is_numeric or not t2.is_numeric:
        return None
    else:
        return [Instr('conv' + t1.typespec + t2.typespec)]


def get_literal_type(literal):
    if literal.startswith('"'): # string literal
        return Type('$')
    elif literal[-1] in '!#%&': # numeric literal with typespec
        # sanity check first
        if '.' in literal and literal[-1] not in '!#':
            raise CompileError(EC.INVALID_NUM_LITERAL,
                               'Invalid numeric literal: {}'.format(literal))
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
                raise CompileError(EC.INT_OUT_OF_RANGE)


def builtin_func_abs(parent, args):
    if len(args) != 1:
        raise CompileError(EC.INVALID_FUNC_NARGS)

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


@unique
class ErrorCodes(IntEnum):
    AS_CLAUSE_REQUIRED_ON_FIRST_DECL = 1
    INVALID_VAR_NAME = 2
    DUP_DEF = 3
    FUNC_RET_TYPE_MISMATCH = 4
    INT_OUT_OF_RANGE = 5
    SUB_CALLED_AS_FUNC = 6
    PARAM_TYPE_MISMATCH = 7
    DECL_ONLY_AT_TOP = 8
    INVALID_SUB_NAME = 9
    DECL_NOT_MATCH_DEF = 10
    CONFLICTING_DECL = 11
    SUB_DEF_NOT_MATCH_DECL = 12
    FUNC_DEF_NOT_MATCH_DECL = 13
    INVALID_FUNC_NARGS = 14
    INVALID_FOR_VAR = 15
    NEXT_VAR_NOT_MATCH_FOR = 16
    DUP_PARAM = 17
    NEXT_WITHOUT_FOR = 18
    NO_SUCH_FUNC = 19
    NO_SUCH_SUB = 20
    INVALID_NUM_LITERAL = 21
    CANNOT_CONVERT_TYPE = 22
    SUBSCRIPT_OUT_OF_RANGE = 23
    NOT_AN_ARRAY = 24
    WRONG_NUMBER_OF_DIMENSIONS = 25
    TYPE_MISMATCH = 26
    INVALID_LVALUE = 27
    INVALID_TYPE_NAME = 30
    ELEMENT_NOT_DEFINED = 31
    UNDEFINED_TYPE = 32
    INVALID_OPERATION = 33
    INVALID_USE_OF_DOT = 34
    INVALID_DEFTYPE = 35
    ARGUMENT_COUNT_MISMATCH = 36


    def __str__(self):
        return {
            self.AS_CLAUSE_REQUIRED_ON_FIRST_DECL:
            'AS clause required on first declaration',

            self.INVALID_VAR_NAME:
            'Invalid variable name',

            self.DUP_DEF:
            'Duplicate definition',

            self.FUNC_RET_TYPE_MISMATCH:
            'FUNCTION return type mismatch',

            self.INT_OUT_OF_RANGE:
            'Integer value out of range',

            self.SUB_CALLED_AS_FUNC:
            'SUB cannot be called like a FUNCTION',

            self.PARAM_TYPE_MISMATCH:
            'Parameter type mismatch',

            self.DECL_ONLY_AT_TOP:
            'DECLARE statement only valid in top-level',

            self.INVALID_SUB_NAME:
            'Invalid SUB name',

            self.DECL_NOT_MATCH_DEF:
            'DECLARE statement does not match definition',

            self.CONFLICTING_DECL:
            'Conflicting DECLARE statements',

            self.SUB_DEF_NOT_MATCH_DECL:
            'SUB definition does not match previous DECLARE',

            self.FUNC_DEF_NOT_MATCH_DECL:
            'FUNCTION definition does not match previous DECLARE',

            self.INVALID_FUNC_NARGS:
            'Invalid number of arguments passed to FUNCTION',

            self.INVALID_FOR_VAR:
            'Invalid FOR variable',

            self.NEXT_VAR_NOT_MATCH_FOR:
            'NEXT variable does not match FOR',

            self.DUP_PARAM:
            'Duplicate parameter',

            self.NEXT_WITHOUT_FOR:
            'NEXT without FOR',

            self.NO_SUCH_FUNC:
            'No such function',

            self.NO_SUCH_SUB:
            'No such sub-routine',

            self.INVALID_NUM_LITERAL:
            'Invalid numeric literal',

            self.CANNOT_CONVERT_TYPE:
            'Cannot convert type.',

            self.SUBSCRIPT_OUT_OF_RANGE:
            'Subscript out of range',

            self.NOT_AN_ARRAY:
            'Not an array',

            self.WRONG_NUMBER_OF_DIMENSIONS:
            'Wrong number of dimensions',

            self.TYPE_MISMATCH:
            'Type mismatch',

            self.INVALID_LVALUE:
            'Invalid L-Value',

            self.INVALID_TYPE_NAME:
            'Invalid TYPE name',

            self.ELEMENT_NOT_DEFINED:
            'Element not defined',

            self.UNDEFINED_TYPE:
            'Undefined type.',

            self.INVALID_OPERATION:
            'Invalid operation',

            self.INVALID_USE_OF_DOT:
            'Invalid use of DOT',

            self.INVALID_DEFTYPE:
            'Invalid DEFtype statement',

            self.ARGUMENT_COUNT_MISMATCH:
            'Argument-count mismatch'
        }.get(int(self), super().__str__())


# alias for easier typing
EC = ErrorCodes


class CompileError(Exception):
    def __init__(self, code, msg=None):
        assert isinstance(code, ErrorCodes)

        if not msg:
            msg = str(code)

        self.code = code

        super().__init__(msg)


class Type:
    basic_types = [
        'INTEGER',
        'LONG',
        'SINGLE',
        'DOUBLE',
        'STRING'
    ]

    def __init__(self, name, *, dimensions=None, elements=None):
        if name in typespec_chars:
            self.name = typespec_to_typename[name]
        else:
            self.name = name.upper()

        self.dimensions = dimensions
        self.elements = elements

        assert self.name in self.basic_types or self.elements


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
        return self.name in self.basic_types


    @property
    def is_numeric(self):
        numeric_types = [
            'INTEGER',
            'LONG',
            'SINGLE',
            'DOUBLE',
        ]
        return self.name in numeric_types and self.dimensions == None

    @property
    def is_array(self):
        return self.dimensions != None


    def get_size(self):
        if self.dimensions:
            array_size = 1
            for d_from, d_to in self.dimensions:
                array_size *= d_to - d_from + 1
        else:
            array_size = 1

        if self.is_basic:
            return {
                'INTEGER': 2,
                'LONG': 4,
                'SINGLE': 4,
                'DOUBLE': 8,
                'STRING': 4,
            }[self.name] * array_size
        else:
            base_size = sum(e_type.get_size()
                            for e_name, e_type in self.elements)
            return array_size * base_size


    def get_base_type_name(self):
        return self.name


    def __repr__(self):
        if self.typespec == '':
            name = f'User-Defined: {self.name}'
        else:
            name = self.name
        if self.dimensions:
            suffix = '('
            for d_from, d_to in self.dimensions:
                if suffix != '(':
                    suffix += ', '
                suffix += f'{d_from} TO {d_to}'
            suffix += ')'
        else:
            suffix = ''
        return f'<Type {name}{suffix}>'


    def __eq__(self, other):
        return isinstance(other, Type) and \
            self.name == other.name and \
            self.is_array == other.is_array


    def __hash__(self):
        return hash((self.name, self.is_array))


class Var:
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
        dims = '()' if self.type.is_array else ''
        if hasattr(self, 'idx'):
            return f'<Var {self.name}{self.type.typespec}{dims} idx={self.idx}{pass_type}>'
        else:
            return f'<Var {self.name}{self.type.typespec}{dims}{pass_type}>'


class Lvalue:
    def __init__(self, lv_list, *, compiler=None):
        assert lv_list

        self.compiler = compiler

        self.base = lv_list[0]
        self.ops = lv_list[1:]

        # convert the list of operations into groups of two, like:
        # [('IDX', args), ('DOT', element), ('DOT', element)]
        self.ops = list(zip(self.ops[::2], self.ops[1::2]))

        self.set_type()


    def set_type(self):
        # set self.type based on self.base and self.ops

        t = self.base.type

        for op, arg in self.ops:
            if op == 'IDX':
                t = self.compiler.get_type(t.get_base_type_name())
            elif op == 'DOT':
                type_elements = self.compiler.user_defined_types.get(t.name)
                if not type_elements:
                    raise CompileError(EC.INVALID_USE_OF_DOT)
                for e_name, e_type in type_elements:
                    if e_name == arg:
                        t = e_type
                        break
                else:
                    raise CompileError(EC.ELEMENT_NOT_DEFINED)
            else:
                assert False

        self.type = t


    def process_idx(self, prev_type, indices):
        # this function calculates the byte-offset indicated by the
        # given indices into the array the type of which is
        # prev_type. we start from the last index, and work
        # backwards. for the last, we just add the index value. for
        # the one before that, we multiply the size of the last
        # dimension by the value and then add it to what we had in the
        # previous step. this will continue until the first index.
        #
        # for the array:
        #    arr(n1 TO m1, n2 TO m2, n3 TO m3)
        # when trying to access arr(x, y, z) the offset would be:
        #    offset = (z - n3) + (y - n2) * (m3 - n3 + 1) + (x - n1) * (m3 - n3 + 1) * (m2 - n2 + 1)
        #    offset *= size_of_cell_type
        #
        # as a more concrete example, for this array:
        #    DIM arr(1 TO 5, 1 TO 5) AS LONG
        # when accessing arr(5, 5):
        #    offset = 4 + 5 * 4 => 24
        #    offset *= 4        => 96
        # given that the array has 100 cells, and (5, 5) is the last
        # 4-byte cell, the offset 96 calculated is indeed correct.

        base_type = self.compiler.get_type(prev_type.get_base_type_name())
        base_size = base_type.get_size()
        instrs = []

        # push last offset first
        instrs += indices[-1].instrs
        t = indices[-1].type.typespec
        instrs += [Instr(f'conv{t}_ul')]
        d_from, d_to = prev_type.dimensions[-1]
        if d_from != 0:
            instrs += [Instr('pushi_ul', d_from),
                       Instr('sub_ul')]

        # now add the offsets for the higher dimensions
        dim_size = 1
        for i, (d_from, d_to) in enumerate(prev_type.dimensions[:-1]):
            dim_size *= (d_to - d_from + 1)
            instrs += [Instr('pushi_ul', dim_size)]
            instrs += indices[i].instrs
            t = indices[i].type.typespec
            instrs += [Instr(f'conv{t}_ul')]

            instrs += [Instr('pushi_ul', d_from),
                       Instr('sub_ul')]

            instrs += [Instr('mul_ul'),
                       Instr('add_ul')]

        instrs += [Instr('pushi_ul', base_type.get_size()),
                   Instr('mul_ul')]

        return base_type, instrs


    def process_dot(self, prev_type, element):
        # returns instructions to add the offset for the given element
        # to the address on the stack.

        offset = 0
        for e_name, e_type in prev_type.elements:
            if e_name == element:
                break
            else:
                offset += e_type.get_size()
        else:
            raise CompileError(EC.ELEMENT_NOT_DEFINED)

        return e_type, [Instr('pushi_ul', offset)]


    def gen_instructions(self, iname, *, ref=False):
        # iname is either 'read' or 'write'.
        #
        # ref means whether only a reference is being calculated or
        # the value itself.

        assert iname == 'write' or not ref

        size = self.type.get_size()
        if self.ops == []:
            if self.base.byref:
                instrs = [Instr('readf4', self.base)]
                if not ref:
                    instrs += [Instr(f'{iname}i{size}')]
            else:
                if not ref:
                    if size in [1, 2, 4, 8]:
                        instrs = [Instr(f'{iname}f{size}', self.base)]
                    else:
                        instrs = [Instr(f'{iname}f_n', size, self.base)]
                else:
                    instrs = [Instr('pushfp', self.base)]
        else:
            # push base address on the stack
            if self.base.byref:
                instrs = [Instr('readf4', self.base)]
            else:
                instrs = [Instr('pushfp', self.base)]

            # calculate offset on the stack
            prev_type = self.base.type
            for op, arg in self.ops:
                if op == 'IDX':
                    new_type, new_instrs = self.process_idx(prev_type, arg)
                elif op == 'DOT':
                    new_type, new_instrs = self.process_dot(prev_type, arg)
                else:
                    assert False

                prev_type = new_type
                instrs += new_instrs

                # add base address and offset
                instrs += [Instr('add_ul')]

            # now do an indirect read/write (if asked for)
            if not ref:
                if size in [1, 2, 4, 8]:
                    instrs += [Instr(f'{iname}i{size}')]
                else:
                    instrs += [Instr(f'{iname}i_n', size)]

        return instrs


    def gen_read_instructions(self):
        return self.gen_instructions('read')


    def gen_write_instructions(self):
        return self.gen_instructions('write')


    def gen_ref_instructions(self):
        # generate instructions that put the address of the lvalue on
        # the stack.
        return self.gen_instructions('write', ref=True)


class Routine:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.instrs = []
        self.all_vars = []


    def add_var(self, var):
        self.all_vars.append(var)


    def lookup_var(self, name):
        for v in self.all_vars:
            if v.name == name:
                return v

        return None


    @property
    def local_vars(self):
        return [v for v in self.all_vars if v.klass == 'local']


    @property
    def params(self):
        return [v for v in self.all_vars if v.klass == 'param']


    def __repr__(self):
        routine_type = self.type.upper()
        if hasattr(self, 'ret_type'):
            ret_type = self.ret_type.name
            return f'<{routine_type} {self.name} {self.params} ret={ret_type}>'
        else:
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
        if fname[-1] in typespec_chars:
            called_type = self.parent.get_type_from_var_name(fname)
            fname = fname[:-1]
        else:
            called_type = None

        if fname in builtin_functions:
            f = builtin_functions[fname]
            self.type, self.instrs = f(self.parent, args)
            if called_type and self.type != called_type:
                raise CompileError(EC.FUNC_RET_TYPE_MISMATCH)
        else:
            defined = self.parent.routines.get(fname, None)
            declared = self.parent.declared_routines.get(fname, None)
            if defined:
                if len(args) != len(defined.params):
                    raise CompileError(EC.ARGUMENT_COUNT_MISMATCH)
                self.type = defined.ret_type
            elif declared:
                if declared.type == 'sub':
                    raise CompileError(EC.SUB_CALLED_AS_FUNC)
                if len(args) != len(declared.param_types):
                    raise CompileError(EC.ARGUMENT_COUNT_MISMATCH)
                self.type = declared.ret_type
            else:
                raise CompileError(EC.NO_SUCH_FUNC, f'No such function: {fname}')

            if called_type and self.type != called_type:
                raise CompileError(EC.FUNC_RET_TYPE_MISMATCH)

            i = len(args) - 1
            for a in reversed(args):
                lv = None
                if isinstance(a, Tree) and \
                   a.data == 'value' and \
                   isinstance(a.children[0], Tree) and \
                   a.children[0].data in lvalue_aliases:
                    lv = self.parent.create_lvalue_if_possible(a.children[0])

                if isinstance(lv, Lvalue):
                    # an lvalue: send byref
                    if fname in self.parent.routines:
                        if lv.type != self.parent.routines[fname].params[i].type:
                            raise CompileError(EC.PARAM_TYPE_MISMATCH)
                    else:
                        if lv.type != self.parent.declared_routines[fname].param_types[i]:
                            raise CompileError(EC.PARAM_TYPE_MISMATCH)
                    self.instrs += lv.gen_ref_instructions()
                else:
                    e = Expr(a, self.parent)
                    if fname in self.parent.routines:
                        typespec = self.parent.routines[fname].params[i].type.typespec
                    else:
                        typespec = self.parent.declared_routines[fname].param_types[i].typespec
                    v = self.parent.get_var(self.parent.gen_var('rvalue', typespec))
                    self.parent.gen_set_var_code(v, e)
                    self.instrs += [Instr('pushfp', v)]

                i -= 1

            self.instrs += [Instr('call', f'__function_{fname}')]


    def process_value(self, ast):
        v = ast.children[0]
        if isinstance(v, Tree):
            lv = self.parent.create_lvalue_if_possible(ast.children[0])
            if lv:
                self.instrs += lv.gen_read_instructions()
                self.type = lv.type
            else:
                self.process_function_call(ast.children[0])
        elif v.type == 'STRING_LITERAL':
            self.parent.add_string_literal(v[1:-1])
            self.instrs += [Instr('pushi$', v.value)]
            self.type = Type('$')
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
        elif not left.type.is_numeric or not right.type.is_numeric:
            raise CompileError(EC.INVALID_OPERATION)

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
        instrs += gen_conv_instrs(left.type, Type(t))

        instrs += right.instrs
        instrs += gen_conv_instrs(right.type, Type(t))

        instrs += [Instr(op + t)]

        self.type = Type(t)
        self.instrs += instrs


    def compare_op(self, op, left, right):
        self.binary_op('sub', left, right)
        self.instrs += gen_conv_instrs(self.type, Type('%'))
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


class DeclaredRoutine:
    def __init__(self, type, name, param_types, ret_type=None):
        self.type = type
        self.name = name
        self.param_types = param_types
        self.ret_type = ret_type


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
        code += '\n'

        self.instrs = []
        self.cur_routine = Routine('__main', 'sub')
        self.routines = {'__main': self.cur_routine}
        self.gen_labels = {}
        self.gen_vars = {}
        self.endif_labels = []
        self.string_literals = {}
        self.last_string_literal_idx = 0
        self.default_array_base = 1
        self.user_defined_types = {}
        self.default_types = {}

        # map the names of declared subs/functions to a
        # DeclaredRoutine object which contains parameter types and
        # possibly return value.
        self.declared_routines = {}

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
        return Module.create(self.bytecode, self.string_literals)


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
            if isinstance(ast.children[0], Token) and \
               ast.children[0].type == 'CALL_KW':
                _, sub_name = ast.children
                args = Tree('argument_list', [])
            else:
                sub_name, args = ast.children
        else:
            _, sub_name, args = ast.children

        # due to some grammar conflicts, we're using a
        # 'possibly_lvalue' as the sub name (instead of the more
        # sensible ID). so here, we check whether it is in fact
        # actually an ID.
        if isinstance(sub_name, Tree):
            sub_name = sub_name.children[0].value
        else:
            sub_name = sub_name.value

        defined = self.routines.get(sub_name, None)
        declared = self.declared_routines.get(sub_name, None)
        if declared and declared.type != 'sub':
            declared = None

        if defined:
            if len(args.children) != len(defined.params):
                raise CompileError(EC.ARGUMENT_COUNT_MISMATCH)
        elif declared:
            if len(args.children) != len(declared.param_types):
                raise CompileError(EC.ARGUMENT_COUNT_MISMATCH)
        elif not defined and not declared:
            raise CompileError(EC.NO_SUCH_SUB,
                               f'No such sub-routine: {sub_name}')

        i = len(args.children) - 1
        for a in reversed(args.children):
            lv = None
            if isinstance(a, Tree) and \
               a.data == 'value' and \
               isinstance(a.children[0], Tree) and \
               a.children[0].data in lvalue_aliases:
                lv = self.create_lvalue_if_possible(a.children[0])

            if isinstance(lv, Lvalue):
                # just a variable: send byref
                if sub_name in self.routines:
                    if lv.type != self.routines[sub_name].params[i].type:
                        raise CompileError(EC.PARAM_TYPE_MISMATCH)
                else:
                    if lv.type != self.declared_routines[sub_name].param_types[i]:
                        raise CompileError(EC.PARAM_TYPE_MISMATCH)
                self.instrs += lv.gen_ref_instructions()
            else:
                e = Expr(a, self)
                if sub_name in self.routines:
                    typespec = self.routines[sub_name].params[i].type.typespec
                else:
                    typespec = self.declared_routines[sub_name].param_types[i].typespec
                v = self.get_var(self.gen_var('rvalue', typespec))
                self.gen_set_var_code(v, e)
                self.instrs += [Instr('pushfp', v)]

            i -= 1

        self.instrs += [Instr('call', f'__sub_{sub_name}')]


    def process_cls_stmt(self, ast):
        self.instrs += [Instr('syscall', '__cls')]


    def process_declare_stmt(self, ast):
        if self.cur_routine.name != '__main':
            raise CompileError(EC.DECL_ONLY_AT_TOP)

        _, routine_type, name, params = ast.children
        name = name.value
        if name[-1] in typespec_chars:
            if routine_type == 'sub':
                raise CompileError(EC.INVALID_SUB_NAME)
            else:
                ret_type = self.get_type_from_var_name(name)
                name = name[:-1]
        else:
            if routine_type == 'sub':
                ret_type = None
            else:
                ret_type = self.get_type_from_var_name(name)

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
                ptype = self.get_type_from_var_name(pname)

            param_types.append(ptype)

        if name in self.routines:
            defined_param_types = [v.type for v in self.routines[name].params]
            if param_types != defined_param_types:
                raise CompileError(EC.DECL_NOT_MATCH_DEF)

        if name in self.declared_routines:
            if self.declared_routines[name].type != routine_type or \
               self.declared_routines[name].ret_type != ret_type or \
               self.declared_routines[name].param_types != param_types:
                raise CompileError(EC.CONFLICTING_DECL)
        else:
            dr = DeclaredRoutine(routine_type, name, param_types, ret_type)
            self.declared_routines[name] = dr


    def process_deftype_stmt(self, ast):
        keyword, *ranges = ast.children
        keyword = keyword.upper()
        t = {
            'DEFINT': Type('%'),
            'DEFLNG': Type('&'),
            'DEFSNG': Type('!'),
            'DEFDBL': Type('#'),
            'DEFSTR': Type('$'),
        }[keyword]

        for r in ranges:
            if len(r.children) == 2:
                r_from, r_to = r.children
            else:
                r_from, = r.children
                r_to = r_from

            if len(r_from) != 1 or len(r_to) != 1:
                raise CompileError(EC.INVALID_DEFTYPE)

            for i in range(ord(r_from), ord(r_to) + 1):
                self.default_types[chr(i)] = t


    def process_dim_stmt(self, ast):
        if len(ast.children) == 2:
            _, name = ast.children
            self.dim_var(name)
        else:
            _, name, dimensions,  _, typename = ast.children
            if dimensions.children:
                dimensions = self.parse_dimensions(dimensions)
            else:
                dimensions = None
            typename = typename.children[0].value
            self.dim_var(name,
                         type=self.get_type(typename, dimensions=dimensions),
                         dimensions=dimensions)


    def parse_dimensions(self, ast):
        dimensions = []

        for d in ast.children:
            if len(d.children) == 1:
                d_from = self.default_array_base
                d_to = d.children[0]
            else:
                d_from, _, d_to = d.children

            d_from, d_to = int(d_from), int(d_to)

            if d_to < d_from:
                raise CompileError(EC.SUBSCRIPT_OUT_OF_RANGE)

            dimensions.append((d_from, d_to))

        return dimensions


    def process_end_stmt(self, ast):
        self.instrs += [Instr('end')]


    def process_for_block(self, ast):
        _, var, start, _, end, step, body, next_stmt = ast.children
        var = self.get_var(var)
        if var.type.typespec == '$':
            raise CompileError(EC.INVALID_FOR_VAR)

        if len(next_stmt.children) == 2:
            # "NEXT var" is used. Check if NEXT variable matches FOR
            # variable.
            next_var = self.get_var(next_stmt.children[1])
            if next_var != var:
                raise CompileError(EC.NEXT_VAR_NOT_MATCH_FOR)

        step_var = self.get_var(self.gen_var('for_step', var.type.typespec))
        if step.children:
            e = Expr(step.children[1], self)
            self.gen_set_var_code(step_var, e)
        else:
            self.instrs += [Instr(f'pushi{var.type.typespec}', 1)]
            self.instrs += Lvalue([step_var]).gen_write_instructions()

        end_var = self.get_var(self.gen_var('for_end', var.type.typespec))
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
        self.instrs += gen_conv_instrs(var.type, Type('%'))
        self.instrs += [Instr('gt'),
                        Instr('jmpt', end_label)]
        self.compile_ast(body)
        self.instrs += [Instr(f'readf{var.size}', var),
                        Instr(f'readf{step_var.size}', step_var),
                        Instr(f'add{var.type.typespec}')]
        self.instrs += Lvalue([var]).gen_write_instructions()
        self.instrs += [Instr('jmp', top_label),
                        Label(end_label)]


    def process_sub_block(self, ast):
        _, name, params, body, _, _ = ast.children

        if name[-1] in typespec_chars:
            raise CompileError(EC.INVALID_SUB_NAME)

        saved_instrs = self.instrs
        self.instrs = []
        self.cur_routine = Routine(name.value, 'sub')

        self.instrs += [Label(f'__sub_{name}'),
                        Instr('frame', self.cur_routine)]

        for p in params.children:
            self.parse_param_def(p)

        if name in self.declared_routines:
            defined_param_types = [v.type for v in self.cur_routine.params]
            r = self.declared_routines[name]
            if r.type != 'sub' or \
               defined_param_types != r.param_types:
                raise CompileError(EC.SUB_DEF_NOT_MATCH_DECL)

        self.routines[name.value] = self.cur_routine
        self.compile_ast(body)

        arg_size = sum(v.size for v in self.cur_routine.params)
        self.instrs += [Instr('unframe', self.cur_routine),
                        Instr('ret', arg_size)]
        self.cur_routine.instrs = self.instrs

        self.cur_routine = self.routines['__main']
        self.instrs = saved_instrs


    def parse_param_def(self, p):
        if len(p.children) == 3:
            # form: var AS type
            pname, _, ptype = p.children
            pname = pname.value
            ptype = self.get_type(ptype.children[0].value)
        else:
            # form: var$
            pname = p.children[0].value
            ptype = None

        if any(pname == i.name for i in self.cur_routine.params):
            raise CompileError(EC.DUP_PARAM)

        self.dim_var(pname, type=ptype, klass='param', byref=True)


    def process_function_block(self, ast):
        _, name, params, body, _, _ = ast.children

        ftype = self.get_type_from_var_name(name.value)
        name = name.value
        if name[-1] in typespec_chars:
            name = name[:-1]

        saved_instrs = self.instrs
        self.instrs = []
        self.cur_routine = Routine(name, 'function')
        self.cur_routine.ret_type = ftype

        self.instrs += [Label(f'__function_{name}'),
                        Instr('frame', self.cur_routine)]

        for p in params.children:
            self.parse_param_def(p)

        # create a variable with the same name as the function. this
        # will be used for returning values.
        self.dim_var(name, type=ftype)

        # mark the variable as used so that an index is allocated to
        # it.
        ret_var = self.get_var(name)

        # initialize the return variable.
        if ftype.name == 'STRING':
            self.instrs += [Instr(f'pushi$'), '""']
        else:
            self.instrs += [Instr(f'pushi{ftype.typespec}', 0)]
        self.instrs += Lvalue([ret_var]).gen_write_instructions()

        if name in self.declared_routines:
            defined_param_types = [v.type for v in self.cur_routine.params]
            if defined_param_types != self.declared_routines[name].param_types or \
               ftype != self.declared_routines[name].ret_type:
                raise CompileError(EC.FUNC_DEF_NOT_MATCH_DECL)

        self.routines[name] = self.cur_routine
        self.compile_ast(body)

        arg_size = sum(v.size for v in self.cur_routine.params)
        self.instrs += [Instr('unframe_r', self.cur_routine, ret_var),
                        Instr(f'ret_r', arg_size, ftype.get_size())]
        self.cur_routine.instrs = self.instrs

        self.cur_routine = self.routines['__main']
        self.instrs = saved_instrs


    def process_type_block(self, ast):
        _, type_name, *element_defs, _, _ = ast.children
        if type_name[-1] in typespec_chars:
            raise CompileError(EC.INVALID_TYPE_NAME)

        if type_name in self.user_defined_types:
            raise CompileError(EC.DUP_DEF)

        type_name = type_name.value.upper()

        if len(element_defs) == 0:
            raise CompileError(EC.ELEMENT_NOT_DEFINED)

        elements = []
        for e in element_defs:
            if len(e.children) == 1:
                e_name, = e.children
                e_type = self.get_default_type(e_name)
            else:
                e_name, _, e_type = e.children
                e_type = self.get_type(e_type.children[0])

            e_name = e_name.value
            elements.append((e_name, e_type))

        self.user_defined_types[type_name] = elements


    def process_next_stmt(self, ast):
        # NEXT statements with a matching FOR will be processed by the
        # process_for_block function. This will only be called when
        # there is a NEXT without FOR.
        raise CompileError('NEXT without FOR.')


    def process_goto_stmt(self, ast):
        _, target = ast.children
        if target.type == 'ID':
            self.instrs += [Instr('jmp', target)]
        else:
            self.instrs += [Instr('jmp', f'__lineno_{target.value}')]


    def process_if_block(self, ast):
        _, cond, _, then_body, rest, _, _ = ast.children
        cond = Expr(cond, self)
        self.instrs += cond.instrs + gen_conv_instrs(cond.type, Type('%'))

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

        dest, expr = ast.children
        lv = self.create_lvalue_if_possible(dest)
        if lv == None:
            if dest.data == 'var_or_no_arg_func':
                # this is for the case of assigning to the function
                # name in functions.
                name = dest.children[0].value
                var = self.get_var(name)
                lv = Lvalue([var])

        if lv == None:
            raise CompileError(EC.INVALID_LVALUE)

        expr = Expr(expr, self)
        self.instrs += expr.instrs
        if expr.type != lv.type:
            conv_instrs = gen_conv_instrs(expr.type, lv.type)
            if conv_instrs == None:
                raise CompileError(EC.TYPE_MISMATCH)
            self.instrs += conv_instrs
        self.instrs += lv.gen_write_instructions()


    def gen_set_var_code(self, var, expr):
        conv_instrs = gen_conv_instrs(expr.type, var.type)
        if conv_instrs == None:
            raise CompileError(
                EC.CANNOT_CONVERT_TYPE,
                f'Cannot convert {expr.type.name} to {var.type.name}.')
        else:
            self.instrs += expr.instrs + conv_instrs

            lv = Lvalue([var])
            self.instrs += lv.gen_write_instructions()


    def create_lvalue_if_possible(self, ast):
        # this function receive a tree for the 'possibly_lvalue' rule,
        # and attempts to create an lvalue from it. if not possible,
        # None is returned.

        flattened = self.flatten_lvalue_tree(ast)
        assert len(flattened) % 2 == 1

        if len(flattened) == 1 and \
           self.is_function(flattened[0]):
            return None
        elif len(flattened) == 3 and \
             flattened[1] == 'IDX' and \
             self.is_function(flattened[0]):
            return None
        else:
            flattened[0] = self.get_var(flattened[0])

            for i in range(1, len(flattened), 2):
                op = flattened[i]
                if op == 'IDX':
                    indices = [Expr(i, self) for i in flattened[i + 1]]
                    flattened[i + 1] = indices

            return Lvalue(flattened, compiler=self)


    def flatten_lvalue_tree(self, tree):
        if tree.data == 'var_or_no_arg_func':
            return [tree.children[0].value]
        elif tree.data == 'func_call_or_idx':
            return [tree.children[0].value, 'IDX', tree.children[1].children]
        elif tree.data == 'dotted':
            return \
                self.flatten_lvalue_tree(tree.children[0]) + \
                ["DOT"] + \
                self.flatten_lvalue_tree(tree.children[1])
        else:
            assert False, 'This should not have happened.'


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
        ftype = self.get_type_from_var_name(name)
        if name[-1] in typespec_chars:
            name = name[:-1]
        return \
            name in builtin_functions or \
            (name in self.declared_routines and
             self.declared_routines[name].ret_type == ftype and
             self.declared_routines[name].type == 'function') or \
            any(name == r.name for r in self.routines.values()
                if r.type == 'function')

    def dim_var(self, used_name, *,
                klass='local', byref=False, type=None, dimensions=None):
        assert klass in ['local', 'param']
        assert klass == 'param' or not byref

        if used_name[-1] in typespec_chars:
            name = used_name[:-1]
            typespec = used_name[-1]

            if type:
                raise CompileError(EC.INVALID_VAR_NAME)

            type = Type(typespec, dimensions=dimensions)
        else:
            name = used_name
            typespec = None
            if not type:
                type = self.get_default_type(used_name)

        containers = [self.cur_routine]
        for c in containers:
            var = c.lookup_var(name)
            if var:
                if var.no_type_dimmed and not typespec:
                    raise CompileError(EC.AS_CLAUSE_REQUIRED_ON_FIRST_DECL)
                else:
                    raise CompileError(EC.DUP_DEF)

        if klass in ['local', 'param']:
            container = self.cur_routine
        else:
            assert False, 'This should not happen!'

        var = Var()
        var.name = name
        var.type = type
        var.container = container
        var.no_type_dimmed = (typespec != None)
        var.klass = klass
        var.byref = byref

        container.add_var(var)

        return var


    def get_var(self, used_name):
        if used_name[-1] in typespec_chars:
            name = used_name[:-1]
            typespec = used_name[-1]
        else:
            name = used_name
            typespec = None

        containers = [self.cur_routine]
        for c in containers:
            var = c.lookup_var(name)
            if var:
                break
        else:
            var = self.dim_var(used_name)

        if var.klass == 'param':
            pidx = var.container.params.index(var)
            var.idx = 8 + sum(v.size for v in var.container.params[:pidx])
        else:
            lidx = var.container.local_vars.index(var)
            var.idx = -sum(v.size for v in var.container.local_vars[:lidx+1])

        return var


    def get_type(self, name, *, dimensions=None):
        name = name.upper()
        if name in Type.basic_types:
            return Type(name, dimensions=dimensions)

        elements = self.user_defined_types.get(name, None)
        if not elements:
            raise CompileError(EC.UNDEFINED_TYPE)

        return Type(name, elements=elements, dimensions=dimensions)


    def get_type_from_var_name(self, var_name):
        if var_name[-1] in typespec_chars:
            return Type(var_name[-1])
        else:
            return self.get_default_type(var_name)


    def get_default_type(self, var_name):
        t = self.default_types.get(var_name[0], None)
        return t if t else Type('SINGLE')


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
                operands = [labels[fix_label(instr.operands[0])] - Module.INITIAL_ADDR - p]
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
            elif instr.name == 'pushi$':
                s = instr.operands[0][1:-1]
                operands = [Module.STRING_ADDR + self.compiler.string_literals[s]]
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
                    labels[i.value[:-1]] = Module.INITIAL_ADDR + p
                else:
                    labels[i.value] = Module.INITIAL_ADDR + p
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
