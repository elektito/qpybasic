import struct
import asm
from collections import OrderedDict
from enum import IntEnum, unique
from copy import deepcopy
from lark import Lark, Token, Tree
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
            v = parse_int_literal(literal)
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
    const_value = abs(e.const_value) if e.is_const else None
    instrs = e.instrs
    end_label = parent.gen_label('abs_end')
    instrs += [Instr(f'dup{e.type.get_size()}'),
               Instr(f'sgn{e.type.typespec}'),
               Instr('lt'),
               Instr('jmpf', end_label),
               Instr(f'neg{e.type.typespec}'),
               Label(end_label)]
    return e.type, instrs, e.is_const, const_value


def parse_int_literal(value):
    value = value.lower()
    if value.startswith('&h'):
        return int(value[2:], base=16)
    else:
        if value[-1] in '%&':
            ret = int(value[:-1])

            if (value[-1] == '%' and (ret < -32768 or ret > 32767)) or \
               (value[-1] == '&' and (ret < -2**31 or ret > 2**31 - 1)):
                raise CompileError(EC.ILLEGAL_NUMBER)
        else:
            ret = int(value)

        return ret


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
    EXIT_SUB_INVALID = 37
    EXIT_FUNC_INVALID = 38
    EXIT_FOR_INVALID = 39
    INVALID_CONSTANT = 40
    CANNOT_ASSIGN_TO_CONST = 41
    INVALID_TYPE_ELEMENT = 42
    INVALID_ARRAY_BOUNDS = 43
    EXIT_DO_INVALID = 44
    STRING_NOT_ALLOWED_IN_TYPE = 45
    ILLEGAL_NUMBER = 46
    BLOCK_END_MISMATCH = 47
    SYNTAX_ERROR = 48


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
            'Argument-count mismatch',

            self.EXIT_SUB_INVALID:
            'Exit sub only valid inside a SUB',

            self.EXIT_FUNC_INVALID:
            'Exit sub only valid inside a FUNCTION',

            self.EXIT_FOR_INVALID:
            'EXIT not within FOR...NEXT',

            self.INVALID_CONSTANT:
            'Invalid constant',

            self.CANNOT_ASSIGN_TO_CONST:
            'Cannot assign to const',

            self.INVALID_TYPE_ELEMENT:
            'Invalid type element.',

            self.INVALID_ARRAY_BOUNDS:
            'Invalid array bounds',

            self.EXIT_DO_INVALID:
            'EXIT not within DO...LOOP',

            self.STRING_NOT_ALLOWED_IN_TYPE:
            'Variable-length string not allowed in user-defined type.',

            self.ILLEGAL_NUMBER:
            'Illegal number',

            self.BLOCK_END_MISMATCH:
            'Block end mismatch',

            self.SYNTAX_ERROR:
            'Syntax error',
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

    def __init__(self, name, *, is_array=False, elements=None):
        if name in typespec_chars:
            self.name = typespec_to_typename[name]
        else:
            self.name = name.upper()

        self.is_array = is_array
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
        return self.name in numeric_types and not self.is_array


    def get_size(self):
        if self.is_array:
            if self.dynamic:
                return 4
            else:
                n = 1
                for d_from, d_to in self.dims:
                    n *= d_to - d_from + 1
                hdr_size = 2 + 2 * 2 * len(self.dims)
                return hdr_size + n * Type(self.get_base_type_name()).get_size()

        if self.is_basic:
            return {
                'INTEGER': 2,
                'LONG': 4,
                'SINGLE': 4,
                'DOUBLE': 8,
                'STRING': 4,
            }[self.name]
        else:
            return sum(e_type.get_size()
                       for e_name, e_type in self.elements)


    def get_base_type_name(self):
        return self.name


    def get_base_size(self):
        # if this is an array, return the size of its base type,
        # otherwise return its size.

        if self.is_array:
            return Type(var.type.get_base_type_name()).get_size()
        else:
            return self.get_size()


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


class Var:
    @property
    def size(self):
        if self.byref:
            return 4
        else:
            return self.type.get_size()


    def gen_free_instructions(self):
        # variables that need freeing:
        #    - anything that is on the heap (dynamic arrays)
        #    - variable-length strings (which are on the heap)
        #    - arrays of variable-length strings (whether static or dynamic)
        if not self.on_heap and \
           not (self.type.is_array and \
                self.type.get_base_type_name() == 'STRING') and \
           not self.type == Type('$'):
            return []

        instrs = []
        instrs += Lvalue(self).gen_ref_instructions()

        if self.type.is_array and \
           self.type.get_base_type_name() == 'STRING':
            # an array of variable-length strings. free the strings first.

            # first duplicate the pointer on the stack because we're
            # going to use it with two syscalls.
            instrs += [Instr('dup4'),
                       Instr('syscall', '__free_strings_in_array')]

        if self.type == Type('$'):
            # read the actual pointer
            instrs += [Instr('readi4')]

        if self.on_heap or self.type == Type('$'):
            instrs += [Instr('syscall', '__free')]

        return instrs


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
    def __init__(self, base, *, indices=None, dots=[], compiler=None):
        self.compiler = compiler

        self.base = base
        self.indices = indices
        self.dots = dots

        self.set_type()


    def set_type(self):
        # set self.type based on self.base and self.indices, and
        # self.dots

        t = self.base.type
        if self.indices:
            t = self.compiler.get_type(t.get_base_type_name())

        for element in self.dots:
            type_elements = self.compiler.user_defined_types.get(t.name)
            if not type_elements:
                raise CompileError(EC.INVALID_USE_OF_DOT)
            for e_name, e_type in type_elements:
                if e_name == element:
                    t = e_type
                    break
            else:
                raise CompileError(EC.ELEMENT_NOT_DEFINED)

        self.type = t


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


    def process_base_addr(self):
        # returns instructions to add the address of the base variable
        # on the stack.

        instrs = []

        if self.base.klass in ['param', 'local']:
            if self.base.byref:
                addr_instrs = [Instr('readf4', self.base)]
            else:
                addr_instrs = [Instr('pushfp', self.base)]
        else:
            addr_instrs = [Instr('pushi_ul', self.base)]

        if self.indices:
            for i in reversed(self.indices):
                instrs += i.instrs
                if i.type.typespec != '%':
                    if i.type.is_numeric:
                        instrs += [Instr(f'conv{i.type.typespec}%')]
                    else:
                        raise CompileError(EC.TYPE_MISMATCH)

            if self.base.type.is_array:
                t = self.base.type.get_base_type_name()
                element_size = self.compiler.get_type(t).get_size()
            else:
                element_size = self.base.type.get_size()
            instrs += [Instr('pushi%', len(self.indices)),
                       Instr('pushi%', element_size)]
            instrs += addr_instrs
            instrs += [Instr('syscall', '__access_array')]
        else:
            instrs = addr_instrs

        return instrs


    def process_param_or_local(self, iname, ref):
        # returns instructions to put the value of the base variable
        # (or its address, when ref=True) on the stack, when base
        # variable is a local variable or a parameter.

        size = self.type.get_size()
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

        return instrs


    def process_const(self):
        # returns the instructions to put the value of the base, when
        # it is a constant.

        if self.base.type.is_numeric:
            instrs = [Instr(f'pushi{self.base.type.typespec}', self.base.const_value)]
        else:
            instrs = [Instr(f'pushi$', f'"{self.base.const_value}"')]

        return instrs


    def process_shared(self, iname, ref):
        # returns instructions to put the value of the base variable
        # (or its address, when ref=True) on the stack when the base
        # variable is shared.

        instrs = [Instr('pushi_ul', self.base)]
        if not ref:
            instrs += [Instr(f'{iname}i{self.base.type.get_base_size()}')]

        return instrs


    def gen_instructions(self, iname, *, ref=False):
        # iname is either 'read' or 'write'.
        #
        # ref means whether only a reference is being calculated or
        # the value itself.

        assert iname == 'write' or not ref

        size = self.type.get_size()
        if not self.indices and not self.dots:
            if self.base.klass in ['param', 'local']:
                instrs = self.process_param_or_local(iname, ref)
            elif self.base.klass == 'shared':
                instrs = self.process_shared(iname, ref)
            elif self.base.klass == 'const':
                if iname != 'read':
                    raise CompileError(EC.CANNOT_ASSIGN_TO_CONST)
                instrs = self.process_const()
            else:
                assert False
        else:
            # push base address on the stack
            instrs = self.process_base_addr()

            # calculate offset on the stack
            prev_type = self.base.type
            for element in self.dots:
                new_type, new_instrs = self.process_dot(prev_type, element)

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


class VarContainer:
    def __init__(self, compiler):
        self.compiler = compiler
        self.vars = []


    def add_var(self, var):
        self.vars.append(var)


    def lookup_var(self, name, typespec, *, get_all=False):
        matching = [v for v in self.vars if v.name == name]
        if get_all:
            return matching

        for v in matching:
            if typespec and v.type.typespec == typespec:
                return v
            if not typespec and not v.no_type_dimmed:
                return v
            if not typespec and \
               v.type.typespec == self.compiler.get_default_type(name).typespec:
                return v

        return None


class SharedVarContainer(VarContainer):
    def get_var_offset(self, var):
        offset = 0
        for v in self.vars:
            if v == var:
                break
            offset += v.type.get_size()
        else:
            assert False

        return offset


class Routine(VarContainer):
    def __init__(self, name, type, compiler):
        super().__init__(compiler)

        self.name = name
        self.type = type
        self.compiler = compiler
        self.instrs = []
        self.init_instrs = []


    def gen_free_instructions(self):
        instrs = []
        for v in self.vars:
            if v.klass == 'local':
                instrs += v.gen_free_instructions()
        return instrs


    @property
    def local_vars(self):
        return [v for v in self.vars if v.klass == 'local']


    @property
    def params(self):
        return [v for v in self.vars if v.klass == 'param']


    @property
    def exit_label(self):
        return f'__{self.type}_{self.name}_exit'


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
        self._instrs = []
        self.is_const = False
        self.const_value = None
        self.compile_ast(ast)


    def compile_ast(self, ast):
        getattr(self, 'process_' + ast.data)(ast)


    def process_not_expr(self, ast):
        _, arg = ast.children
        e = Expr(arg, self.parent)
        if e.type.name == 'INTEGER':
            self._instrs += e.instrs + [Instr('not%')]
            self.type = e.type
        elif e.type.name == 'LONG':
            self._instrs += e.instrs + [Instr('not&')]
            self.type = e.type
        elif e.type.name in ['SINGLE', 'DOUBLE']:
            self._instrs += e.instrs
            self._instrs += [Instr(f'conv{e.type.typespec}&'),
                             Instr('not&')]
            e.type = Type('&')
        else:
            raise CompileError(EC.TYPE_MISMATCH)

        self.is_const = e.is_const
        if self.is_const:
            self.const_value = ~e.const_value


    def process_and_expr(self, ast):
        left, _, right = ast.children
        left = Expr(left, self.parent)
        right = Expr(right, self.parent)
        self.binary_logical_op('and', left, right)


    def process_or_expr(self, ast):
        left, _, right = ast.children
        left = Expr(left, self.parent)
        right = Expr(right, self.parent)
        self.binary_logical_op('or', left, right)


    def process_xor_expr(self, ast):
        left, _, right = ast.children
        left = Expr(left, self.parent)
        right = Expr(right, self.parent)
        self.binary_logical_op('xor', left, right)


    def process_eqv_expr(self, ast):
        left, _, right = ast.children
        left = Expr(left, self.parent)
        right = Expr(right, self.parent)
        self.binary_logical_op('xor', left, right)
        self._instrs += [Instr(f'not{self.type.typespec}')]


    def process_eqv_expr(self, ast):
        left, _, right = ast.children
        left = Expr(left, self.parent)
        right = Expr(right, self.parent)
        self.binary_logical_op('xor', left, right)
        self._instrs += [Instr(f'not{self.type.typespec}')]

        # binary_logical_op has already done constant folding, but
        # here we change the value afterward, so we need to do it
        # again.
        self.is_const = left.is_const and right.is_const
        if self.is_const:
            self.calc_binary_const('eqv', left, right)


    def process_imp_expr(self, ast):
        left, _, right = ast.children
        left = Expr(left, self.parent)
        right = Expr(right, self.parent)
        self.binary_logical_op('or', left, right, inv_left=True)

        # binary_logical_op has already done constant folding, but
        # here we change the value afterward, so we need to do it
        # again.
        self.is_const = left.is_const and right.is_const
        if self.is_const:
            self.calc_binary_const('imp', left, right)


    def process_expr_add(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.binary_op('add', left, right)


    def process_expr_sub(self, ast):
        left, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.binary_op('sub', left, right)


    def process_mod_expr(self, ast):
        left, _, right = ast.children
        left, right = Expr(left, self.parent), Expr(right, self.parent)
        self.binary_op('mod', left, right)


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
        if len(ast.children) == 4:
            fname, _, args, _ = ast.children
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
            self.type, self._instrs, self.is_const, self.const_value = f(self.parent, args)
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
                   a.children[0].data == 'lvalue':
                    lv = self.parent.create_lvalue_if_possible(a.children[0])

                if isinstance(lv, Lvalue):
                    # an lvalue: send byref
                    if fname in self.parent.routines:
                        if lv.type != self.parent.routines[fname].params[i].type:
                            raise CompileError(EC.PARAM_TYPE_MISMATCH)
                    else:
                        if lv.type != self.parent.declared_routines[fname].param_types[i]:
                            raise CompileError(EC.PARAM_TYPE_MISMATCH)
                    self._instrs += lv.gen_ref_instructions()
                else:
                    e = Expr(a, self.parent)
                    if fname in self.parent.routines:
                        typespec = self.parent.routines[fname].params[i].type.typespec
                    else:
                        typespec = self.parent.declared_routines[fname].param_types[i].typespec
                    v = self.parent.get_var(self.parent.gen_var('rvalue', typespec))
                    self.parent.gen_set_var_code(v, e)
                    self._instrs += [Instr('pushfp', v)]

                i -= 1

            self._instrs += [Instr('call', f'__function_{fname}')]


    def process_value(self, ast):
        v = ast.children[0]
        if isinstance(v, Tree):
            lv = self.parent.create_lvalue_if_possible(ast.children[0])
            if lv:
                self._instrs += lv.gen_read_instructions()
                self.type = lv.type
                if not lv.indices and not lv.dots:
                    self.is_const = (lv.base.klass == 'const')
                    if self.is_const:
                        self.const_value = lv.base.const_value
                if lv.type.name == 'STRING':
                    self._instrs += [Instr('syscall', '__strcpy')]
            else:
                self.process_function_call(ast.children[0].children[0])
        elif v.type == 'STRING_LITERAL':
            self.parent.add_string_literal(v[1:-1])
            self._instrs += [Instr('pushi$', v.value)]
            self.type = Type('$')
            self.is_const = True
            self.const_value = v.value[1:-1]
            self._instrs += [Instr('syscall', '__strcpy')]
        elif v.type == 'NUMERIC_LITERAL':
            t = get_literal_type(v.value)
            if t.name.lower() in ['integer', 'long']:
                value = parse_int_literal(v.value)
            else:
                if v.value[-1] in '!#':
                    value = float(v.value[:-1])
                else:
                    value = float(v.value)
            self._instrs += [Instr('pushi' + t.typespec, value)]
            self.type = t
            self.is_const = True
            self.const_value = int(value) if t.typespec in ('%', '&') \
                               else float(value)
        else:
            assert False, 'This should not have happened.'


    def process_negation(self, ast):
        arg = ast.children[0]
        e = Expr(arg, self.parent)
        self._instrs += e.instrs + [Instr(f'neg{e.type.typespec}')]
        self.type = e.type
        self.is_const = e.is_const
        if self.is_const:
            self.const_value = -e.const_value


    def calc_binary_const(self, op, left, right):
        l, r = left.const_value, right.const_value

        # we use lambdas in the following so that potentially illegal
        # values (like division by zero) are not performed. For
        # example, without the lambdas, attempting to calculate the
        # expression '1 - 0' would cause a division by zero because we
        # would calculate all of the values, regardless.
        result = {
            'add': lambda: l + r,
            'sub': lambda: l - r,
            'mul': lambda: l * r,
            'div': lambda: l / r,
            'and': lambda: int(l) & int(r),
            'or':  lambda: int(l) | int(r),
            'xor': lambda: int(l) ^ int(r),
            'eqv': lambda: ~(int(l) ^ int(r)),
            'imp': lambda: (~int(l)) | int(r),
            'mod': lambda: l % r,
        }[op]()

        if all(v.type.typespec in ('%', '&') for v in (left, right)):
            result = int(result)

        self.const_value = result


    def binary_op(self, op, left, right):
        self.is_const = left.is_const and right.is_const
        if self.is_const:
            self.calc_binary_const(op, left, right)

        ltype, rtype = left.type.typespec, right.type.typespec

        if ltype == rtype == '$' and op == 'add':
            self._instrs += left.instrs
            self._instrs += right.instrs
            self._instrs += [Instr('syscall', '__concat')]
            self.type = Type('$')
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
        self._instrs += instrs


    def binary_logical_op(self, op, left, right, *,
                          inv_left=False, inv_right=False):
        self.is_const = left.is_const and right.is_const
        if self.is_const:
            self.calc_binary_const(op, left, right)

        ltype, rtype = left.type, right.type
        if any(not i.type.is_numeric for i in [left, right]):
            raise CompileError(EC.TYPE_MISMATCH)

        result_type = {
            frozenset({'!'}): '&',
            frozenset({'#'}): '&',
            frozenset({'%'}): '%',
            frozenset({'&'}): '&',
            frozenset({'!', '#'}): '&',
            frozenset({'!', '%'}): '&',
            frozenset({'!', '&'}): '&',
            frozenset({'#', '%'}): '&',
            frozenset({'#', '&'}): '&',
            frozenset({'%', '&'}): '&',
        }[frozenset({ltype.typespec, rtype.typespec})]
        result_type = Type(result_type)

        instrs = left.instrs
        if left.type != result_type:
            instrs += gen_conv_instrs(left.type, result_type)
        if inv_left:
            instrs += [Instr(f'not{result_type.typespec}')]

        instrs += right.instrs
        if right.type != result_type:
            instrs += gen_conv_instrs(right.type, result_type)
        if inv_right:
            instrs += [Instr(f'not{result_type.typespec}')]

        instrs += [Instr(f'{op}{result_type.typespec}')]

        self.type = result_type
        self._instrs += instrs


    def compare_op(self, op, left, right):
        self.binary_op('sub', left, right)
        self._instrs += gen_conv_instrs(self.type, Type('%'))
        self._instrs += [Instr(op)]
        self.type = Type('%')

        if self.is_const:
            self.const_value = {
                'eq': self.const_value == 0,
                'ne': self.const_value != 0,
                'lt': self.const_value < 0,
                'gt': self.const_value > 0,
                'le': self.const_value <= 0,
                'ge': self.const_value >= 0,
            }[op]


    @property
    def instrs(self):
        if self.parent.optimization > 0 and self.is_const:
            if self.type.is_numeric:
                return [Instr(f'pushi{self.type.typespec}', self.const_value)]
            else:
                return [Instr(f'pushi$', f'"{self.const_value}"'),
                        Instr('syscall', '__strcpy')]
        else:
            return self._instrs


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
            if tok.type == 'COMMENT_QUOTE':
                self.compiler.check_for_comment_directive(tok.value[1:])
                continue
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
    def __init__(self, optimization=0):
        with open('qpybasic.ebnf') as f:
            grammar_text = f.read()

        self.parser = Lark(grammar_text,
                           parser='lalr',
                           lexer='standard',
                           postlex=PostLex(self),
                           propagate_positions=True,
                           start='program')

        self.optimization = optimization


    def compile(self, code):
        code += '\n'

        self.instrs = []
        self.cur_routine = Routine('__main', 'sub', self)
        self.routines = {'__main': self.cur_routine}
        self.gen_labels = {}
        self.gen_vars = {}
        self.endif_labels = []
        self.string_literals = {}
        self.last_string_literal_idx = 0
        self.default_array_base = 1
        self.user_defined_types = {}
        self.default_types = {}
        self.exit_labels = []
        self.const_container = VarContainer(self)
        self.shared_container = SharedVarContainer(self)
        self.default_allocation = 'static'
        self.cur_blocks = []

        self.add_string_literal('')

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

        # this is where variable init instructions are to be inserted
        # later.
        init_loc = len(self.instrs)

        ast = self.parser.parse(code)
        self.compile_ast(ast)

        self.instrs[init_loc:init_loc] = self.cur_routine.init_instrs

        self.instrs += self.cur_routine.gen_free_instructions()
        self.instrs += [Instr('unframe', self.routines['__main']),
                        Instr('ret', 0)]
        self.instrs += sum((r.instrs for r in self.routines.values()), [])

        assembler = Assembler(self)
        self.bytecode = assembler.assemble(self.instrs)
        return Module.create(self.bytecode,
                             self.string_literals,
                             self.shared_container.vars)


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

        # due to some grammar conflicts, we're using an 'lvalue' as
        # the sub name (instead of the more sensible 'ID'). so here,
        # we check whether it is in fact actually an ID.
        if isinstance(sub_name, Tree):
            base, suffix = sub_name.children
            if len(suffix.children) > 0 or \
               len(base.children) > 1:
                raise CompileError(EC.INVALID_SUB_NAME)
            sub_name = base.children[0].value
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
               a.children[0].data == 'lvalue':
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


    def process_const_stmt(self, ast):
        _, name, value = ast.children
        value = Expr(value, self)
        if not value.is_const:
            raise CompileError(EC.INVALID_CONSTANT)
        var = self.dim_var(name, klass='const')
        if var.type.typespec in ['%', '&']:
            conv = int
        elif var.type.typespec in ['!', '#']:
            conv = float
        else:
            conv = str
        var.const_value = conv(value.const_value)


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
            if len(p.children) == 4:
                # form: var AS type
                pname, lpar_rpar, _, ptype = p.children
                pname = pname.value
                ptype = Type(ptype.children[0].value,
                             is_array=bool(lpar_rpar.children))
            else:
                # form: var$
                pname, lpar_rpar = p.children
                pname = pname.value
                ptype = self.get_type_from_var_name(pname)
                ptype.is_array = bool(lpar_rpar.children)

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
        _, is_shared, *decls = ast.children
        is_shared = bool(is_shared.children)
        for d in decls:
            self.process_dim_clause(d, is_shared)


    def process_dim_clause(self, clause, is_shared):
        klass = 'shared' if is_shared else 'local'
        if len(clause.children) == 1:
            name, = clause.children
            self.dim_var(name, klass=klass)
        else:
            if len(clause.children) == 2:
                name, dimensions = clause.children
                typename = None
            else:
                name, dimensions,  _, typename = clause.children

            if dimensions.children:
                dimensions = self.parse_dimensions(dimensions)
            else:
                dimensions = None

            if typename:
                typename = typename.children[0].value
                type = self.get_type(typename, is_array=bool(dimensions))
            else:
                type = None

            self.dim_var(name,
                         type=type,
                         klass=klass,
                         dimensions=dimensions)


    def parse_dimensions(self, ast):
        dimensions = []

        for d in ast.children:
            if len(d.children) == 1:
                d_from = self.get_constant_expr(self.default_array_base)
                d_to = Expr(d.children[0], self)
            else:
                d_from, _, d_to = d.children
                d_from = Expr(d_from, self)
                d_to = Expr(d_to, self)

            dimensions.append((d_from, d_to))

        return dimensions


    def get_constant_expr(self, value):
        if isinstance(value, (int, float)):
            tree = Tree('value', [Token('NUMERIC_LITERAL', str(value))])
        else:
            tree = Tree('value', [Token('STRING_LITERAL', f'"{value}"')])
        return Expr(tree, self)


    def process_end_stmt(self, ast):
        self.instrs += [Instr('end')]


    def process_exit_stmt(self, ast):
        _, target = ast.children
        target = target.value.lower()
        if target == 'sub':
            if self.cur_routine.type != 'sub' or \
               self.cur_routine.name == '__main':
                raise CompileError(EC.EXIT_SUB_INVALID)
            self.instrs += [Instr('jmp', self.cur_routine.exit_label)]
        elif target == 'function':
            if self.cur_routine.type != 'function':
                raise CompileError(EC.EXIT_FUNC_INVALID)
            self.instrs += [Instr('jmp', self.cur_routine.exit_label)]
        elif target == 'for':
            if self.exit_labels == []:
                raise CompileError(EC.EXIT_FOR_INVALID)

            type, label = self.exit_labels[-1]
            if type != 'for':
                raise CompileError(EC.EXIT_FOR_INVALID)

            self.instrs += [Instr('jmp', label)]
        elif target == 'do':
            if self.exit_labels == []:
                raise CompileError(EC.EXIT_DO_INVALID)

            type, label = self.exit_labels[-1]
            if type != 'do':
                raise CompileError(EC.EXIT_DO_INVALID)

            self.instrs += [Instr('jmp', label)]
        else:
            assert False


    def process_for_stmt(self, ast):
        _, var, start, _, end, step = ast.children
        var = self.get_var(var)
        if var.type.typespec == '$':
            raise CompileError(EC.INVALID_FOR_VAR)

        step_var = self.get_var(self.gen_var('for_step', var.type.typespec))
        if step.children:
            e = Expr(step.children[1], self)
            self.gen_set_var_code(step_var, e)
        else:
            self.instrs += [Instr(f'pushi{var.type.typespec}', 1)]
            self.instrs += Lvalue(step_var).gen_write_instructions()

        end_var = self.get_var(self.gen_var('for_end', var.type.typespec))
        e = Expr(end, self)
        self.gen_set_var_code(end_var, e)

        e = Expr(start, self)
        self.gen_set_var_code(var, e)

        top_label = self.gen_label('for_top')
        end_label = self.gen_label('for_bottom')
        self.exit_labels.append(('for', end_label))
        self.instrs += [Label(top_label),
                        Instr(f'readf{var.size}', var),
                        Instr(f'readf{end_var.size}', end_var),
                        Instr(f'sub{var.type.typespec}')]
        self.instrs += gen_conv_instrs(var.type, Type('%'))
        self.instrs += [Instr('gt'),
                        Instr('jmpt', end_label)]

        block_data = {
            'type': 'for',
            'var': var,
            'step_var': step_var,
            'top_label': top_label,
            'end_label': end_label,
        }
        self.enter_block(block_data)


    def process_next_stmt(self, ast):
        block_data = self.exit_block('for')
        var = block_data['var']
        step_var = block_data['step_var']
        top_label = block_data['top_label']
        end_label = block_data['end_label']

        if len(ast.children) == 2:
            # "NEXT var" is used. Check if NEXT variable matches FOR
            # variable.
            next_var = self.get_var(ast.children[1])
            if next_var != var:
                raise CompileError(EC.NEXT_VAR_NOT_MATCH_FOR)

        self.instrs += [Instr(f'readf{var.size}', var),
                        Instr(f'readf{step_var.size}', step_var),
                        Instr(f'add{var.type.typespec}')]
        self.instrs += Lvalue(var).gen_write_instructions()
        self.instrs += [Instr('jmp', top_label),
                        Label(end_label)]

        self.exit_labels.pop()


    def process_sub_block(self, ast):
        _, name, params, body, _, _ = ast.children

        if name[-1] in typespec_chars:
            raise CompileError(EC.INVALID_SUB_NAME)

        saved_instrs = self.instrs
        self.instrs = []
        self.cur_routine = Routine(name.value, 'sub', self)

        self.instrs += [Label(f'__sub_{name}'),
                        Instr('frame', self.cur_routine)]

        # this is where variable init instructions are to be inserted
        # later.
        init_loc = len(self.instrs)

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

        self.instrs[init_loc:init_loc] = self.cur_routine.init_instrs

        arg_size = sum(v.size for v in self.cur_routine.params)
        self.instrs += [Label(self.cur_routine.exit_label)]
        self.instrs += self.cur_routine.gen_free_instructions()
        self.instrs += [Instr('unframe', self.cur_routine),
                        Instr('ret', arg_size)]
        self.cur_routine.instrs = self.instrs

        self.cur_routine = self.routines['__main']
        self.instrs = saved_instrs


    def parse_param_def(self, p):
        if len(p.children) == 4:
            # form: var AS type
            pname, is_array, _, ptype = p.children
            pname = pname.value
            ptype = self.get_type(ptype.children[0].value,
                                  is_array=bool(is_array.children))
        else:
            # form: var$
            pname, is_array = p.children
            pname = pname.value
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
        self.cur_routine = Routine(name, 'function', self)
        self.cur_routine.ret_type = ftype

        self.instrs += [Label(f'__function_{name}'),
                        Instr('frame', self.cur_routine)]

        # this is where variable init instructions are to be inserted
        # later.
        init_loc = len(self.instrs)

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
        self.instrs += Lvalue(ret_var).gen_write_instructions()

        if name in self.declared_routines:
            defined_param_types = [v.type for v in self.cur_routine.params]
            if defined_param_types != self.declared_routines[name].param_types or \
               ftype != self.declared_routines[name].ret_type:
                raise CompileError(EC.FUNC_DEF_NOT_MATCH_DECL)

        self.routines[name] = self.cur_routine
        self.compile_ast(body)

        self.instrs[init_loc:init_loc] = self.cur_routine.init_instrs

        arg_size = sum(v.size for v in self.cur_routine.params)
        self.instrs += [Label(self.cur_routine.exit_label),
                        Instr('unframe_r', self.cur_routine, ret_var),
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
            if len(e.children) == 2:
                e_name, is_array = e.children
                e_type = self.get_default_type(e_name)
            else:
                e_name, is_array, _, e_type = e.children
                e_type = self.get_type(e_type.children[0])

            if e_type == Type('$'):
                raise CompileError(EC.STRING_NOT_ALLOWED_IN_TYPE)

            if is_array.children:
                raise CompileError(EC.INVALID_TYPE_ELEMENT)

            e_name = e_name.value
            elements.append((e_name, e_type))

        self.user_defined_types[type_name] = elements


    def process_gosub_stmt(self, ast):
        _, target = ast.children
        if target.type == 'ID':
            self.instrs += [Instr('call', target)]
        else:
            self.instrs += [Instr('call', f'__lineno_{target.value}')]


    def process_goto_stmt(self, ast):
        _, target = ast.children
        if target.type == 'ID':
            self.instrs += [Instr('jmp', target)]
        else:
            self.instrs += [Instr('jmp', f'__lineno_{target.value}')]


    def process_if_block_begin(self, ast):
        _, cond, _ = ast.children

        cond = Expr(cond, self)
        self.instrs += cond.instrs

        # This instruction will be updated upon encountering an ELSE,
        # ELSEIF or END IF.
        jmpf_instr = Instr('jmpf', None)
        self.instrs += [jmpf_instr]

        block_data = {
            'type': 'if',
            'jmpf_instr': jmpf_instr,
            'endif_label': self.gen_label('endif'),
            'else_encountered': False,
        }
        self.enter_block(block_data)


    def process_else_stmt(self, ast):
        block_data = self.exit_block('if')
        endif_label = block_data['endif_label']

        label = self.gen_label('else')
        block_data['jmpf_instr'].operands = [label]

        if block_data['else_encountered']:
            raise CompileError(EC.SYNTAX_ERROR,
                               'Multiple else statements')

        self.instrs += [Instr('jmp', endif_label),
                        Label(label)]

        block_data['else_encountered'] = True
        self.enter_block(block_data)


    def process_elseif_stmt(self, ast):
        _, cond, _ = ast.children

        label = self.gen_label('elseif')

        block_data = self.exit_block('if')
        endif_label = block_data['endif_label']
        block_data['jmpf_instr'].operands = [label]

        if block_data['else_encountered']:
            raise CompileError(EC.SYNTAX_ERROR, 'ELSEIF after ELSE')

        self.instrs += [Instr('jmp', endif_label),
                        Label(label)]

        cond = Expr(cond, self)
        self.instrs += cond.instrs

        # This instruction will be updated upon encountering an ELSE,
        # ELSEIF or END IF.
        jmpf_instr = Instr('jmpf', None)
        self.instrs += [jmpf_instr]

        block_data['jmpf_instr'] = jmpf_instr
        self.enter_block(block_data)


    def process_end_block_stmt(self, ast):
        _, block_type = ast.children

        block_type = block_type.lower()
        assert block_type in ['if']

        block_data = self.exit_block(block_type)

        if block_type == 'if':
            label = block_data['endif_label']
            if not block_data['else_encountered']:
                block_data['jmpf_instr'].operands = [label]
            self.instrs += [Label(label)]


    def process_if_stmt(self, ast):
        if len(ast.children) == 3:
            self.process_if_block_begin(ast)
            return

        _, cond, _, *then_stmts, else_clause = ast.children
        if else_clause.children:
            _, *else_stmts = else_clause.children
        else:
            else_stmts = []

        else_label = self.gen_label('else')
        end_label = self.gen_label('endif')
        not_true_label = else_label if else_stmts else end_label

        cond = Expr(cond, self)
        self.instrs += cond.instrs
        self.instrs += [Instr('jmpf', not_true_label)]
        for stmt in then_stmts:
            self.compile_ast(stmt)
        if else_stmts:
            self.instrs += [Instr('jmp', end_label)]
            self.instrs += [Label(else_label)]
            for stmt in else_stmts:
                self.compile_ast(stmt)
        self.instrs += [Label(end_label)]


    def process_input_stmt(self, ast):
        if isinstance(ast.children[1], Token) and ast.children[1] == ';':
            _, _, prompt_phrase, *lvalues = ast.children
            same_line = True
        else:
            _, prompt_phrase, *lvalues = ast.children
            same_line = False

        if prompt_phrase.children:
            prompt, sep = prompt_phrase.children
            prompt = prompt.value[1:-1]
            sep = sep.value
        else:
            prompt = ''
            sep = ';'

        self.add_string_literal(prompt)

        lvalues = [self.create_lvalue_if_possible(lv) for lv in lvalues]
        if any(lv == None for lv in lvalues):
            raise CompileError(EC.INVALID_LVALUE)

        if any(not lv.type.is_basic or lv.type.is_array for lv in lvalues):
            raise CompileError(EC.TYPE_MISMATCH)

        type_values = {
            'INTEGER': 1,
            'LONG': 2,
            'SINGLE': 3,
            'DOUBLE': 4,
            'STRING': 5,
        }

        for lv in lvalues:
            self.instrs += lv.gen_ref_instructions()
            self.instrs += [Instr('pushi%', type_values[lv.type.name])]

        self.instrs += [Instr('pushi%', len(lvalues)),
                        Instr('pushi$', f'"{prompt}"'),
                        Instr('pushi%', 0 if sep == ';' else 1),
                        Instr('pushi%', 0 if not same_line else 1)]

        self.instrs += [Instr('syscall', '__input')]


    def process_do_stmt(self, ast):
        if len(ast.children) == 1:
            cond = None
        else:
            _, cond_type, cond = ast.children

            cond_type = cond_type.lower()
            cond = Expr(cond, self)

        top_label = self.gen_label('loop_top')
        bottom_label = self.gen_label('loop_bottom')
        self.exit_labels.append(('do', bottom_label))

        self.instrs += [Label(top_label)]
        if cond:
            self.instrs += cond.instrs
            if cond_type == 'until':
                self.instrs += [Instr('not%')]
            self.instrs += [Instr('jmpf', bottom_label)]

        block_data = {
            'type': 'do',
            'top_label': top_label,
            'bottom_label': bottom_label,
        }
        self.enter_block(block_data)


    def process_loop_stmt(self, ast):
        block_data = self.exit_block('do')
        top_label = block_data['top_label']
        bottom_label = block_data['bottom_label']

        if len(ast.children) == 1:
            cond = None
        else:
            _, cond_type, cond = ast.children

            cond_type = cond_type.lower()
            cond = Expr(cond, self)

        if cond:
            self.instrs += cond.instrs
            if cond_type == 'until':
                self.instrs += [Instr('not%')]
            self.instrs += [Instr('jmpt', top_label)]
        else:
            self.instrs += [Instr('jmp', top_label)]

        self.instrs += [Label(bottom_label)]
        self.exit_labels.pop()


    def process_let_stmt(self, ast):
        if len(ast.children) == 3:
            # cut the LET keyword
            ast.children = ast.children[1:]

        dest, expr = ast.children
        lv = self.create_lvalue_if_possible(dest)
        if lv == None:
            # check for the case of assigning to the function name in
            # functions (which won't be detected by
            # `create_lvalue_if_possible` because the name is a
            # function).
            base, suffix = dest.children
            if len(base.children) == 1 and \
               len(suffix.children) == 0:
                name = base.children[0].value
                var = self.get_var(name)
                lv = Lvalue(var)

        if lv == None:
            raise CompileError(EC.INVALID_LVALUE)

        expr = Expr(expr, self)
        self.instrs += expr.instrs
        if expr.type != lv.type:
            conv_instrs = gen_conv_instrs(expr.type, lv.type)
            if conv_instrs == None:
                raise CompileError(EC.TYPE_MISMATCH)
            self.instrs += conv_instrs
        if lv.type == Type('$'):
            self.instrs += lv.gen_ref_instructions()
            self.instrs += [Instr('readi4'),
                            Instr('syscall', '__free')]
        self.instrs += lv.gen_write_instructions()


    def gen_set_var_code(self, var, expr):
        conv_instrs = gen_conv_instrs(expr.type, var.type)
        if conv_instrs == None:
            raise CompileError(
                EC.CANNOT_CONVERT_TYPE,
                f'Cannot convert {expr.type.name} to {var.type.name}.')
        else:
            self.instrs += expr.instrs + conv_instrs

            lv = Lvalue(var)
            if lv.type == Type('$'):
                self.instrs += lv.gen_ref_instructions()
                self.instrs += [Instr('readi4'),
                                Instr('syscall', '__free')]
            self.instrs += lv.gen_write_instructions()


    def create_lvalue_if_possible(self, ast):
        # this function receive a tree for the 'lvalue' rule, and
        # attempts to create an lvalue from it. if not possible, None
        # is returned.

        empty_parentheses = False
        indices = None
        base, suffix = ast.children

        if len(suffix.children) == 0 and \
           self.is_function(base.children[0]):
            return None
        else:
            var = self.get_var(base.children[0].value)
            if len(base.children) == 4:
                _, _, indices, _ = base.children
                indices = [Expr(i, self) for i in indices.children]

                if len(indices) == 0:
                    # an array is being sent, like in 'sub_name arr(), arg2'
                    if not var.type.is_array:
                        raise CompileError(EC.TYPE_MISMATCH)
                    empty_parentheses = True
            dots = [i.value for i in suffix.children]

            if not indices and not dots and var.type.is_array and not empty_parentheses:
                raise CompileError(EC.TYPE_MISMATCH)

            return Lvalue(var, indices=indices, dots=dots, compiler=self)


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


    def process_rem_stmt(self, ast):
        self.check_for_comment_directive(ast.children[0].value[4:])


    def process_return_stmt(self, ast):
        self.instrs += [Instr('ret', 0)]


    def process_view_print_stmt(self, ast):
        if len(ast.children) == 2:
            self.instrs += [Instr('pushi%', -1),
                            Instr('pushi%', -1)]
        else:
            _, _, top_line, _, bottom_line = ast.children
            top_line = Expr(top_line, self)
            bottom_line = Expr(bottom_line, self)
            self.instrs += top_line.instrs
            self.instrs += bottom_line.instrs

        self.instrs += [Instr('syscall', '__view_print')]


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
        assert klass in ['local', 'param', 'const', 'shared']
        assert klass == 'param' or not byref

        no_type = (type == None)

        if used_name[-1] in typespec_chars:
            name = used_name[:-1]
            typespec = used_name[-1]

            if type:
                raise CompileError(EC.INVALID_VAR_NAME)

            type = Type(typespec, is_array=bool(dimensions))
        else:
            name = used_name
            typespec = None
            if not type:
                type = self.get_default_type(used_name)
                type.is_array = bool(dimensions)


        containers = [self.cur_routine,
                      self.const_container,
                      self.shared_container]
        for c in containers:
            var = c.lookup_var(name, typespec)
            if var:
                if var.no_type_dimmed and not typespec:
                    raise CompileError(EC.AS_CLAUSE_REQUIRED_ON_FIRST_DECL)
                else:
                    raise CompileError(EC.DUP_DEF)

            # if there has been a variable declared before this, with
            # a full 'dim v as type' statement, it's still a duplicate
            # even if the typespec char doesn't match.
            var = c.lookup_var(name, None)
            if var and not var.no_type_dimmed:
                raise CompileError(EC.DUP_DEF)

            # if there has been any type (fully typed or not,
            # including implicit) of declaration with this name
            # before, and this is a fully typed dim
            same_named = c.lookup_var(name, None, get_all=True)
            if not no_type and same_named:
                if same_named[0].type == type:
                    raise CompileError(EC.AS_CLAUSE_REQUIRED_ON_FIRST_DECL)
                else:
                    raise CompileError(EC.DUP_DEF)

        container = {
            'local': self.cur_routine,
            'param': self.cur_routine,
            'const': self.const_container,
            'shared': self.shared_container,
        }[klass]

        var = Var()
        var.name = name
        var.type = type
        var.container = container
        var.no_type_dimmed = no_type
        var.klass = klass
        var.byref = byref
        var.on_heap = False

        container.add_var(var)

        init_instrs, init_loc = self.gen_init_var(var, dimensions)
        if init_loc == 'top':
            self.cur_routine.init_instrs += init_instrs
        else:
            self.instrs += init_instrs

        if var.klass == 'param':
            pidx = var.container.params.index(var)
            var.idx = 8 + sum(v.size for v in var.container.params[:pidx])
        elif var.klass == 'local':
            lidx = var.container.local_vars.index(var)
            var.idx = -sum(v.size for v in var.container.local_vars[:lidx+1])

        return var


    def gen_init_var(self, var, dimensions):
        # Normally, variables should be initialized at the beginning
        # of the current sub/function, so we set this to 'top' for
        # now. In case of dynamic arrays, which depend on the current
        # value of variables at the location where variables are
        # declared, 'curloc' is returned.
        init_loc = 'top'

        instrs = []
        if var.klass in ['local', 'shared']:
            lv = Lvalue(var)

            if var.type.is_array and dimensions:
                for d_from, d_to in dimensions:
                    if not d_from.type.is_numeric or \
                       not d_to.type.is_numeric:
                        raise CompileError(EC.TYPE_MISMATCH)
                if self.default_allocation != 'dynamic' and \
                   all(d_from.is_const and d_to.is_const
                       for d_from, d_to in dimensions):
                    instrs += self.gen_static_array_init(var, lv, dimensions)
                else:
                    instrs += self.gen_dynamic_array_init(var, lv, dimensions)
                    init_loc = 'cur'
            elif var.type.is_basic:
                if var.type.name == 'STRING':
                    instrs = [Instr(f'pushi$', '""'),
                              Instr('syscall', '__strcpy')]
                else:
                    instrs = [Instr(f'pushi{var.type.typespec}', 0)]
                instrs += lv.gen_write_instructions()
            else:
                instrs = [Instr('pushi%', 0),
                          Instr('pushi_ul', var.type.get_size())]
                instrs += lv.gen_ref_instructions()
                instrs += [Instr('syscall', '__memset')]

        return instrs, init_loc


    def gen_static_array_init(self, var, lv, dimensions):
        var.on_heap = False
        var.type.dynamic = False
        var.type.dims = []

        instrs = []
        for d_from, d_to in dimensions:
            if d_from.const_value > d_to.const_value:
                raise CompileError(EC.INVALID_ARRAY_BOUNDS)
            var.type.dims.append((int(d_from.const_value), int(d_to.const_value)))
            instrs += d_to.instrs
            if d_to.type.typespec != '%':
                instrs += [Instr(f'conv{d_to.type.typespec}%')]
            instrs += d_from.instrs
            if d_from.type.typespec != '%':
                instrs += [Instr(f'conv{d_from.type.typespec}%')]
        element_size = Type(var.type.get_base_type_name()).get_size()
        instrs += [Instr('pushi%', len(dimensions)),
                   Instr('pushi%', element_size)]
        instrs += lv.gen_ref_instructions()

        if var.type.get_base_type_name() == 'STRING':
            instrs += [Instr('syscall', '__init_str_array')]
        else:
            instrs += [Instr('syscall', '__init_array')]

        return instrs


    def gen_dynamic_array_init(self, var, lv, dimensions):
        var.type.dynamic = True

        instrs = [Instr('pushi_ul', 1)]
        for d_from, d_to in dimensions:
            instrs += d_to.instrs
            if d_to.type.typespec != '%':
                instrs += [Instr(f'conv{d_to.type.typespec}%')]
            instrs += d_from.instrs
            if d_from.type.typespec != '%':
                instrs += [Instr(f'conv{d_from.type.typespec}%')]
            instrs += [Instr('sub%'),
                       Instr('pushi%', 1),
                       Instr('add%'),
                       Instr('conv%_ul'),
                       Instr('mul_ul')]
        element_size = Type(var.type.get_base_type_name()).get_size()
        instrs += [Instr('pushi_ul', element_size),
                   Instr('mul_ul'),
                   Instr('syscall', '__malloc')]
        instrs += lv.gen_write_instructions()

        var.byref = True
        var.on_heap = True

        for d_from, d_to in dimensions:
            instrs += d_to.instrs
            if d_to.type.typespec != '%':
                instrs += [Instr(f'conv{d_to.type.typespec}%')]
            instrs += d_from.instrs
            if d_from.type.typespec != '%':
                instrs += [Instr(f'conv{d_from.type.typespec}%')]
        instrs += [Instr('pushi%', len(dimensions)),
                   Instr('pushi%', element_size)]
        instrs += lv.gen_ref_instructions()

        instrs += [Instr('syscall', '__init_array')]

        return instrs


    def get_var(self, used_name):
        if used_name[-1] in typespec_chars:
            name = used_name[:-1]
            typespec = used_name[-1]
        else:
            name = used_name
            typespec = None

        containers = [self.cur_routine,
                      self.const_container,
                      self.shared_container]
        for c in containers:
            var = c.lookup_var(name, typespec)
            if var:
                break
        else:
            var = self.dim_var(used_name)

        return var


    def get_type(self, name, *, is_array=False):
        name = name.upper()
        if name in Type.basic_types:
            return Type(name, is_array=is_array)

        elements = self.user_defined_types.get(name, None)
        if not elements:
            raise CompileError(EC.UNDEFINED_TYPE)

        return Type(name, elements=elements, is_array=is_array)


    def get_type_from_var_name(self, var_name):
        if var_name[-1] in typespec_chars:
            return Type(var_name[-1])
        else:
            return self.get_default_type(var_name)


    def get_default_type(self, var_name):
        t = self.default_types.get(var_name[0], None)
        t = t if t else Type('SINGLE')
        return deepcopy(t)


    def check_for_comment_directive(self, comment):
        comment = comment.strip().lower()
        if comment == '$dynamic':
            self.default_allocation = 'dynamic'
        elif comment == '$static':
            self.default_allocation = 'static'


    def enter_block(self, block_data):
        self.cur_blocks.append(block_data)


    def exit_block(self, block_type):
        block_data = self.cur_blocks.pop()
        if block_data['type'] != block_type:
            raise CompileError(EC.BLOCK_END_MISMATCH, msg)

        return block_data


class Instr:
    """An instance of this class, is a representation of an instruction in
use, i.e. the instruction itself, plus the values of its arguments.

    """

    def __init__(self, name, *args):
        self.name = name
        self.operands = args
        self.abstract_instruction = asm.Instruction.from_name(name)


    @property
    def size(self):
        return self.abstract_instruction.size


    @property
    def opcode(self):
        return self.abstract_instruction.opcode


    def assemble(self, resolver):
        fmt = ''.join(o.bin_fmt
                      for o in self.abstract_instruction.operands)
        operands = resolver(self)
        assert all(isinstance(o, (int, float, str)) for o in operands)
        code = bytes([self.opcode]) + struct.pack('>' + fmt, *operands)
        assert len(code) == self.abstract_instruction.size
        return code


    def __repr__(self):
        return f'<Instr {self.name} {self.operands}>'


    def __str__(self):
        args = ', '.join(str(i) for i in self.operands)
        if len(self.name) >= 8:
            return f'\t{self.name}\t{args}'
        else:
            return f'\t{self.name}\t\t{args}'


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
            elif instr.name == 'pushi_ul':
                val = instr.operands[0]
                if isinstance(val, Var) and val.klass == 'shared':
                    val = Module.SHARED_ADDR + self.compiler.shared_container.get_var_offset(val)
                operands = [val]
            elif instr.name == 'syscall':
                call_code = {
                    '__cls': 0x02,
                    '__concat': 0x03,
                    '__print': 0x04,
                    '__init_array': 0x05,
                    '__memset': 0x06,
                    '__access_array': 0x07,
                    '__malloc': 0x08,
                    '__free': 0x09,
                    '__view_print': 0x0a,
                    '__strcpy': 0x0b,
                    '__init_str_array': 0x0c,
                    '__free_strings_in_array': 0x0d,
                    '__input': 0x0e,
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
