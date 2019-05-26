import struct
from functools import partial

opcode_to_instr = {}
instr_name_to_instr = {}


class Operand:
    def __init__(self, bin_fmt, text_fmt):
        self.bin_fmt = bin_fmt
        self.text_fmt = text_fmt

        self.size = {
            'h': 2,
            'H': 2,
            'i': 4,
            'I': 4,
            'f': 4,
            'd': 8,
        }[bin_fmt]

        self.__format_funcs = {
            'hex2': partial(self.__format_hex, 2),
            'hex4': partial(self.__format_hex, 4),
            'decimal': self.__format_decimal,
            'float': self.__format_float,
        }


    def encode(self, value):
        return struct.pack('>' + self.bin_fmt, value)


    def format(self, value):
        if not isinstance(value, (int, float, str)):
            value = value.get_operand_value()


    def __format_hex(self, size, value):
        assert isinstance(value, int)

        size = size * 2 + 2 # two digits per byte plus two more for "0x"
        return f'{value:#0{size}x}'


    def __format_decimal(self, value):
        assert isinstance(value, int)
        return f'{value}'


    def __format_float(self, value):
        assert isinstance(value, float)
        return f'{value}'



# Different kinds of operands. Each operand is defined by a binary
# representation (which is a format character as defined by Python's
# struct module), and a text format, which tells us how the value
# should be printed out in disassembly.


# An signed, 16-bit index relative to FP.
FIDX = Operand('h', 'decimal')

# An unsigned, 16-bit size
SIZE = Operand('H', 'decimal')

# A signed offset relative to IP, representing a relative location in
# memory.
LOC_OFFSET = Operand('h', 'decimal')


# An unsigned 32-bit integer, representing an absolute location in
# memory.
LOC_ABSOLUTE = Operand('I', 'hex4')

# A signed, 16-bit integer immediate value.
IMM_INT = Operand('h', 'decimal')

# A signed, 32-bit long immediate value.
IMM_LONG = Operand('i', 'decimal')

# A single-precision (32-bit) floating point immediate value.
IMM_SINGLE = Operand('f', 'float')

# A double-precision (64-bit) floating point immediate value.
IMM_DOUBLE = Operand('d', 'float')

# A unsigned, 32-bit long immediate value.
IMM_UL = Operand('i', 'hex4')


class Instr:
    """An instance of this class, is a representation of an instruction in
use, i.e. the instruction itself, plus the values of its arguments.

    """

    def __init__(self, name, *args):
        self.name = name
        self.operands = args
        self.abstract_instruction = Instruction.from_name(name)


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


class Instruction:
    """An instance of this class represents an instruction in the
abstract, e.g. the 'add%' instruction in general and not a particular
usage of it.

    """
    def __init__(self, opcode, name, operands, *, stack=None):
        assert 0 < opcode < 255
        assert isinstance(name, str)
        assert all(isinstance(i, Operand) for i in operands)

        self.opcode = opcode
        self.name = name
        self.operands = operands
        self.stack = stack


    @property
    def size(self):
        return 1 + sum(i.size for i in self.operands)


    @staticmethod
    def from_name( name):
        return instr_name_to_instr[name]


    @staticmethod
    def from_opcode(opcode):
        return opcode_to_instr[opcode]


def def_instr(opcode, name, *operands, stack=None):
    instr = Instruction(opcode, name, operands, stack=stack)
    opcode_to_instr[opcode] = instr
    instr_name_to_instr[name] = instr


def_instr(0x01, 'add%', stack=-2)
def_instr(0x02, 'add&', stack=-4)
def_instr(0x03, 'add!', stack=-4)
def_instr(0x04, 'add#', stack=-8)
def_instr(0x05, 'call', LOC_ABSOLUTE, stack=4)
def_instr(0x06, 'conv%&', stack=2)
def_instr(0x07, 'conv%!', stack=2)
def_instr(0x08, 'conv%#', stack=6)
def_instr(0x09, 'conv&%', stack=-2)
def_instr(0x0a, 'conv&!', stack=0)
def_instr(0x0b, 'conv&#', stack=4)
def_instr(0x0c, 'conv!%', stack=-2)
def_instr(0x0d, 'conv!&', stack=0)
def_instr(0x0e, 'conv!#', stack=4)
def_instr(0x0f, 'conv#%', stack=-6)
def_instr(0x10, 'conv#&', stack=-4)
def_instr(0x12, 'conv#!', stack=-4)
def_instr(0x13, 'end', stack=0)
def_instr(0x14, 'frame', SIZE, stack='4 + op1')
def_instr(0x15, 'eq', stack=0)
def_instr(0x16, 'ge', stack=0)
def_instr(0x17, 'gt', stack=0)
def_instr(0x18, 'jmp', LOC_ABSOLUTE, stack=0)
def_instr(0x19, 'jmpf', LOC_OFFSET, stack=-2)
def_instr(0x1a, 'jmpt', LOC_OFFSET, stack=-2)
def_instr(0x1b, 'le', stack=0)
def_instr(0x1c, 'lt', stack=0)
def_instr(0x1d, 'mul%', stack=-2)
def_instr(0x1e, 'mul&', stack=-4)
def_instr(0x1f, 'mul!', stack=-4)
def_instr(0x20, 'mul#', stack=-8)
def_instr(0x21, 'ne', stack=0)
def_instr(0x22, 'neg%', stack=0)
def_instr(0x23, 'neg&', stack=0)
def_instr(0x24, 'neg!', stack=0)
def_instr(0x25, 'neg#', stack=0)
def_instr(0x26, 'pushi%', IMM_INT, stack=2)
def_instr(0x27, 'pushi&', IMM_LONG, stack=4)
def_instr(0x28, 'pushi!', IMM_SINGLE, stack=4)
def_instr(0x29, 'pushi#', IMM_DOUBLE, stack=8)
def_instr(0x2a, 'pushi$', LOC_ABSOLUTE, stack=4)
def_instr(0x2b, 'readf1', FIDX, stack=1)
def_instr(0x2c, 'readf2', FIDX, stack=2)
def_instr(0x2d, 'readf4', FIDX, stack=4)
def_instr(0x2e, 'sub%', stack=-2)
def_instr(0x2f, 'sub&', stack=-4)
def_instr(0x30, 'sub!', stack=-4)
def_instr(0x31, 'sub#', stack=-8)
def_instr(0x32, 'syscall', IMM_INT)
def_instr(0x33, 'writef1', FIDX, stack=-1)
def_instr(0x34, 'writef2', FIDX, stack=-2)
def_instr(0x35, 'writef4', FIDX, stack=-4)
def_instr(0x36, 'writef8', FIDX, stack=-8)
def_instr(0x37, 'ret', SIZE, stack='-4 - op1')
def_instr(0x38, 'unframe', SIZE)
def_instr(0x39, 'readf8', FIDX, stack=8)
def_instr(0x3a, 'readi1', stack=-3)
def_instr(0x3b, 'readi2', stack=-2)
def_instr(0x3c, 'readi4', stack=0)
def_instr(0x3d, 'readi8', stack=4)
def_instr(0x3e, 'writei1', stack=-5)
def_instr(0x3f, 'writei2', stack=-6)
def_instr(0x40, 'writei4', stack=-8)
def_instr(0x41, 'writei8', stack=-12)
def_instr(0x42, 'pushfp', FIDX, stack=4)
def_instr(0x43, 'dup1', stack=1)
def_instr(0x44, 'dup2', stack=2)
def_instr(0x45, 'dup4', stack=4)
def_instr(0x46, 'sgn%', stack=0)
def_instr(0x47, 'sgn&', stack=-2)
def_instr(0x48, 'sgn!', stack=-2)
def_instr(0x49, 'sgn#', stack=-6)
def_instr(0x4a, 'ret_r', SIZE, SIZE, stack='-4 - op1')
def_instr(0x4b, 'unframe_r', SIZE, FIDX, SIZE)
def_instr(0x4c, 'pushi_ul', IMM_UL, stack=4)
def_instr(0x4d, 'add_ul', stack=-4)
def_instr(0x4e, 'sub_ul', stack=-4)
def_instr(0x4f, 'mul_ul', stack=-4)
def_instr(0x50, 'div_ul', stack=-4)
def_instr(0x51, 'conv%_ul', stack=2)
def_instr(0x52, 'conv&_ul', stack=0)
def_instr(0x53, 'conv!_ul', stack=0)
def_instr(0x54, 'conv#_ul', stack=-4)
def_instr(0x55, 'conv_ul%', stack=-2)
def_instr(0x56, 'conv_ul&', stack=0)
def_instr(0x57, 'conv_ul!', stack=0)
def_instr(0x58, 'conv_ul#', stack=4)
