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
        return f'\t{self.name}\t{args}'


class Instruction:
    """An instance of this class represents an instruction in the
abstract, e.g. the 'add%' instruction in general and not a particular
usage of it.

    """
    def __init__(self, opcode, name, operands):
        assert 0 < opcode < 255
        assert isinstance(name, str)
        assert all(isinstance(i, Operand) for i in operands)

        self.opcode = opcode
        self.name = name
        self.operands = operands


    @property
    def size(self):
        return 1 + sum(i.size for i in self.operands)


    @staticmethod
    def from_name( name):
        return instr_name_to_instr[name]


    @staticmethod
    def from_opcode(opcode):
        return opcode_to_instr[opcode]


def def_instr(opcode, name, *operands):
    instr = Instruction(opcode, name, operands)
    opcode_to_instr[opcode] = instr
    instr_name_to_instr[name] = instr


def_instr(0x01, 'add%')
def_instr(0x02, 'add&')
def_instr(0x03, 'add!')
def_instr(0x04, 'add#')
def_instr(0x05, 'call', LOC_ABSOLUTE)
def_instr(0x06, 'conv%&')
def_instr(0x07, 'conv%!')
def_instr(0x08, 'conv%#')
def_instr(0x09, 'conv&%')
def_instr(0x0a, 'conv&!')
def_instr(0x0b, 'conv&#')
def_instr(0x0c, 'conv!%')
def_instr(0x0d, 'conv!&')
def_instr(0x0e, 'conv!#')
def_instr(0x0f, 'conv#%')
def_instr(0x10, 'conv#&')
def_instr(0x12, 'conv#!')
def_instr(0x13, 'end')
def_instr(0x14, 'frame', SIZE)
def_instr(0x15, 'eq')
def_instr(0x16, 'ge')
def_instr(0x17, 'gt')
def_instr(0x18, 'jmp', LOC_ABSOLUTE)
def_instr(0x19, 'jmpf', LOC_OFFSET)
def_instr(0x1a, 'jmpt', LOC_OFFSET)
def_instr(0x1b, 'le')
def_instr(0x1c, 'lt')
def_instr(0x1d, 'mul%')
def_instr(0x1e, 'mul&')
def_instr(0x1f, 'mul!')
def_instr(0x20, 'mul#')
def_instr(0x21, 'ne')
def_instr(0x22, 'neg%')
def_instr(0x23, 'neg&')
def_instr(0x24, 'neg!')
def_instr(0x25, 'neg#')
def_instr(0x26, 'pushi%', IMM_INT)
def_instr(0x27, 'pushi&', IMM_LONG)
def_instr(0x28, 'pushi!', IMM_SINGLE)
def_instr(0x29, 'pushi#', IMM_DOUBLE)
def_instr(0x2a, 'pushi$', LOC_ABSOLUTE)
def_instr(0x2b, 'readf1', FIDX)
def_instr(0x2c, 'readf2', FIDX)
def_instr(0x2d, 'readf4', FIDX)
def_instr(0x2e, 'sub%')
def_instr(0x2f, 'sub&')
def_instr(0x30, 'sub!')
def_instr(0x31, 'sub#')
def_instr(0x32, 'syscall', IMM_INT)
def_instr(0x33, 'writef1', FIDX)
def_instr(0x34, 'writef2', FIDX)
def_instr(0x35, 'writef4', FIDX)
def_instr(0x36, 'writef8', FIDX)
def_instr(0x37, 'ret', SIZE)
def_instr(0x38, 'unframe', SIZE)
def_instr(0x39, 'readf8', FIDX)
def_instr(0x3a, 'readi1')
def_instr(0x3b, 'readi2')
def_instr(0x3c, 'readi4')
def_instr(0x3d, 'readi8')
def_instr(0x3e, 'writei1')
def_instr(0x3f, 'writei2')
def_instr(0x40, 'writei4')
def_instr(0x41, 'writei8')
def_instr(0x42, 'pushfp', FIDX)
def_instr(0x43, 'dup1')
def_instr(0x44, 'dup2')
def_instr(0x45, 'dup4')
def_instr(0x46, 'sgn%')
def_instr(0x47, 'sgn&')
def_instr(0x48, 'sgn!')
def_instr(0x49, 'sgn#')
def_instr(0x4a, 'unframe_r', SIZE, FIDX, SIZE)
def_instr(0x4b, 'ret_r', SIZE, SIZE)
