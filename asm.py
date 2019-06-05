#!/usr/bin/env python3

import struct
import argparse
from functools import partial
from module import Module, Sections

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


    @property
    def bin_fmt(self):
        "The binary format of the operands of this instruction."
        return ''.join(o.bin_fmt for o in self.operands)


    @staticmethod
    def from_name(name):
        return instr_name_to_instr[name]


    @staticmethod
    def from_opcode(opcode):
        return opcode_to_instr[opcode]


def main():
    parser = argparse.ArgumentParser(
        description='Q-pie-Disassembler! Disassembles the code in a '
        'given module file and outputs assembly.')

    parser.add_argument(
        'module_file',
        help='The module file to disassemble.')

    args = parser.parse_args()

    with open(args.module_file, 'rb') as f:
        module = f.read()

    module = Module.parse(module)
    code = module.sections[Sections.CODE]['data']
    addr = module.sections[Sections.CODE]['addr']

    i = 0

    while i < len(code):
        formatted, size = format_instr_from_stream(code[i:], addr + i)
        print(formatted)
        i += size


def format_instr_from_stream(stream, addr=None):
    """Given an instruction stream (as a bytes object), this function
decodes one instruction from the beginning of the stream and formats
it as human-readable assembly text.

Return value is a (formatted_str, size) tuple in which `size` is the
number of bytes read from the stream.

    """
    instruction = opcode_to_instr[stream[0]]
    text_fmt = [o.text_fmt for o in instruction.operands]
    args = stream[1:instruction.size]
    formatted_args = format_instr_args(args, instruction.bin_fmt, text_fmt)

    spaces = (max_instr_name_len - len(instruction.name) + 1) * ' '
    formatted = f'{instruction.name}{spaces}{formatted_args}'

    if addr:
        formatted = f'{addr:08x}  ' + formatted

    return formatted, instruction.size


def format_instr_args(args, bin_fmt, text_fmt):
    args = struct.unpack('>' + bin_fmt, args)
    assert len(args) == len(text_fmt)
    fargs = []
    for arg, fmt in zip(args, text_fmt):
        if fmt in ['hex1', 'hex2', 'hex4']:
            n = int(fmt[-1]) * 2 + 2 # two digits per byte plus two more for 0x
            farg = f'{arg:#0{n}x}'
        elif fmt == 'float':
            farg = str(arg)
        elif fmt == 'decimal':
            farg = str(arg)
        else:
            assert False, f'Unknown arg format: {fmt}'

        fargs.append(farg)

    fargs = ', '.join(fargs)
    return f'{fargs}'


def def_instr(opcode, name, *operands, stack=None):
    if opcode in opcode_to_instr or name in instr_name_to_instr:
        raise RuntimeError('Duplicate instruction definition.')
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
def_instr(0x59, 'readf_n', SIZE, FIDX, stack='op1')
def_instr(0x5a, 'writef_n', SIZE, FIDX, stack='-op1')
def_instr(0x5b, 'readi_n', SIZE, stack='-4 + op1')
def_instr(0x5c, 'writei_n', SIZE, stack='-4 - op1')
def_instr(0x5d, 'div%', stack=-2)
def_instr(0x5e, 'div&', stack=-4)
def_instr(0x5f, 'div!', stack=-4)
def_instr(0x60, 'div#', stack=-8)
def_instr(0x61, 'not%', stack=0)
def_instr(0x62, 'not&', stack=0)
def_instr(0x63, 'and%', stack=-2)
def_instr(0x64, 'and&', stack=-4)
def_instr(0x65, 'or%', stack=-2)
def_instr(0x66, 'or&', stack=-4)
def_instr(0x67, 'xor%', stack=-2)
def_instr(0x68, 'xor&', stack=-4)

# maximum instruction name length, calculated here for formatting
# purposes.
max_instr_name_len = max(len(name) for name in instr_name_to_instr)

if __name__ == '__main__':
    main()
