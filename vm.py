#!/usr/bin/env python3

import struct
import logging
import argparse
import asm
import logging.config
from mmap import mmap
from functools import reduce
from enum import IntEnum, unique
from compiler import Module
from malloc import Allocator, InvalidPointer


TRUE = struct.pack('>h', -1)
FALSE = struct.pack('>h', 0)

logger = logging.getLogger(__name__)


@unique
class ErrorCodes(IntEnum):
    INVALID_ARRAY_BOUNDS = 1
    INVALID_MEMSET_VALUE = 2
    SUBSCRIPT_OUT_OF_RANGE = 3
    OUT_OF_MEMORY = 4
    INVALID_POINTER = 5


    def __str__(self):
        return {
            self.INVALID_ARRAY_BOUNDS:
            'Invalid array bounds',

            self.INVALID_MEMSET_VALUE:
            'Invalid value for memset',

            self.SUBSCRIPT_OUT_OF_RANGE:
            'Subscript out of range',

            self.OUT_OF_MEMORY:
            'Out of memory',

            self.INVALID_POINTER:
            'Invalid pointer',
        }.get(int(self), super().__str__())


RE = ErrorCodes


class MachineRuntimeError(Exception):
    def __init__(self, code, msg=None):
        assert isinstance(code, ErrorCodes)

        if not msg:
            msg = str(code)

        self.code = code

        super().__init__(msg)


class Jump:
    def __init__(self, target):
        self.target = target


class Machine:
    def __init__(self, mem):
        self.mem = mem
        self.sp = 0xffffffff
        self.fp = self.sp
        self.ip = 0x100000
        self.stopped = False

        self.check_stack_changes = True
        self.event_handler = self.event_handler_routine

        HEAP_START = 0x40000000
        HEAP_SIZE = 0x40000000
        self.allocator = Allocator([(HEAP_START, HEAP_SIZE)])


    def event_handler_routine(self, event):
        event_name, *args = event
        if event_name == 'cls':
            seq =  '\033[2J'    # clear screen
            seq += '\033[1;1H'  # move cursor to screen top-left
            print(seq)
        elif event_name == 'print':
            buf, = args
            print(buf, end='')
        elif event_name == 'error':
            code, = args
            msg = str(ErrorCodes(code))
            logger.error(f'Machine error: {msg}')
        else:
            logger.error(f'Unknown machine event: {event_name}')


    def launch(self):
        self.stopped = False
        while not self.stopped:
            self.step()


    def step(self):
        def safe_name(s):
            for c in '%&!#$':
                r = {
                    '%': '_integer',
                    '&': '_long',
                    '!': '_single',
                    '#': '_double',
                    '$': '_pointer'
                }[c]
                s = s.replace(c, r)

            return s

        self.mem.seek(self.ip)
        opcode = self.mem.read_byte()
        instr = asm.opcode_to_instr[opcode]
        prev_sp = self.sp
        opname = safe_name(instr.name)
        try:
            n = getattr(self, f'exec_{opname}')()
        except MachineRuntimeError as e:
            self.event_handler(('error', e.code))
            self.stopped = True
        else:
            self.check_stack(instr, prev_sp, self.sp)
            if isinstance(n, Jump):
                # return value is an absolute jump.
                self.ip = n.target
            else:
                # return value is the number of bytes used from the
                # instruction stream by the exec function.
                self.ip += n + 1


    def parse_operands(self, instr):
        bin_fmt = '>' + ''.join(o.bin_fmt for o in instr.operands)
        size = sum(o.size for o in instr.operands)
        self.mem.seek(self.ip + 1)
        operands = self.mem.read(size)
        operands = struct.unpack(bin_fmt, operands)
        return operands


    def check_stack(self, instr, old_sp, new_sp):
        if not self.check_stack_changes:
            return
        if instr.stack == None:
            return
        if isinstance(instr.stack, int):
            expected_diff = instr.stack
        else:
            operands = self.parse_operands(instr)
            env = {f'op{i+1}': operands[i] for i in range(len(operands))}
            expected_diff = eval(instr.stack, env)

        # our stack is from top to bottom, but with the numbers we get
        # from instruction stack checks, positive means adding to
        # stack. Here we negate our number to fix this.
        diff = -(new_sp - old_sp)
        assert diff == expected_diff


    def exec_add_integer(self):
        y = self.pop(2)
        x = self.pop(2)
        y, = struct.unpack('>h', y)
        x, = struct.unpack('>h', x)
        result = x + y
        if result > 32767 or result < -32768:
            result = -32768
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: add%')
        return 0


    def exec_add_long(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>i', y)
        x, = struct.unpack('>i', x)
        result = x + y
        if result > 2**31-1 or result < -2**31:
            result = -2**31
        result = struct.pack('>i', result)
        self.push(result)
        logger.debug(f'EXEC: add& ({x} + {y} = {struct.unpack(">i", result)[0]})')
        return 0


    def exec_add_ul(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>I', y)
        x, = struct.unpack('>I', x)
        result = x + y
        if result >= 2**32:
            result = 2**32 - 1
        result = struct.pack('>I', result)
        self.push(result)
        logger.debug(f'EXEC: add_ul ({hex(x)} + {hex(y)} = {hex(struct.unpack(">I", result)[0])})')
        return 0


    def exec_add_single(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>f', y)
        x, = struct.unpack('>f', x)
        result = x + y
        result = struct.pack('>f', result)
        self.push(result)
        logger.debug('EXEC: add!')
        return 0


    def exec_and_integer(self):
        y = self.pop(2)
        x = self.pop(2)
        y, = struct.unpack('>h', y)
        x, = struct.unpack('>h', x)
        result = x & y
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: and%')
        return 0


    def exec_and_long(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>i', y)
        x, = struct.unpack('>i', x)
        result = x & y
        result = struct.pack('>i', result)
        self.push(result)
        logger.debug('EXEC: and&')
        return 0


    def exec_call(self):
        target = self.mem.read(4)
        target, = struct.unpack('>I', target)
        self.push(struct.pack('>I', self.ip + 5))
        logger.debug('EXEC: call')
        return Jump(target)


    def exec_conv_integer_long(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        value = struct.pack('>i', value)
        self.push(value)
        logger.debug('EXEC: conv%&')
        return 0


    def exec_conv_integer_double(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        value = struct.pack('>d', value)
        self.push(value)
        logger.debug('EXEC: conv%#')
        return 0


    def exec_conv_integer_ul(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        value = struct.pack('>I', value)
        self.push(value)
        logger.debug('EXEC: conv%_ul')
        return 0


    def exec_conv_integer_single(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        value = struct.pack('>f', value)
        self.push(value)
        logger.debug('EXEC: conv%!')
        return 0


    def exec_conv_long_integer(self):
        value = self.pop(4)
        value, = struct.unpack('>i', value)
        value = int(value)
        if value > 32767 or value < -32768:
            value = -32768
        value = struct.pack('>h', value)
        self.push(value)
        logger.debug('EXEC: conv&%')
        return 0


    def exec_conv_long_single(self):
        value = self.pop(4)
        value, = struct.unpack('>i', value)
        value = struct.pack('>f', value)
        self.push(value)
        logger.debug('EXEC: conv&%')
        return 0


    def exec_conv_single_integer(self):
        value = self.pop(4)
        value, = struct.unpack('>f', value)
        value = int(value)
        if value > 32767 or value < -32768:
            value = -32768
        value = struct.pack('>h', value)
        self.push(value)
        logger.debug('EXEC: conv!%')
        return 0


    def exec_conv_single_long(self):
        value = self.pop(4)
        value, = struct.unpack('>f', value)
        value = int(value)
        if value > 2**31-1 or value < -2**31:
            value = -2**31
        value = struct.pack('>i', value)
        self.push(value)
        logger.debug('EXEC: conv!&')
        return 0


    def exec_conv_single_double(self):
        value = self.pop(4)
        value, = struct.unpack('>f', value)
        value = int(value)
        value = struct.pack('>d', value)
        self.push(value)
        logger.debug('EXEC: conv!#')
        return 0


    def exec_conv_single_ul(self):
        value = self.pop(4)
        value, = struct.unpack('>f', value)
        value = int(value)
        if value >= 2**32:
            value = 2**32 - 1
        elif value < 0:
            value = 0
        value = struct.pack('>I', value)
        self.push(value)
        logger.debug('EXEC: conv!_ul')
        return 0


    def exec_conv_double_integer(self):
        value = self.pop(8)
        value, = struct.unpack('>d', value)
        value = int(value)
        if value > 32767 or value < -32768:
            value = -32768
        value = struct.pack('>h', value)
        self.push(value)
        logger.debug('EXEC: conv#%')
        return 0


    def exec_conv_double_long(self):
        value = self.pop(8)
        value, = struct.unpack('>d', value)
        value = int(value)
        if value > 2**31-1 or value < -2**31:
            value = -2**31
        value = struct.pack('>i', value)
        self.push(value)
        logger.debug('EXEC: conv#&')
        return 0


    def exec_conv_double_single(self):
        value = self.pop(8)
        value, = struct.unpack('>d', value)
        value = struct.pack('>f', value)
        self.push(value)
        logger.debug('EXEC: conv#!')
        return 0


    def exec_div_long(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>i', y)
        x, = struct.unpack('>i', x)
        result = x // y
        result = struct.pack('>i', result)
        self.push(result)
        logger.debug('EXEC: div&')
        return 0



    def exec_dup2(self):
        value = self.pop(2)
        self.push(value)
        self.push(value)
        logger.debug('EXEC: dup2')
        return 0


    def exec_end(self):
        self.stopped = True
        logger.debug('EXEC: end')
        return 0


    def exec_eq(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        if value == 0:
            self.push(TRUE)
        else:
            self.push(FALSE)
        logger.debug('EXEC: eq')
        return 0


    def exec_frame(self):
        frame_size = self.mem.read(2)
        frame_size, = struct.unpack('>H', frame_size)
        self.push(struct.pack('>I', self.fp))
        self.fp = self.sp
        self.sp -= frame_size
        logger.debug('EXEC: frame')
        return 2


    def exec_gt(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        if value > 0:
            self.push(TRUE)
        else:
            self.push(FALSE)
        logger.debug('EXEC: gt')
        return 0


    def exec_ge(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        if value >= 0:
            self.push(TRUE)
        else:
            self.push(FALSE)
        logger.debug('EXEC: ge')
        return 0


    def exec_le(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        if value <= 0:
            self.push(TRUE)
        else:
            self.push(FALSE)
        logger.debug('EXEC: le')
        return 0


    def exec_jmp(self):
        target = self.mem.read(4)
        target, = struct.unpack('>I', target)
        return Jump(target)


    def exec_jmpf(self):
        offset = self.mem.read(2)
        offset, = struct.unpack('>h', offset)
        cond = self.pop(2)
        cond, = struct.unpack('>h', cond)

        logger.debug('EXEC: jmpf')
        if cond != 0:
            # TRUE
            return 2
        else:
            # FALSE
            return Jump(self.ip + offset)


    def exec_jmpt(self):
        offset = self.mem.read(2)
        offset, = struct.unpack('>h', offset)
        cond = self.pop(2)
        cond, = struct.unpack('>h', cond)

        logger.debug('EXEC: jmpt')
        if cond != 0:
            # TRUE
            return Jump(self.ip + offset)
        else:
            # FALSE
            return 2


    def exec_lt(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        if value < 0:
            self.push(TRUE)
        else:
            self.push(FALSE)
        logger.debug('EXEC: lt')
        return 0


    def exec_mul_integer(self):
        y = self.pop(2)
        x = self.pop(2)
        y, = struct.unpack('>h', y)
        x, = struct.unpack('>h', x)
        result = x * y
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: mul%')
        return 0


    def exec_mul_long(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>i', y)
        x, = struct.unpack('>i', x)
        result = x * y
        result = struct.pack('>i', result)
        self.push(result)
        logger.debug('EXEC: mul&')
        return 0


    def exec_mul_ul(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>I', y)
        x, = struct.unpack('>I', x)
        result = x * y
        result = struct.pack('>I', result)
        self.push(result)
        logger.debug(f'EXEC: mul_ul ({hex(x)} * {hex(y)} = {hex(struct.unpack(">I", result)[0])})')
        return 0


    def exec_mul_single(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>f', y)
        x, = struct.unpack('>f', x)
        result = x * y
        result = struct.pack('>f', result)
        self.push(result)
        logger.debug('EXEC: mul!')
        return 0


    def exec_neg_single(self):
        x = self.pop(4)
        x, = struct.unpack('>f', x)
        result = -x
        result = struct.pack('>f', result)
        self.push(result)
        logger.debug('EXEC: neg!')
        return 0


    def exec_neg_integer(self):
        x = self.pop(2)
        x, = struct.unpack('>h', x)
        result = -x
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: neg%')
        return 0


    def exec_not_integer(self):
        x = self.pop(2)
        x, = struct.unpack('>h', x)
        result = ~x
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: not%')
        return 0


    def exec_not_long(self):
        x = self.pop(4)
        x, = struct.unpack('>i', x)
        result = ~x
        result = struct.pack('>i', result)
        self.push(result)
        logger.debug('EXEC: not&')
        return 0


    def exec_or_integer(self):
        y = self.pop(2)
        x = self.pop(2)
        y, = struct.unpack('>h', y)
        x, = struct.unpack('>h', x)
        result = x | y
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: or%')
        return 0


    def exec_or_long(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>i', y)
        x, = struct.unpack('>i', x)
        result = x | y
        result = struct.pack('>i', result)
        self.push(result)
        logger.debug('EXEC: or&')
        return 0


    def exec_pushfp(self):
        idx = self.mem.read(2)
        idx, = struct.unpack('>h', idx)
        addr = self.fp + idx
        self.push(struct.pack('>I', addr))
        logger.debug(f'EXEC: pushfp ({hex(addr)})')
        return 2


    def exec_pushi_integer(self):
        value = self.mem.read(2)
        self.push(value)
        logger.debug('EXEC: pushi%')
        return 2


    def exec_pushi_long(self):
        value = self.mem.read(4)
        self.push(value)
        logger.debug('EXEC: pushi&')
        return 4


    def exec_pushi_ul(self):
        value = self.mem.read(4)
        self.push(value)
        logger.debug('EXEC: pushi_ul')
        return 4


    def exec_pushi_single(self):
        value = self.mem.read(4)
        self.push(value)
        logger.debug('EXEC: pushi!')
        return 4


    def exec_pushi_double(self):
        value = self.mem.read(8)
        self.push(value)
        logger.debug('EXEC: pushi#')
        return 8


    def exec_pushi_pointer(self):
        value = self.mem.read(4)
        self.push(value)
        logger.debug('EXEC: pushi$')
        return 4


    def exec_readf2(self):
        idx = self.mem.read(2)
        idx, = struct.unpack('>h', idx)
        self.mem.seek(self.fp + idx)
        value = self.mem.read(2)
        self.push(value)
        logger.debug('EXEC: readf2')
        return 2


    def exec_readf4(self):
        idx = self.mem.read(2)
        idx, = struct.unpack('>h', idx)
        self.mem.seek(self.fp + idx)
        value = self.mem.read(4)
        self.push(value)
        logger.debug('EXEC: readf4')
        return 2


    def exec_readf8(self):
        idx = self.mem.read(2)
        idx, = struct.unpack('>h', idx)
        self.mem.seek(self.fp + idx)
        value = self.mem.read(8)
        self.push(value)
        logger.debug('EXEC: readf8')
        return 2


    def exec_readf_n(self):
        size = self.mem.read(2)
        size, = struct.unpack('>H', size)
        idx = self.mem.read(2)
        idx, = struct.unpack('>h', idx)
        self.mem.seek(self.fp + idx)
        value = self.mem.read(size)
        self.push(value)
        logger.debug('EXEC: readf_n')
        return 4


    def exec_readi2(self):
        addr, = struct.unpack('>I', self.pop(4))
        self.mem.seek(addr)
        value = self.mem.read(2)
        self.push(value)
        logger.debug('EXEC: readi2')
        return 0


    def exec_readi4(self):
        addr, = struct.unpack('>I', self.pop(4))
        self.mem.seek(addr)
        value = self.mem.read(4)
        self.push(value)
        logger.debug(f'EXEC: readi4 from address {hex(addr)}. got value: {hex(struct.unpack(">I", value)[0])}')
        return 0


    def exec_readi8(self):
        addr, = struct.unpack('>I', self.pop(4))
        self.mem.seek(addr)
        value = self.mem.read(8)
        self.push(value)
        logger.debug(f'EXEC: readi8 from address {hex(addr)}. got value: {hex(struct.unpack(">Q", value)[0])}')
        return 0


    def exec_ret(self):
        arg_size = self.mem.read(2)
        arg_size, = struct.unpack('>H', arg_size)
        target, = struct.unpack('>I', self.pop(4))
        self.sp += arg_size
        logger.debug('EXEC: ret')
        return Jump(target)


    def exec_ret_r(self):
        arg_size = self.mem.read(2)
        retv_size = self.mem.read(2)
        arg_size, = struct.unpack('>H', arg_size)
        retv_size, = struct.unpack('>H', retv_size)
        retv = self.pop(retv_size)
        target, = struct.unpack('>I', self.pop(4))
        self.sp += arg_size
        self.push(retv)
        logger.debug('EXEC: ret_r')
        return Jump(target)


    def exec_sgn_integer(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        if value > 0:
            self.push(struct.pack('>h', 1))
        elif value == 0:
            self.push(struct.pack('>h', 0))
        else:
            self.push(struct.pack('>h', -1))
        logger.debug('EXEC: sgn%')
        return 0


    def exec_sub_integer(self):
        y = self.pop(2)
        x = self.pop(2)
        y, = struct.unpack('>h', y)
        x, = struct.unpack('>h', x)
        result = x - y
        if result < -32768:
            result = -32768
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: sub%')
        return 0


    def exec_sub_long(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>i', y)
        x, = struct.unpack('>i', x)
        result = x - y
        if result > 2**31-1 or result < -2**31:
            result = -2**31
        result = struct.pack('>i', result)
        self.push(result)
        logger.debug(f'EXEC: sub& ({x} - {y} = {struct.unpack(">i", result)[0]})')
        return 0


    def exec_sub_ul(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>I', y)
        x, = struct.unpack('>I', x)
        result = x - y
        if result < 0:
            result = 0
        result = struct.pack('>I', result)
        self.push(result)
        logger.debug(f'EXEC: sub_ul ({x} - {y} = {struct.unpack(">I", result)[0]})')
        return 0


    def exec_sub_single(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>f', y)
        x, = struct.unpack('>f', x)
        result = x - y
        result = struct.pack('>f', result)
        self.push(result)
        logger.debug('EXEC: sub!')
        return 0


    def exec_sub_double(self):
        y = self.pop(8)
        x = self.pop(8)
        y, = struct.unpack('>d', y)
        x, = struct.unpack('>d', x)
        result = x - y
        result = struct.pack('>d', result)
        self.push(result)
        logger.debug('EXEC: sub#')
        return 0


    def exec_syscall(self):
        value = self.mem.read(2)
        value, = struct.unpack('>H', value)
        if value == 0x02: #cls
            self.syscall_cls()
        elif value == 0x03: #concat
            logger.error('CONCAT not implemented yet. corruption might occur.')
        elif value == 0x04: #print
            self.syscall_print()
        elif value == 0x05: # init_array
            self.syscall_init_array()
        elif value == 0x06: # memset
            self.syscall_memset()
        elif value == 0x07: # access_array
            self.syscall_access_array()
        elif value == 0x08:
            self.syscall_malloc()
        elif value == 0x09:
            self.syscall_free()
        else:
            logger.error('Invalid syscall number.')
        logger.debug('EXEC: syscall')
        return 2


    def exec_unframe(self):
        frame_size = self.mem.read(2)
        frame_size, = struct.unpack('>H', frame_size)
        self.sp = self.fp
        self.fp, = struct.unpack('>I', self.pop(4))
        logger.debug('EXEC: unframe')
        return 2


    def exec_unframe_r(self):
        frame_size = self.mem.read(2)
        retv_idx = self.mem.read(2)
        retv_size = self.mem.read(2)
        frame_size, = struct.unpack('>H', frame_size)
        retv_idx, = struct.unpack('>h', retv_idx)
        retv_size, = struct.unpack('>H', retv_size)

        # read return value
        self.mem.seek(self.fp + retv_idx)
        retv = self.mem.read(retv_size)

        self.sp = self.fp
        self.fp, = struct.unpack('>I', self.pop(4))
        self.push(retv)
        logger.debug('EXEC: unframe_r')
        return 6


    def exec_writef2(self):
        idx = self.mem.read(2)
        idx, = struct.unpack('>h', idx)
        value = self.pop(2)
        self.mem.seek(self.fp + idx)
        self.mem.write(value)
        logger.debug('EXEC: writef2')
        return 2


    def exec_writef4(self):
        idx = self.mem.read(2)
        idx, = struct.unpack('>h', idx)
        value = self.pop(4)
        self.mem.seek(self.fp + idx)
        self.mem.write(value)
        logger.debug('EXEC: writef4')
        return 2


    def exec_writef_n(self):
        size = self.mem.read(2)
        size, = struct.unpack('>H', size)
        idx = self.mem.read(2)
        idx, = struct.unpack('>h', idx)
        value = self.pop(size)
        self.mem.seek(self.fp + idx)
        self.mem.write(value)
        logger.debug('EXEC: writef_n')
        return 4


    def exec_writei2(self):
        addr, = struct.unpack('>I', self.pop(4))
        value = self.pop(2)
        self.mem.seek(addr)
        self.mem.write(value)
        logger.debug(f'EXEC: writei2 from address {hex(addr)}. wrote value: {hex(struct.unpack(">H", value)[0])}')
        return 0


    def exec_writei4(self):
        addr, = struct.unpack('>I', self.pop(4))
        value = self.pop(4)
        self.mem.seek(addr)
        self.mem.write(value)
        logger.debug(f'EXEC: writei4 from address {hex(addr)}. wrote value: {hex(struct.unpack(">I", value)[0])}')
        return 0


    def exec_writei8(self):
        addr, = struct.unpack('>I', self.pop(4))
        value = self.pop(8)
        self.mem.seek(addr)
        self.mem.write(value)
        logger.debug(f'EXEC: writei8 from address {hex(addr)}. wrote value: {hex(struct.unpack(">Q", value)[0])}')
        return 0


    def exec_writei_n(self):
        size = self.mem.read(2)
        size, = struct.unpack('>H', size)
        addr, = struct.unpack('>I', self.pop(4))
        value = self.pop(size)
        self.mem.seek(addr)
        self.mem.write(value)
        logger.debug(f'EXEC: writei_n')
        return 2


    def exec_xor_integer(self):
        y = self.pop(2)
        x = self.pop(2)
        y, = struct.unpack('>h', y)
        x, = struct.unpack('>h', x)
        result = x ^ y
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: or%')
        return 0


    def exec_xor_long(self):
        y = self.pop(4)
        x = self.pop(4)
        y, = struct.unpack('>i', y)
        x, = struct.unpack('>i', x)
        result = x ^ y
        result = struct.pack('>i', result)
        self.push(result)
        logger.debug('EXEC: or&')
        return 0


    def pop(self, size):
        self.mem.seek(self.sp)
        value = self.mem.read(size)
        self.sp += size
        logger.debug(f'POP: {size}')
        return value


    def push(self, value):
        self.sp -= len(value)
        self.mem.seek(self.sp)
        self.mem.write(value)
        logger.debug(f'PUSH: {len(value)}')


    def read_string(self, addr):
        self.mem.seek(addr)
        length = self.mem.read(2)
        length, = struct.unpack('>h', length)
        s = self.mem.read(length)
        logger.debug(f'READ string of length {length} from addr {hex(addr)}: "{s.decode("ascii")}"')
        return s


    def syscall_cls(self):
        logger.debug('SYSCALL: cls')
        self.event_handler(('cls',))


    def syscall_print(self):
        logger.debug('SYSCALL: print')
        buf = ''
        def print_number(n):
            nonlocal buf
            if n >= 0:
                buf += f' {n} '
            else:
                buf += f'{n} '

        nargs = self.pop(2)
        nargs, = struct.unpack('>h', nargs)
        logger.info(f'nargs is {nargs}')
        for i in range(nargs):
            typeid = self.pop(2)
            typeid, = struct.unpack('>h', typeid)
            if typeid == 1: #int
                n = self.pop(2)
                n, = struct.unpack('>h', n)
                print_number(n)
            elif typeid == 2: # long
                n = self.pop(4)
                n, = struct.unpack('>i', n)
                print_number(n)
            elif typeid == 3: # single
                n = self.pop(4)
                n, = struct.unpack('>f', n)
                print_number(n)
            elif typeid == 4: # double
                n = self.pop(8)
                n, = struct.unpack('>d', n)
                print_number(n)
            elif typeid == 5: # string
                n = self.pop(4)
                n, = struct.unpack('>I', n)
                s = self.read_string(n)
                buf += s.decode('ascii')
            elif typeid == 6: # semicolon
                pass # nothing to do
            elif typeid == 7: # comma
                n = 14 - (len(buf) % 14)
                buf += n * ' '
            else:
               self.error(f'Unknown PRINT type id: {typeid}')

        # if last type id is a semicolon or a comma do not add a
        # newline
        if nargs == 0 or typeid not in (6, 7):
            buf += '\n'

        self.event_handler(('print', buf))


    def syscall_init_array(self):
        addr = self.pop(4)
        addr, = struct.unpack('>I', addr)
        element_size = self.pop(2)
        element_size, = struct.unpack('>h', element_size)
        ndims = self.pop(2)
        ndims, = struct.unpack('>h', ndims)
        dims = []
        for i in range(ndims):
            d_from, = struct.unpack('>h', self.pop(2))
            d_to, = struct.unpack('>h', self.pop(2))
            if d_from > d_to:
                raise MachineRuntimeError(RE.INVALID_ARRAY_BOUNDS)
            dims.append((d_from, d_to))
        ranges = [d_to - d_from + 1 for d_from, d_to in dims]
        size = reduce(lambda x, y: x*y, ranges) * element_size

        self.mem.seek(addr)
        self.mem.write(struct.pack('>h', ndims))
        for d_from, d_to in dims:
            self.mem.write(struct.pack('>h', d_from))
            self.mem.write(struct.pack('>h', d_to))

        self.mem.write(size * bytes([0]))


    def syscall_memset(self):
        addr = self.pop(4)
        addr, = struct.unpack('>I', addr)
        n = self.pop(4)
        n, = struct.unpack('>I', n)
        b = self.pop(2)
        b, = struct.unpack('>h', b)
        if b < 0 or b > 255:
            raise MachineRuntimeError(RE.INVALID_MEMSET_VALUE)
        self.mem.seek(addr)
        self.mem.write(n * bytes([b]))


    def syscall_access_array(self):
        addr, = struct.unpack('>I', self.pop(4))
        element_size, = struct.unpack('>h', self.pop(2))
        n_indices, = struct.unpack('>h', self.pop(2))
        indices = []
        for i in range(n_indices):
            indices.append(struct.unpack('>h', self.pop(2))[0])

        # now read array header
        self.mem.seek(addr)
        ndims, = struct.unpack('>h', self.mem.read(2))
        dims = []
        for i in range(ndims):
            d_from, d_to = struct.unpack('>hh', self.mem.read(4))
            dims.append((d_from, d_to))

        if ndims != n_indices:
            raise MachineRuntimeError(RE.SUBSCRIPT_OUT_OF_RANGE)

        offset = 0
        prev_dim_size = 1
        dims = reversed(dims)
        for idx, (d_from, d_to) in list(zip(indices, dims)):
            if idx < d_from or idx > d_to:
                raise MachineRuntimeError(RE.SUBSCRIPT_OUT_OF_RANGE)

            offset += (idx - d_from) * prev_dim_size

        header_size = 2 + 2 * 2 * ndims
        addr += header_size + offset * element_size
        addr = struct.pack('>I', addr)

        self.push(addr)


    def syscall_malloc(self):
        size, = struct.unpack('>I', self.pop(4))
        try:
            ptr = self.allocator.malloc(size)
        except MemoryError:
            raise MachineRuntimeError(RE.OUT_OF_MEMORY)
        self.push(struct.pack('>I', ptr))


    def syscall_free(self):
        ptr, = struct.unpack('>I', self.pop(4))
        try:
            self.allocator.free(ptr)
        except InvalidPointer as e:
            raise MachineRuntimeError(RE.INVALID_POINTER, f'Invalid pointer: {e.ptr:x}')


    def error(self, msg):
        logger.error(msg)
        exit(1)


    def shutdown(self):
        self.stopped = True
        self.mem.close()


    @staticmethod
    def load(module):
        if isinstance(module, str):
            with open(module, 'rb') as f:
                logger.info('Reading module file...')
                module = f.read()

            logger.info('Parsing module...')
            module = Module.parse(module)
        elif hasattr(module, 'read'):
            logger.info('Reading module file...')
            module = module.read()

            logger.info('Parsing module...')
            module = Module.parse(module)
        else:
            assert isinstance(module, Module)

        logger.info('Mapping module memory...')
        mem = mmap(-1, 4 * 2**30)
        for sec_type, sec in module.sections.items():
            mem.seek(sec['addr'])
            mem.write(sec['data'])
            logger.info(f'Mapped section (type={sec_type}) to address {hex(sec["addr"])}.')

        return Machine(mem)


def main():
    parser = argparse.ArgumentParser(
        description='Run a qpybasic module.')

    parser.add_argument('module_file', help='The module file to load.')
    args = parser.parse_args()

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s %(name)s [%(levelname)s]: %(message)s',
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stderr',
            },
        },
        'loggers': {
            __name__: {
                'handlers': ['console'],
                'level': 'ERROR',
                'propagate': True,
            }
        },
    })

    machine = Machine.load(args.module_file)
    machine.launch()
    machine.shutdown()


if __name__ == '__main__':
    main()
