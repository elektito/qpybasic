#!/usr/bin/env python3

import struct
import logging
import logging.config
from mmap import mmap


TRUE = struct.pack('>h', -1)
FALSE = struct.pack('>h', 0)

logger = logging.getLogger(__name__)


class Machine:
    def __init__(self, mem):
        self.mem = mem
        self.sp = 0xffffffff
        self.fp = self.sp
        self.ip = 0x100000


    def launch(self):
        self.stopped = False
        while not self.stopped:
            self.mem.seek(self.ip)
            opcode = self.mem.read_byte()
            opname = {
                0x03: 'add_single',
                0x07: 'conv_int_single',
                0x0c: 'conv_single_int',
                0x13: 'end',
                0x14: 'frame',
                0x17: 'gt',
                0x18: 'jmp',
                0x19: 'jmpf',
                0x1a: 'jmpt',
                0x1c: 'lt',
                0x1f: 'mul_single',
                0x24: 'neg_single',
                0x26: 'pushi_int',
                0x28: 'pushi_single',
                0x2a: 'pushi_string',
                0x2d: 'readf4',
                0x2e: 'sub_int',
                0x30: 'sub_single',
                0x32: 'syscall',
                0x35: 'writef4',
            }[opcode]
            n = getattr(self, f'exec_{opname}')()
            self.ip += n + 1


    def exec_add_single(self):
        x = self.pop(4)
        y = self.pop(4)
        x, = struct.unpack('>f', x)
        y, = struct.unpack('>f', y)
        result = x + y
        result = struct.pack('>f', result)
        self.push(result)
        logger.debug('EXEC: add!')
        return 0


    def exec_conv_int_single(self):
        value = self.pop(2)
        value, = struct.unpack('>h', value)
        value = struct.pack('>f', value)
        self.push(value)
        logger.debug('EXEC: conv%!')
        return 0


    def exec_conv_single_int(self):
        value = self.pop(4)
        value, = struct.unpack('>f', value)
        value = int(value)
        if value > 32767 or value < -32768:
            value = -32768
        value = struct.pack('>h', value)
        self.push(value)
        logger.debug('EXEC: conv%!')
        return 0


    def exec_end(self):
        self.stopped = True
        return 0


    def exec_frame(self):
        frame_size = self.mem.read(2)
        frame_size, = struct.unpack('>H', frame_size)
        self.sp -= frame_size
        self.fp = self.sp
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


    def exec_jmp(self):
        target = self.mem.read(4)
        target, = struct.unpack('>I', target)
        return target - self.ip - 1


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

            # relative jumps are calculated based on the address of
            # the jump instruction itself. the -1 we add is so that we
            # offset the size of the opcode which is automatically
            # added to IP upon return.
            return -1 + offset


    def exec_jmpt(self):
        offset = self.mem.read(2)
        offset, = struct.unpack('>h', offset)
        cond = self.pop(2)
        cond, = struct.unpack('>h', cond)

        logger.debug('EXEC: jmpt')
        if cond != 0:
            # TRUE

            # relative jumps are calculated based on the address of
            # the jump instruction itself. the -1 we add is so that we
            # offset the size of the opcode which is automatically
            # added to IP upon return.
            return -1 + offset
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


    def exec_mul_single(self):
        x = self.pop(4)
        y = self.pop(4)
        x, = struct.unpack('>f', x)
        y, = struct.unpack('>f', y)
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


    def exec_pushi_int(self):
        value = self.mem.read(2)
        self.push(value)
        logger.debug('EXEC: pushi%')
        return 2


    def exec_pushi_single(self):
        value = self.mem.read(4)
        self.push(value)
        logger.debug('EXEC: pushi!')
        return 4


    def exec_pushi_string(self):
        value = self.mem.read(4)
        self.push(value)
        logger.debug('EXEC: pushi$')
        return 4


    def exec_readf4(self):
        idx = self.mem.read(2)
        idx, = struct.unpack('>H', idx)
        self.mem.seek(self.fp + idx)
        value = self.mem.read(2)
        self.push(value)
        logger.debug('EXEC: readf4')
        return 2


    def exec_sub_int(self):
        x = self.pop(2)
        y = self.pop(2)
        x, = struct.unpack('>h', x)
        y, = struct.unpack('>h', y)
        result = x - y
        if result < -32768:
            result = -32768
        result = struct.pack('>h', result)
        self.push(result)
        logger.debug('EXEC: sub%')
        return 0


    def exec_sub_single(self):
        x = self.pop(4)
        y = self.pop(4)
        x, = struct.unpack('>f', x)
        y, = struct.unpack('>f', y)
        result = x - y
        result = struct.pack('>f', result)
        self.push(result)
        logger.debug('EXEC: sub!')
        return 0


    def exec_syscall(self):
        value = self.mem.read(2)
        value, = struct.unpack('>H', value)
        if value == 0x02: #cls
            pass
        elif value == 0x03: #concat
            pass
        elif value == 0x04: #print
            self.syscall_print()
        logger.debug('EXEC: syscall')
        return 2


    def exec_writef4(self):
        idx = self.mem.read(2)
        idx, = struct.unpack('>H', idx)
        value = self.pop(4)
        self.mem.seek(self.fp + idx)
        self.mem.write(value)
        logger.debug('EXEC: writef4')
        return 2


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


    def syscall_print(self):
        logger.debug('SYSCALL: print')
        buf = ''
        def print_number(n):
            nonlocal buf
            if n > 0:
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

        print(buf, end='')


    def error(self, msg):
        logger.error(msg)
        exit(1)


def parse_module(module):
    file_hdr = module[:5]
    magic = file_hdr[:2]
    if magic != b'QB':
        logger.error('Invalid magic. Module is not valid.')
        exit(1)
    version, nsections = struct.unpack('>BH', file_hdr[2:])

    module = module[5:]

    sections = []
    for i in range(nsections):
        hdr = module[:9]
        t, l, a = struct.unpack('>BII', hdr)
        data = module[9:9+l]
        assert len(data) == l
        sections.append((t, l, a, data))
        module = module[9+l:]

    return sections


def main():
    import sys
    filename = sys.argv[1]

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
                'level': 'DEBUG',
                'propagate': True,
            }
        },
    })

    with open(filename, 'rb') as f:
        logger.info('Reading module file...')
        module = f.read()

    logger.info('Parsing module...')
    sections = parse_module(module)

    logger.info('Mapping module memory...')
    mem = mmap(-1, 4 * 2**30)
    for t, l, a, data in sections:
        mem.seek(a)
        mem.write(data)
        logger.info(f'Mapped section (type={t}) to address {hex(a)}.')

    machine = Machine(mem)
    machine.launch()


if __name__ == '__main__':
    main()
