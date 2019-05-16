import cmd
import struct
import vm
from mmap import mmap


class Cmd(cmd.Cmd):
    intro = """
qdb: qpybasic interactive debugger
Type help or ? to list commands.
"""
    prompt = '(qdb) '


    def __init__(self, machine):
        super().__init__()
        self.machine = machine

        self.set_prompt()
        self.print_next_up()


    def set_prompt(self):
        ip = hex(self.machine.ip)
        self.prompt = f'(qdb IP={ip}) '


    def print_next_up(self):
        if self.machine.stopped:
            print('Nothing more to do.')
            return

        self.machine.mem.seek(self.machine.ip)
        opcode = self.machine.mem.read_byte()
        name, extra_bytes, bin_fmt, text_fmt = {
            0x01: ('add%', 0, '', ''),
            0x05: ('call', 4, 'I', 'h4'),
            0x13: ('end', 0, '', ''),
            0x14: ('frame', 2, 'H', 'i'),
            0x18: ('jmp', 4, 'I', 'h4'),
            0x19: ('jmpf', 2, 'h', 'i'),
            0x1c: ('lt', 0, '', ''),
            0x26: ('pushi%', 2, 'h', 'i'),
            0x29: ('pushi#', 8, 'd', 'f'),
            0x2a: ('pushi$', 4, 'I', 'h4'),
            0x2d: ('readf4', 2, 'h', 'i'),
            0x2e: ('sub%', 0, '', ''),
            0x32: ('syscall', 2, 'H', 'i'),
            0x34: ('writef2', 2, 'h', 'i'),
            0x37: ('ret', 2, 'H', 'i'),
            0x38: ('unframe', 2, 'H', 'i'),
            0x3b: ('readi2', 0, '', ''),
            0x42: ('pushfp', 2, 'h', 'i'),
            0x4a: ('unframe_r', 6, 'HhH', 'i i i'),
            0x4b: ('ret_r', 4, 'HH', 'i i'),
        }[opcode]
        args = self.machine.mem.read(extra_bytes)
        instr = self.format_instr(name, args, bin_fmt, text_fmt)
        print(f'NEXT UP: {instr}')


    def format_instr(self, name, args, bin_fmt, text_fmt):
        args = struct.unpack('>' + bin_fmt, args)
        text_fmt = text_fmt.split()
        assert len(args) == len(text_fmt)
        fargs = []
        for arg, fmt in zip(args, text_fmt):
            if fmt in ['h1', 'h2', 'h4']:
                n = int(fmt[-1]) * 2 + 2 # two digits per byte plus two more for 0x
                farg = f'{arg:#0{n}x}'
            elif fmt == 'f':
                farg = str(arg)
            elif fmt == 'i':
                farg = str(arg)
            else:
                assert False, f'Unknown arg format: {fmt}'

            fargs.append(farg)

        fargs = ', '.join(fargs)
        return f'{name} {fargs}'


    def do_quit(self, arg):
        "Quit interactive debugger."
        return True


    def do_step(self, arg):
        "Execute one instruction."
        if self.machine.stopped:
            print('Machine is stopped.')
        else:
            self.machine.step()
            self.set_prompt()
            self.print_next_up()


    def do_EOF(self, arg):
        "Quit interactive debugger."
        print()
        return self.do_quit(arg)


def main():
    import sys
    filename = sys.argv[1]

    with open(filename, 'rb') as f:
        print('Reading module file...')
        module = f.read()

    print('Parsing module...')
    sections = vm.parse_module(module)

    print('Mapping module memory...')
    mem = mmap(-1, 4 * 2**30)
    for t, l, a, data in sections:
        mem.seek(a)
        mem.write(data)
        print(f'Mapped section (type={t}) to address {hex(a)}.')

    machine = vm.Machine(mem)

    Cmd(machine).cmdloop()


if __name__ == '__main__':
    main()
