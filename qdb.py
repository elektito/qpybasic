import cmd
import struct
import vm
import asm
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
        instruction = asm.opcode_to_instr[opcode]
        bin_fmt = ''.join(o.bin_fmt for o in instruction.operands)
        text_fmt = [o.text_fmt for o in instruction.operands]

        args = self.machine.mem.read(instruction.size - 1)
        instr = self.format_instr(instruction.name, args, bin_fmt, text_fmt)
        print(f'NEXT UP: {instr}')


    def format_instr(self, name, args, bin_fmt, text_fmt):
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
