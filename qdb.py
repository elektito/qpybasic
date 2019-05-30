#!/usr/bin/env python3

import cmd
import struct
import asm
from vm import Machine

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

        instruction = self.get_cur_instr()
        bin_fmt = ''.join(o.bin_fmt for o in instruction.operands)
        text_fmt = [o.text_fmt for o in instruction.operands]

        args = self.machine.mem.read(instruction.size - 1)
        instr = self.format_instr(instruction.name, args, bin_fmt, text_fmt)
        print(f'NEXT UP: {instr}')


    def get_cur_instr(self):
        self.machine.mem.seek(self.machine.ip)
        opcode = self.machine.mem.read_byte()
        return asm.opcode_to_instr[opcode]


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


    def print_mem(self, addr):
        def m(a):
            self.machine.mem.seek(a)
            n = self.machine.mem.read(2)
            return struct.unpack('>H', n)[0]

        if addr > 4*2**30 - 4*16:
            print('Requested address too large.')
            return

        for i in range(4):
            line = ' '.join('{:04x}'.format(m(addr + a)) for a in range(0, 16, 2))
            print(f'{addr:08x} {line}')
            addr += 16


    def do_mem(self, arg):
        "Display contents of memory at the given address."
        arg = arg.strip()
        if not arg:
            print('Please provide an address.')
            return
        try:
            addr = int(arg, base=0)
        except ValueError:
            addr = -1
        if addr < 0 or addr >= 4*2**30:
            print('Invalid address.')
            return
        self.print_mem(addr)


    def do_next(self, arg):
        """If current instruction is not call, exactly the same as the 'step'
command. If it is a call, then repeatedly 'step' until the instruction
after call is reached.

        """
        if self.machine.stopped:
            print('Nothing more to do.')
            return

        instruction = self.get_cur_instr()
        if instruction.name != 'call':
            return self.do_step('')

        expected_addr = self.machine.ip + 5
        self.do_step('')
        instruction = self.get_cur_instr()
        while self.machine.ip != expected_addr:
            self.do_step('')


    def do_quit(self, arg):
        "Quit interactive debugger."
        return True


    def do_reg(self, arg):
        "Display the value of machine registers."
        ip = self.machine.ip
        sp = self.machine.sp
        fp = self.machine.fp
        print(f'IP={ip:#0{10}x} SP={sp:#0{10}x} FP={fp:#0{10}x}')


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

    machine = Machine.load(sys.argv[1])
    Cmd(machine).cmdloop()
    machine.shutdown()


if __name__ == '__main__':
    main()
