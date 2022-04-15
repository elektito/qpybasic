# qpybasic

This is an attempt to create a QBASIC compiler in Python, along with a
VM and tools like a debugger. The compile target is a
stack-machine. Most of QBASIC features are currently supported. DATA
statements are not yet supported though.

You can compile a program by running:

    ./qpybasic.py foo.bas

This will produce a file named `a.mod` which can be run by:

    ./vm.py a.mod

The `a.mod` file can be disassembled by:

    ./asm.py a.mod

Which outputs something like this:

    00100000  call      0x00100006
    00100005  end
    00100006  frame     0
    00100009  pushi%    2
    0010000c  pushi%    2
    0010000f  add%
    00100010  pushi%    10
    00100013  add%
    00100014  pushi%    1
    00100017  pushi%    1
    0010001a  syscall   4
    0010001d  unframe   0
    00100020  ret       0

You can also directly get assembly from qpybasic by passing a `-d` option:

    ./qpybasic.py -d foo.bas

An assembly level debugger is also available which can be used to step
through compiled programs:

    $ ./qdb.py a.mod
    NEXT UP: call 0x00100006

    qdb: qpybasic interactive debugger
    Type help or ? to list commands.

    (qdb IP=0x100000) help

    Documented commands (type help <topic>):
    ========================================
    EOF  help  mem  next  quit  reg  stack  step

    (qdb IP=0x100000) step
    NEXT UP: frame 0
    (qdb IP=0x100006)
    NEXT UP: pushi% 2
    (qdb IP=0x100009)
    NEXT UP: pushi% 2
    (qdb IP=0x10000c)
    NEXT UP: add%
    (qdb IP=0x10000f)
    NEXT UP: pushi% 10
    (qdb IP=0x100010)
    NEXT UP: add%
    (qdb IP=0x100013)
    NEXT UP: pushi% 1
    (qdb IP=0x100014)
    NEXT UP: pushi% 1
    (qdb IP=0x100017)

