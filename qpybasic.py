import argparse
import codecs
from compiler import Compiler, CompileError


def main():
    parser = argparse.ArgumentParser(
        description='QpyBASIC')

    parser.add_argument('input_file', help='Input source code.')
    parser.add_argument(
        '--encoding', '-e', default='latin-1',
        help='The encoding of the input file. Defaults to latin-1.')
    parser.add_argument(
        '--dump-asm', '-d', action='store_true', default=False,
        help='Dump assembly output to stdout.')
    parser.add_argument(
        '--output', '-o', default='a.mod',
        help='The output file. Defaults to "a.mod".')

    args = parser.parse_args()

    with codecs.open(args.input_file, encoding=args.encoding) as f:
        source = f.read()

    with open('qpybasic.ebnf') as f:
        grammar_text = f.read()

    c = Compiler()
    try:
        module = c.compile(source)
    except CompileError as e:
        print('COMPILE ERROR:', e)
        exit(1)

    if args.dump_asm:
        for i in c.instrs:
            print(i)

    with open(args.output, 'wb') as f:
        f.write(module.dump())


if __name__ == '__main__':
    main()
