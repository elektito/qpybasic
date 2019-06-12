import enum
import struct


class Sections(enum.IntEnum):
    CODE = 1
    STRINGS = 2
    SHARED = 3


class Module:
    file_magic = b'QB'
    version = 1

    # This is the address programs are loaded, right after where
    # conventional memory and upper memory area were back in the bad
    # old days.
    INITIAL_ADDR = 0x100000

    # The address in which string literals will be loaded.
    STRING_ADDR = 0x80000000

    # The address in which shared variables will be loaded.
    SHARED_ADDR = 0x20000000


    def __init__(self, sections):
        # at the moment only three sections are supported and all
        # three must be present.
        assert set(sections) == {Sections.CODE,
                                 Sections.STRINGS,
                                 Sections.SHARED}

        self.sections = sections


    def dump(self):
        # file header: magic, version, nsections
        file_hdr = self.file_magic + struct.pack('>BH', self.version, 3)

        # code section header: type, len, load_addr
        code_section = self.sections[Sections.CODE]
        code_section_hdr = struct.pack('>BII',
                                       int(Sections.CODE),
                                       len(code_section['data']),
                                       code_section['addr'])

        # const section header: type, len, load_addr
        str_section = self.sections[Sections.STRINGS]
        str_section_hdr = struct.pack('>BII',
                                      int(Sections.STRINGS),
                                      len(str_section['data']),
                                      str_section['addr'])

        # shared section header: type, len, load_addr
        shared_section = self.sections[Sections.SHARED]
        shared_section_hdr = struct.pack('>BII',
                                         int(Sections.SHARED),
                                         len(shared_section['data']),
                                         shared_section['addr'])

        return file_hdr + \
            code_section_hdr + \
            code_section['data'] + \
            str_section_hdr + \
            str_section['data'] + \
            shared_section_hdr + \
            shared_section['data']


    @staticmethod
    def parse(module):
        file_hdr = module[:5]
        magic = file_hdr[:2]
        if magic != Module.file_magic:
            raise RuntimeError('Invalid magic. Module is not valid.')
        version, nsections = struct.unpack('>BH', file_hdr[2:])
        if version != Module.version:
            raise RuntimeError('Unsupported module version.')

        module = module[5:]

        sections = {}
        for i in range(nsections):
            hdr = module[:9]
            t, l, a = struct.unpack('>BII', hdr)
            data = module[9:9+l]
            assert len(data) == l
            sections[Sections(t)] = {
                'addr': a,
                'data': data,
            }
            module = module[9+l:]

        return Module(sections)


    @staticmethod
    def create(bytecode, string_literals, shared_vars):
        # the string_literals passed to this function is a dictionary
        # which maps each string to its index in the string
        # section.
        string_literals = list(sorted(string_literals.items(), key=lambda r: r[1]))
        string_literals = [value for value, index in string_literals]
        string_literals = b''.join(struct.pack('>h', len(i)) + i.encode('ascii')
                                   for i in string_literals)

        shared_vars_size = sum(var.type.get_size() for var in shared_vars)
        shared_vars = b'\x00' * shared_vars_size

        return Module({
            Sections.CODE: {
                'addr': Module.INITIAL_ADDR,
                'data': bytecode,
            },
            Sections.STRINGS: {
                'addr': Module.STRING_ADDR,
                'data': string_literals,
            },
            Sections.SHARED: {
                'addr': Module.SHARED_ADDR,
                'data': shared_vars,
            }
        })
