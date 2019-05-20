import enum
import struct


class Sections(enum.IntEnum):
    CODE = 1
    STRINGS = 2


class Module:
    file_magic = b'QB'
    version = 1

    # This is the address programs are loaded, right after where
    # conventional memory and upper memory area were back in the bad
    # old days.
    INITIAL_ADDR = 0x100000

    # The address in which string literals will be loaded.
    STRING_ADDR = 0x80000000


    def __init__(self, sections):
        # at the moment only two sections are supported.
        assert set(sections) == {Sections.CODE, Sections.STRINGS}

        self.sections = sections


    def dump(self):
        # file header: magic, version, nsections
        file_hdr = self.file_magic + struct.pack('>BH', self.version, 2)

        # code section header: type, len, load_addr
        code_section = self.sections[Sections.CODE]
        code_section_hdr = struct.pack('>BII',
                                       int(Sections.CODE),
                                       len(code_section['data']),
                                       code_section['addr'])

        # const section header: type, len, load_addr
        str_section = self.sections[Sections.STRINGS]
        total_len = sum(len(i) + 2 for i in str_section['data'])
        str_section_hdr = struct.pack('>BII',
                                      int(Sections.STRINGS),
                                      total_len,
                                      str_section['addr'])

        str_section_data = b''.join(struct.pack('>h', len(i)) + i.encode('ascii')
                                    for i in str_section['data'])

        return file_hdr + \
            code_section_hdr + \
            code_section['data'] + \
            str_section_hdr + \
            str_section_data


    @staticmethod
    def parse(module):
        file_hdr = module[:5]
        magic = file_hdr[:2]
        if magic != Module.file_magic:
            logger.error('Invalid magic. Module is not valid.')
            exit(1)
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
    def create(bytecode, string_literals):
        # the string_literals passed to this function is a dictionary
        # which maps each string to its index in the string
        # section. we just convert it to a list of strings sorted by
        # that index here. the strings will be placed in the proper
        # location later as long as this list is sorted correctly.
        string_literals = list(sorted(string_literals.items(), key=lambda r: r[1]))
        string_literals = [value for value, index in string_literals]

        return Module({
            Sections.CODE: {
                'addr': Module.INITIAL_ADDR,
                'data': bytecode,
            },
            Sections.STRINGS: {
                'addr': Module.STRING_ADDR,
                'data': string_literals,
            },
        })
