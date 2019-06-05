class InvalidPointer(Exception):
    def __init__(self, ptr):
        self.ptr = ptr
        super().__init__(f'Invalid pointer: {ptr:x}')


class Allocator:
    """This is an abstract memory allocator. It doesn't really allocate
any memory, but only keeps track of the free and allocated slots. It
is simply given a number of free slots on which it can perform malloc
and free operations.

    """
    def __init__(self, free_slots):
        """`free_slots` is a list of (base, size) tuples, in which base is the
beginning address of the block and size is the block size. These
blocks should not be overlapping.

        """
        assert all(isinstance(i, tuple) and len(i) == 2
                   for i in free_slots)

        self.free_slots = free_slots
        self.alloced_slots = []


    def malloc(self, size):
        for i, (slot_base, slot_size) in enumerate(self.free_slots):
            if slot_size >= size:
                break
        else:
            raise MemoryError

        remaining = slot_size - size
        if remaining > 0:
            self.free_slots[i] = (slot_base + size, remaining)
        else:
            del self.free_slots[i]

        self.alloced_slots.append((slot_base, size))

        return slot_base


    def free(self, ptr):
        try:
            idx = [b for b, s in self.alloced_slots].index(ptr)
        except ValueError:
            raise InvalidPointer(ptr)

        _, size = self.alloced_slots[idx]
        del self.alloced_slots[idx]

        for i, (slot_base, slot_size) in enumerate(self.free_slots):
            if ptr < slot_base + slot_size:
                break
        else:
            i = len(self.free_slots)

        self.free_slots.insert(i, (ptr, size))

        if i > 0:
            prev_ptr, prev_size = self.free_slots[i - 1]
            if prev_ptr + prev_size == ptr:
                # merge with previous free slot
                self.free_slots[i] = (prev_ptr, prev_size + size)
                del self.free_slots[i-1]
                ptr = prev_ptr
                size += prev_size
                i -= 1

        if i < len(self.free_slots) - 1:
            next_ptr, next_size = self.free_slots[i + 1]
            if ptr + size == next_ptr:
                # merge with next free slot
                self.free_slots[i] = (ptr, size + next_size)
                del self.free_slots[i+1]


    def print_stats(self):
        print(f'-- Allocator ------------------------------')
        print(f'| Alloc\'d slots: {self.alloced_slots}')
        print(f'| Free slots:     {self.free_slots}')
        print(f'-------------------------------------------')
