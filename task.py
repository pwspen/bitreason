import hashlib
from typing import Callable

class Task:
    def __init__(
        self,
        transform: Callable[[list[int]], list[int]],
        templates: list[list[int | None]] | None = None,
        name: str | None = None
        # Add way to stack transformations
    ):
        self.template_len = 16

        if templates is None:
            templates = [[None] * self.template_len]

        for template in templates:
            if len(template) != self.template_len:
                raise ValueError(f"All templates must be of length {self.template_len}")
        self.templates = templates
        self.transform = transform

        # Extract name from function if not provided
        # Important that transform is not a lambda because
        # name is used in seed for randomness generation
        self.name = name or transform.__name__
        if "<lambda>" in self.name:
            raise ValueError("Transform function must not be a lambda if name is not provided")
    
    def get_random_bits(self, n: int, seed: str) -> list[int]:
        # Hash the seed + index to get deterministic randomness
        hash_input = seed.encode()
        hash_value = hashlib.sha256(hash_input).digest()
        # Convert to float between 0 and 1
        result = [x & 1 for x in hash_value[0:n]]
        return result
    
    def generate(self, example: bool, index: int) -> list[int]:
        # Grab enough random bits to select template and fill it in
        # template_len bits for each is enough for filling in and for
        # up to 2**template_len templates
        seed = f"{self.name}:{int(example)}:{str(index)}"
        bits = self.get_random_bits(self.template_len*2, seed=seed)
        template_select, bits = bits[:self.template_len], bits[self.template_len:]

        if len(self.templates) <= 2:
            # Convert bit string to int and modulo to get idx
            bitstr = "".join(str(b) for b in template_select)
            template_idx = int(bitstr, 2) % len(self.templates)
            template = self.templates[template_idx].copy()
        else:
            template = self.templates[0].copy()

        # Fill in template
        none_indices = [i for i, x in enumerate(template) if x is None]
        for i, bit in zip(none_indices, bits):
            template[i] = bit
        
        return template # list[int]
    
    def get_pair(self, example: bool, index: int) -> tuple[list[int], list[int]]:
        inp = self.generate(example, index)
        out = self.transform(inp)
        return inp, out