This is kinda like ARC but with some changes to make it easier to try new things.
- One color: Only 0's and 1's
- One dimension: Inputs and outputs are just bitstrings
- Fixed length: Outputs always the same length as inputs (16 bits)

So tasks are MUCH simpler, and you can use architectures that have fixed IO like plain old neural nets.

There's also a big focus on automatically generating examples instead of having a human generate them manually. You define a transformation rule and some templates - where you specify which bits should be fixed (if any) and which should vary - and from there, you can generate as many training or test pairs as you like.

This means, unlike ARC, you have the entire task-space available for each task, so you can train on more data as needed without having to go find some dataset of ARC-like tasks.

ARC is great, but it's pretty complicated. Tasks have varying sizes. There are a bunch of colors that you have to decide how to deal with symmetrically. And tasks are defined manually by humans which means there's not very many of them.

What if you want to experiment with simple fixed-IO architectures? 

ML in general is very "throw things at wall, keep what sticks" + "if it ain't broke don't fix it". 

The goal here is to provide a playground to build intuition and evidence for how ARC-like reasoning works.

If you make a model for ARC and it works well it might be pretty opaque. What parts of it contribute to its success? The goal here is to give you ways to examine that.

