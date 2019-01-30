# Details
Script with example 8.4 and function to create the impedance bus matrix of a power system from an networkx simple graph. This uses the four modification cases from Chapter 8 - "Power System Analysis" W. Stevenson et al.. The script also contains the function for Dr. Gabriel Kron's node reduction algorithm to generate an equivalent electric power network (Chapter 7 Stevenson's book).

**Note:** Input graph requires a reference node labeled -1.

# Todo
*Legend: ☑ (Done) | ☒ (Won't be done for now) | □ (Will be done soon)*

- ☑ Function to create Zbus from scrath depending on edges.
- ☑ Add node index list (from input graph) that represent each row/column from Zbus.
- ☑ Function with Gabriel kron's reduction algorithm.
- ☑ Function to reorder Zbus's rows and columns according to input list.
- ☑ Adapt algorithm to work with MultiGraph instead of simple graph (NetworkX).
- □ Add support for coupled branches (all cases).
- ☒ Ybus creation from scratch function.
