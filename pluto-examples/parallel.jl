### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ b8205a3a-9563-4e20-89e0-c7736635985e
import Pkg; Pkg.activate("/home/johannes/.julia/dev/VoronoiFVM"); Pkg.instantiate()

# ╔═╡ ee017e27-866b-436c-88f2-5ff9ffd7c315
begin
	using ChunkSplitters
	using VoronoiFVM
	using PyPlot
	using GridVisualize
	using ExtendableGrids
end

# ╔═╡ a4ba1fea-56dd-4204-b392-bd45b81bdf55
include("/home/johannes/.julia/dev/VoronoiFVM/examples/Example_parallel.jl"); using .Example_parallel

# ╔═╡ ffb99281-e637-40f1-a4cb-6c0ccab720ff
md"""
## Benchmarks:

The total time (`Runtime`), the assembly time (`Ass.time`), the remaining time (`Run-Ass` = Runtime-Assembly time) and the time to solve the system of linear equations (`LinSolveTime`) are shown, all in seconds.
The allocations (`@allocated`) in bytes of each timestep are also shown. \
Comment: `Example_parallel.benchmark_one(nm, nt, precon_id)` with `nm` = (nx, ny) or (nx, ny, nz). 
`nt` = number of threads, if nt=1, standard VoronoiFVM with standard ExtendableSparse is used. \
`precon_id` = 1 (ILUZero.jl), = 2 ([ILUAM](https://doi.org/10.1016/S0898-1221(03)00154-8)), = 3 (parallel [ILUAM](https://doi.org/10.1016/S0898-1221(03)00154-8)).\
\

Solving 3 timesteps of a PDE on a 35x35x35 grid on 4 threads using a parallel ILU preconditioner.
"""

# ╔═╡ df2373e5-a7df-42e8-835f-383e151c6d12
Example_parallel.benchmark_one((35,35,35), 4, 3);

# ╔═╡ b4d144d0-ec4f-4c44-8785-183b9df7cf37
md"""
Solving the same system using ILUZero.jl as preconditioner (and still 4 threads)
"""

# ╔═╡ c76d1397-c4d1-4c95-9f83-8e29e926f46c
Example_parallel.benchmark_one((35,35,35), 4, 1);

# ╔═╡ e9d2afa4-8443-4b94-8be6-e7630d710beb
md"""
Solving the same system using 1 thread and ILUZero.jl
"""

# ╔═╡ 3559fcb3-6e7a-4acd-af28-df180c5be251
Example_parallel.benchmark_one((35,35,35), 1, 1);

# ╔═╡ 25a8466a-1456-478e-9b79-56619d81cb0b
md"""
### Validation:

Solves the same PDE and compares the solution vector for 1 and 4 threads, edge- and cellwise assembly and ILUZero.jl and parallel ILU.
"""

# ╔═╡ 9e45b933-c5ff-4306-8c62-a377f101d7c1
Example_parallel.test(; nm=(100,100), do_print_ts=false)

# ╔═╡ ce96d29e-f9a9-4de1-ba46-fcc6fce567ad
md"""
# What was the idea?

Next to the 'ExtendableSparseMatrix' (ESM) structure we introduced a new structure, called 'ExtendableSparseMatrixParallel' (ESMP).

While the ESM consists of one CSC (compressed sparse column) matrix and one LNK (linked list) matrix, the ESMP consists of one CSC matrix and $n$ LNK matrices, where $n$ is the number of threads that run parallel. 
But the ESMP is more than that, it also contains the information which thread accesses which column etc.
The structure is supposed to be used for solving partial differential equations (PDEs) numerically.
When solving a PDE on a grid using the Finite Volume Method (FVM), a system of linear equations $Ax=b$ needs to be solved. $A$ can be viewed as the discretized version of the PDE.
If the grid has $m$ nodes, $A$ is an $m \times m$ matrix, the diagonal entries correspond to nodes and the non-diagonal entries correspond to edges between two nodes.
Assembling this matrix, $A$, is computationally intensive.
It can be done by iterating over each cell of the grid, and computing some values for each node and each edge in a cell and storing them in the corresponding matrix entries.

## Filling the matrix

### Using an ExtendableSparseMatrix

When assembling $A$ using the ESM structure, you iterate over all cells and first you write each entry into the LNK matrix. 
Insertion of new entries is very fast for the LNK matrix, but the usage of the matrix is slow, thus, once you are done, you flush all entries to the CSC matrix. Here insertion is slow and usage is fast.

Often, you want to change the values of the non-zero entries of the matrix later (e.g. when solving a non-linear PDE), but you do not want to change where the non-zero entries are. This can be done quickly by changing the non-zero values of the CSC matrix.

New values can be inserted using the LNK matrix and then flushing again.

### Can we just compute the matrix entries in parallel?

The computations done for each cell could be done in parallel.
But since two adjacent cells share at least one node, both computations would alter the matrix entry of this node. Thus, the storage can not be done trivially parallel.
Thus, we partition the grid into $n$ regions (i.e. sets of cells) and a 'separator' (a set of cells).
The regions are chosen such that a cell in region $i$ is only adjacent to other cells of region $i$ or the separator.
Thus, the computations for cells of different reasons do not interfere and can be done concurrently. After all regions are finished, the cells of the separator are computed.
Hence, each region is identified with a thread.

Here you can see a 2d triangular grid separated in 4 regions (numbers 1-4) and a separator (number 5).
The colors indicate the associated colors (or their numbers).
"""

# ╔═╡ dc3ab392-3661-4674-9c1a-92007d1370bd
begin
	grid = VoronoiFVM.getgrid((20,20))
	nnts, s, onr, cfp, gi, ni, rni, starts, cellparts, adepth = VoronoiFVM.ExtendableSparse.preparatory_multi_ps_less_reverse(grid[CellNodes], num_cells(grid), num_nodes(grid), 4, 1, Int64)
	grid[CellRegions] = cellparts
	#gridplot(grid, Plotter=PyPlot)
	vis=GridVisualizer(Plotter=PyPlot, layout=(1,1))
	gridplot!(vis[1,1],grid)
	reveal(vis)
end

# ╔═╡ 246a50b6-c969-434f-98d5-b0361facc434
md"""
Hence, thread 1 would calculate the matrix entries for all red triangles.
At the same time, thread 2 calculates the matrix entries for all green triangles.
At the same time, thread 3 calculates the matrix entries for all purple triangles.
At the same time, thread 4 calculates the matrix entries for all blue triangles.
After they are all done, thread one calculates the matrix entries for the yellow triangles.


In our specific case, the assembly works like this:

### Using an ExtendableSparseMatrixParallel

Partition the grid into $n$ regions and a separator.
Then assign a thread to each region.
The grid is partitioned in a way such that each node is accessed by only one thread (and maybe the separator).
Thus, we can pre-compute the list of nodes associated with a thread, the $i$-th thread has `nnts[i]` nodes.
We create the LNK matrices of the individual threads before the calculations happen.
The $i$-th LNK matrix has `#nodes` rows and `nnts[i]` columns.

Then, in the loop each thread iterates over its cells and writes the computed numbers in it's own LNK matrix.
Then thread 1 also computes the separator cells and writes the results in its own LNK matrix.

Then the $n$ LNK matrices are jointly converted into one CSC matrix (this is called `flush`). The flushing is described in more detail below.

As in the ESM case, the values can easily be changed in the CSC matrix.
When using the same access pattern (each thread works on its region, and then the separator), this can be done in parallel as well.



## SuperSparseMatrixLNK

Actually, ESMP does not have matrices of type SparseMatrixLNK as LNK, but matrices of type SuperSparseMatrixLNK. SuperSparse means 'even more sparse'.
Sepcifically we want to be able to efficiently access the entries even if some columns are entry. 
To this end, each SuperSparseMatrixLNK has an array 'collnk', such that 'collnk[i] = j' if $j$ is the $i$-th non-zero column of the matrix.


The code is in `src/matrix/ExtendableSparseParallel/supersparse.jl`.

## flushing

Flushing can be computationally intensive.
We have implemented two ways of flushing 'dense' and 'sparse'.
If the matrix is built up from the ground and the LNK matrices are converted to CSC, 'dense' is used. If the LNK matrices only contain a few new entries, 'sparse' is used.

The code is in `src/matrix/ExtendableSparseParallel/struct_flush.jl`.


### 'dense'

Since we have a rigid node (i.e. column) to thread(s) map, we now which LNK matrices can contain non-zeros entries for a specific column.
We iterate over all columns, and for each column, we check if the possible LNK matrices have non-zero entries, if yes, their entries are correctly inserted into the arrays that form the CSC matrix (`colptr`, `rowval`, `nzval`).

### 'sparse'

Here we loop over the LNK matrices.
For each LNK matrix, we loop over the columns containing non-zero entries.
Here, we really make use of the SuperSparseMatrixLNK structure.

## preparatory

The partitioning of the grid, using Metis.jl and computations such as 'which thread accesses which node?' and 'which cells are accessed by thred $i$?' are done in `src/matrix/ExtendableSparseParallel/preparatory.jl`.

"""

# ╔═╡ 03b7d8d4-2515-4ebd-ba45-07a8b2e003c6


# ╔═╡ Cell order:
# ╠═b8205a3a-9563-4e20-89e0-c7736635985e
# ╠═ee017e27-866b-436c-88f2-5ff9ffd7c315
# ╠═a4ba1fea-56dd-4204-b392-bd45b81bdf55
# ╠═ffb99281-e637-40f1-a4cb-6c0ccab720ff
# ╠═df2373e5-a7df-42e8-835f-383e151c6d12
# ╟─b4d144d0-ec4f-4c44-8785-183b9df7cf37
# ╠═c76d1397-c4d1-4c95-9f83-8e29e926f46c
# ╟─e9d2afa4-8443-4b94-8be6-e7630d710beb
# ╠═3559fcb3-6e7a-4acd-af28-df180c5be251
# ╠═25a8466a-1456-478e-9b79-56619d81cb0b
# ╠═9e45b933-c5ff-4306-8c62-a377f101d7c1
# ╟─ce96d29e-f9a9-4de1-ba46-fcc6fce567ad
# ╟─dc3ab392-3661-4674-9c1a-92007d1370bd
# ╟─246a50b6-c969-434f-98d5-b0361facc434
# ╠═03b7d8d4-2515-4ebd-ba45-07a8b2e003c6
