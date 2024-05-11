### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ dd05e525-9a68-47c0-b8cb-755ad7c6e1ae
import Pkg

# ╔═╡ b8205a3a-9563-4e20-89e0-c7736635985e
Pkg.activate("/home/johannes/.julia/dev/VoronoiFVM")

# ╔═╡ 27572a68-0ffc-4b56-b2ff-1fcab97f09a6
Pkg.add("ChunkSplitters");using ChunkSplitters

# ╔═╡ 74985dbc-907e-4a51-85d7-d5548f6f8426
Pkg.add("ColorSchemes"); using ColorSchemes

# ╔═╡ 15972153-9660-4240-a60f-eddede376f92
using VoronoiFVM

# ╔═╡ fae481c9-1976-4992-bb10-26070c73e9a6
begin
	include("/home/johannes/.julia/dev/VoronoiFVM/examples/Example_parallel.jl")
	using .Example_parallel
end

# ╔═╡ 86f7e9e0-65f8-4c62-a685-31674e0012a8
using GLMakie

# ╔═╡ a205dcd8-0eaf-11ef-37da-8da7f1c0e4e4
function f(a, x, T)
	return a*(exp(x/T) - exp(-x/T))
end

# ╔═╡ 92f68732-1d89-4919-98a9-ac8e2aac64ba
f(1,1,1)

# ╔═╡ 3559fcb3-6e7a-4acd-af28-df180c5be251
Example_parallel.benchmark_one((30,30,30), 1, 1)

# ╔═╡ c76d1397-c4d1-4c95-9f83-8e29e926f46c
Example_parallel.benchmark_one((30,30,30), 4, 1)

# ╔═╡ df2373e5-a7df-42e8-835f-383e151c6d12
Example_parallel.benchmark_one((30,30,30), 4, 3)

# ╔═╡ ce96d29e-f9a9-4de1-ba46-fcc6fce567ad
md"""
# What was the idea?

Next to the 'ExtendableSparseMatrix' (ESM) structure we introduced a new structure, called 'ExtendableSparseMatrixParallel' (ESMP).

While the ESM consists of one CSC (compressed sparse column, see below) matrix and one LNK (linked list, see below) matrix, the ESMP consists of one CSC matrix and $n$ LNK matrices, where $n$ is the number of threads that run parallel. 
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
Thus, we partition the grid into $n$ regions and a 'separator'.
The regions are are chosen such that a cell in region $i$ is only adjacent to other cells of region $i$ or the separator.
Thus, the computations for cells of different reasons do not interfere and can be done concurrently. After all regions are finished, the cells of the separator are computed.
Hence, each region is identified with a thread.

In our specific case, it works like this:

### Using an ExtendableSparseMatrixParallel

Partition the grid into $n$ regions and a separator.
Then assign a thread to each region.
Each thread iterates over its cells and writes the computed numbers in it's own LNK matrix.
Then thread 1 also computes the separator cells.

Then the $n$ LNK matrices are jointly converted into one CSC matrix.

As in the ESM case, the values can easily be changed in the CSC matrix.
When using the same access pattern (each thread works on its region, and then the separator), this can be done in parallel as well.

## CSC & LNK

...


## SuperSparseMatrixLNK

Actually, ESMP does not have matrices of type SparseMatrixLNK as LNK, but matrices of type SuperSparseMatrixLNK. SuperSparse means 'even more sparse'.
Sepcifically we want to be able to efficiently access the entries even if some columns are entry. 
To this end, each SuperSparseMatrixLNK has an array 'collnk', such that 'collnk[i] = j' if $j$ is the $i$-th non-zero column of the matrix.


"""

# ╔═╡ fff67b93-681b-4add-8aed-144fa0c98b42
function part2d(X,Y, nt)
    nt=max(4,nt)
    XP=collect(chunks(1:length(X)-1,n=nt))
    YP=collect(chunks(1:length(Y)-1,n=nt))
    partitions = [Tuple{StepRange{Int64}, StepRange{Int64}}[] for i = 1:nt]
    ipart=1
    col=1
    for jp=1:nt
        for ip=1:nt
            push!(partitions[col], (XP[ip], YP[jp]))
            col=(col -1 +1 )%nt+1
        end
        col=(col -1 +2)%nt+1
    end
    partitions
end

# ╔═╡ 03b7d8d4-2515-4ebd-ba45-07a8b2e003c6
partitions = [Tuple{StepRange{Int64}, StepRange{Int64}}[] for i = 1:4]

# ╔═╡ d058e686-f690-4160-8473-85032753e158
collect(chunks(1:10,n=3))

# ╔═╡ bd76e2c1-0076-4a4a-addf-c628e4b98d60
let
	X = collect(range(1,10))
	Y = collect(range(1,10))
	part2d(X, Y, 5)
end

# ╔═╡ b9c8e81d-402f-4646-a70b-1d9da3d4d0ff
function showgrid(Makie, ColorSchemes, X,Y,nt)
    f = Makie.Figure()
    ax = Makie.Axis(f[1, 1]; aspect = 1)
    p=part2d(X,Y,nt)
    ncol=length(p)
    @show sum(length,p), ncol
    colors=get(ColorSchemes.rainbow,collect(1:ncol)/ncol)
    poly=Vector{Makie.Point2f}(undef,4)
    for icol = 1:ncol
        for (xp, yp) in p[icol]
            for j in yp
                for i in xp
                    poly[1]=Makie.Point2f(X[i], Y[j])
                    poly[2]=Makie.Point2f(X[i + 1], Y[j])
                    poly[3]=Makie.Point2f(X[i + 1], Y[j + 1])
                    poly[4]=Makie.Point2f(X[i], Y[j + 1])
                    Makie.poly!(copy(poly),color = colors[icol])
                end
            end
        end
    end
    f
end

# ╔═╡ d15d52c7-805e-4eae-ade8-d10eb671bf7f
let
	X = collect(range(1,10))
	Y = collect(range(1,10))
	part2d(X, Y, 5)
	showgrid(GLMakie, ColorSchemes, X, Y, 5)
end

# ╔═╡ 5086fe60-6a04-4275-8409-2472db0296d7


# ╔═╡ Cell order:
# ╠═a205dcd8-0eaf-11ef-37da-8da7f1c0e4e4
# ╠═92f68732-1d89-4919-98a9-ac8e2aac64ba
# ╠═dd05e525-9a68-47c0-b8cb-755ad7c6e1ae
# ╠═27572a68-0ffc-4b56-b2ff-1fcab97f09a6
# ╠═b8205a3a-9563-4e20-89e0-c7736635985e
# ╠═15972153-9660-4240-a60f-eddede376f92
# ╠═fae481c9-1976-4992-bb10-26070c73e9a6
# ╠═3559fcb3-6e7a-4acd-af28-df180c5be251
# ╠═c76d1397-c4d1-4c95-9f83-8e29e926f46c
# ╠═df2373e5-a7df-42e8-835f-383e151c6d12
# ╠═ce96d29e-f9a9-4de1-ba46-fcc6fce567ad
# ╠═fff67b93-681b-4add-8aed-144fa0c98b42
# ╠═03b7d8d4-2515-4ebd-ba45-07a8b2e003c6
# ╠═d058e686-f690-4160-8473-85032753e158
# ╟─bd76e2c1-0076-4a4a-addf-c628e4b98d60
# ╠═d15d52c7-805e-4eae-ade8-d10eb671bf7f
# ╠═74985dbc-907e-4a51-85d7-d5548f6f8426
# ╠═86f7e9e0-65f8-4c62-a685-31674e0012a8
# ╠═b9c8e81d-402f-4646-a70b-1d9da3d4d0ff
# ╠═5086fe60-6a04-4275-8409-2472db0296d7
