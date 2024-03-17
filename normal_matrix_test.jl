dim = parse(Int, ARGS[1]) # dimension of grid
nna = parse(Int, ARGS[2]) # number of nodes per axis
p   = parse(Int, ARGS[3]) # number of threads

if dim == 3
    nm = (nna,nna,nna)
else
    nm = (nna, nna)
end

import Pkg
Pkg.activate(".")
using Dates
using VoronoiFVM
using LinearAlgebra



include("examples/test_solvers.jl")
include("examples/writing.jl")


x = zeros(Float64, (3, 5))

@info "BLASnt = 1"
BLAS.set_num_threads(1)
bm_now(nm, 0.01, 0)
x[1,:] = bm_now(nm, 0.01, 0; num=20)

@info "BLASnt = $(Int(round(p/2)))"
BLAS.set_num_threads(Int(round(p/2)))
bm_now(nm, 0.01, 0)
x[2,:] = bm_now(nm, 0.01, 0; num=20)

@info "BLASnt = $p"
BLAS.set_num_threads(p)
bm_now(nm, 0.01, 0)
x[3,:] = bm_now(nm, 0.01, 0; num=20)

head = "dim=$dim, nna=$nna, nt=$p, NORMAL, -, ILUZero, $(Dates.now())"
#head = "NORMAL $dim $nna $p $(Dates.now())"
write_to(x, "linsolve_test_data.txt", head)
