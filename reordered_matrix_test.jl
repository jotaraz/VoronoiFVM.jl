dim = parse(Int, ARGS[1]) # dimension of grid
nna = parse(Int, ARGS[2]) # number of nodes per axis
p   = parse(Int, ARGS[3]) # number of threads

parmatvecmul = parse(Bool, ARGS[4])




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


if parmatvecmul
    include("examples/test_solve_matvec_par.jl")
else
    include("examples/test_solve_matvec_seq.jl")
end

include("examples/test_solvers.jl")
include("examples/writing.jl")

precon_names = ["ILUZero", "ILUAM", "PILUAM"]


for precon=1:3
    @info precon
    x = zeros(Float64, (3, 5))

    @info "BLASnt = 1"
    BLAS.set_num_threads(1)
    bm_now(nm, 0.01, p; precon_linear_id=precon)
    x[1,:] = bm_now(nm, 0.01, p; num=20, precon_linear_id=precon)

    @info "BLASnt = $(Int(round(p/2))))"
    BLAS.set_num_threads(Int(round(p/2)))
    bm_now(nm, 0.01, p; precon_linear_id=precon)
    x[2,:] = bm_now(nm, 0.01, p; num=20, precon_linear_id=precon)

    @info "BLASnt = $p"
    BLAS.set_num_threads(p)
    bm_now(nm, 0.01, p; precon_linear_id=precon)
    x[3,:] = bm_now(nm, 0.01, p; num=20, precon_linear_id=precon)


    head = "dim=$dim, nna=$nna, nt=$p, REORDERED, parmatvecmul=$parmatvecmul, "*precon_names[precon]*", $(Dates.now())"
    write_to(x, "linsolve_test_data.txt", head)
end
