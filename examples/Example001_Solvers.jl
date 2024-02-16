#=
# 001: New linear solver API
([source code](@__SOURCE_URL__))
=#

module Example001_Solvers

## under development

using Printf
using VoronoiFVM
using ExtendableGrids
using GridVisualize
using LinearSolve
using ExtendableSparse
using AMGCLWrap
using AlgebraicMultigrid
using LinearAlgebra
using Test

function main(; n = 10, Plotter = nothing, assembly = :edgwwise, kwargs...)
    h = 1.0 / convert(Float64, n)
    X = collect(0.0:h:1.0)
    Y = collect(0.0:h:1.0)

    grid = VoronoiFVM.Grid(X, Y)
    nn = num_nodes(grid)

    eps = 1.0e-2

    function reaction(f, u, node)
        f[1] = u[1]^2
    end

    function flux(f, u, edge)
        f[1] = eps * (u[1, 1]^2 - u[1, 2]^2)
    end

    function source(f, node)
        x1 = node[1] - 0.5
        x2 = node[2] - 0.5
        f[1] = exp(-20.0 * (x1^2 + x2^2))
    end

    function storage(f, u, node)
        f[1] = u[1]
    end

    function bcondition(f, u, node)
        boundary_dirichlet!(f,
                            u,
                            node;
                            species = 1,
                            region = 2,
                            value = ramp(node.time; dt = (0, 0.1), du = (0, 1)))
        boundary_dirichlet!(f,
                            u,
                            node;
                            species = 1,
                            region = 4,
                            value = ramp(node.time; dt = (0, 0.1), du = (0, 1)))
    end

    sys = VoronoiFVM.System(grid; reaction, flux, source, storage, bcondition, assembly,
                            species = [1])
    @info "UMFPACK:"
    umf_sol = solve(sys; inival = 0.5, method_linear = UMFPACKFactorization(), kwargs...)

    @info "KLU:"
    sol = solve(sys; inival = 0.5, method_linear = KLUFactorization(), kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7
    
    @info "Sparspak:"
    sol = solve(sys; inival = 0.5, method_linear = SparspakFactorization(), kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7

    @info "Krylov-ilu0:"
    sol = solve(sys;
                inival = 0.5,
                method_linear = KrylovJL_BICGSTAB(),
                precon_linear = ILUZeroPreconditioner(),
                kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7

    @info "Krylov-block1"
    sol = solve(sys;
                inival = 0.5,
                method_linear = KrylovJL_BICGSTAB(),
                precon_linear = BlockPreconditioner(; partitioning = [1:(nn ÷ 2), (nn ÷ 2 + 1):nn],
                                                    factorization = ILU0Preconditioner()),
                      kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7

    @info "Krylov-block2"
    sol = solve(sys;
                inival = 0.5,
                method_linear = KrylovJL_BICGSTAB(),
                precon_linear = BlockPreconditioner(; partitioning = [1:2:nn, 2:2:nn],
                                                    factorization = UMFPACKFactorization()),
                      kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7

    @info "Krylov - delayed factorization:"
    sol = solve(sys;
                inival = 0.5,
                method_linear = KrylovJL_BICGSTAB(),
                precon_linear = SparspakFactorization(),
                keepcurrent_linear =false,
                kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7

    @info "Krylov - jacobi:"
    sol = solve(sys;
                inival = 0.5,
                method_linear = KrylovJL_BICGSTAB(),
                precon_linear = JacobiPreconditioner(),
                keepcurrent_linear = true,
                kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7
    
    @info "Krylov - SA_AMG:"
    sol = solve(sys;
                inival = 0.5,
                method_linear = KrylovJL_BICGSTAB(),
                precon_linear = SA_AMGPreconditioner(),
                keepcurrent_linear = true,
                kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7

    @info "Krylov - AMGCL_AMG:"
    sol = solve(sys;
                inival = 0.5,
                method_linear = KrylovJL_BICGSTAB(),
                precon_linear = AMGCL_AMGPreconditioner(),
                keepcurrent_linear = true,
                kwargs...)
    @test norm(sol - umf_sol, Inf)<1.0e-7

end

function runtests()
    @testset "edgewise" begin
        main(; assembly = :edgewise)
    end
    @testset "cellwise" begin
        main(; assembly = :cellwise)
    end
end
end
