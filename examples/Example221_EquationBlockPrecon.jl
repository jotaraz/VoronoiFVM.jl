#=

# 221: Equation block preconditioning
([source code](@__SOURCE_URL__))

=#

module Example221_EquationBlockPrecon

using VoronoiFVM, GridVisualize, ExtendableGrids, Printf
using AMGCLWrap, ExtendableSparse
using Test

function main(;dim=1, nref=0, Plotter = nothing, plot_grid = false, verbose = false,
              unknown_storage = :sparse, assembly = :edgewise,strategy = nothing)
    
    nx=30*2^nref+1
    ny=9*2^nref
    
    X = range(0,3,length=nx)
    Y = range(-0.5,0.5,length=ny)
    if dim==1
        grid = VoronoiFVM.Grid(X)
        Γ_in=1
        Γ_out=2
    elseif dim==2
        grid = VoronoiFVM.Grid(X,Y)
        Γ_in=4
        Γ_out=2
    else
        grid = VoronoiFVM.Grid(X,Y,Y)
        Γ_in=4
        Γ_out=2
    end
    cellmask!(grid, [0.0,-0.5,-0.5], [1.0,0.5,0.5], 1)
    cellmask!(grid, [1.0,-0.5,-0.5], [2.0,0.5,0.5], 2)
    cellmask!(grid, [2.0,-0.5,-0.5], [3.0,0.5,0.5], 3)

    
    subgrid1 = subgrid(grid, [1])
    subgrid2 = subgrid(grid, [1, 2, 3])
    subgrid3 = subgrid(grid, [3])
        
    if plot_grid
        return gridplot(grid; Plotter = Plotter)
    end

    eps = [1, 1, 1]
    k = [1, 1, 1]

    function reaction(f, u, node)
        if node.region == 1
            f[1] = k[1] * u[1]
            f[2] = -k[1] * u[1]
        elseif node.region == 3
            f[2] = k[3] * u[2]
            f[3] = -k[3] * u[2]
        else
            f[1] = 0
        end
    end

    function source(f, node)
        if node.region == 1
            f[1] = 1.0e-4 * (3.0 - node[1])
        end
    end

    flux = function (f, u, edge)
        if edge.region == 1
            f[1] = eps[1] * (u[1, 1] - u[1, 2])
            f[2] = eps[2] * (u[2, 1] - u[2, 2])
        elseif edge.region == 2
            f[2] = eps[2] * (u[2, 1] - u[2, 2])
        elseif edge.region == 3
            f[2] = eps[2] * (u[2, 1] - u[2, 2])
            f[3] = eps[3] * (u[3, 1] - u[3, 2])
        end
    end
    
    storage = function (f, u, node)
        if node.region == 1
            f[1] = u[1]
            f[2] = u[2]
        elseif node.region == 2
            f[2] = u[2]
        elseif node.region == 3
            f[2] = u[2]
            f[3] = u[3]
        end
    end

    sys = VoronoiFVM.System(grid; flux, reaction, storage, source,
                            unknown_storage , assembly, is_linear=true)

    enable_species!(sys, 1, [1])
    enable_species!(sys, 2, [1, 2, 3])
    enable_species!(sys, 3, [3])

    boundary_dirichlet!(sys, 3, 2, 0.0)

    control = SolverControl(strategy, sys, verbose="l")
    U=solve(sys;control)
    @info num_dof(U)

    p = GridVisualizer(; Plotter = Plotter, layout = (3,1),
                       limits = (0, 1e-3),
                       xlimits=(0,3))
    
    U1 = view(U[1, :], subgrid1)
    U2 = view(U[2, :], subgrid2)
    U3 = view(U[3, :], subgrid3)

    scalarplot!(p[1, 1], subgrid1, U1; title = "spec1", color = (0.5, 0, 0))
    scalarplot!(p[2, 1], subgrid2, U2; title = "spec2", color = (0.0, 0.5, 0))
    scalarplot!(p[3, 1], subgrid3, U3; title = "spec3", color = (0.0, 0.0, 0.5))
    reveal(p)
    U
end

function runtests()
    strategy=BICGstabIteration(AMGCL_AMGPreconditioner())
    @test sum(main(;dim=1,strategy, unknown_storage=:dense)[2,:]) ≈ 0.014100861021046823
    @test sum(main(;dim=1,strategy, unknown_storage=:sparse)[2,:]) ≈ 0.014100861021046823
    @test sum(main(;dim=2,strategy, unknown_storage=:dense)[2,:]) ≈ 0.12690774918944653
    @test sum(main(;dim=2,strategy, unknown_storage=:sparse)[2,:]) ≈ 0.12690774918944653
    @test sum(main(;dim=3,strategy, unknown_storage=:dense)[2,:]) ≈ 1.1423244466533444
    @test sum(main(;dim=3,strategy, unknown_storage=:sparse)[2,:]) ≈ 1.1423244466533444
end

end
