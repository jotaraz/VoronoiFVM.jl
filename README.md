VoronoiFVM.jl
===============

[![Build status](https://github.com/j-fu/VoronoiFVM.jl/workflows/linux-macos-windows/badge.svg)](https://github.com/j-fu/VoronoiFVM.jl/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://j-fu.github.io/VoronoiFVM.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://j-fu.github.io/VoronoiFVM.jl/dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3529808.svg)](https://doi.org/10.5281/zenodo.3529808)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://julialang.zulipchat.com/#narrow/stream/379007-voronoifvm.2Ejl)


Solver for coupled nonlinear partial differential equations (elliptic-parabolic conservation laws) based on the Voronoi finite volume method.
It uses automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl) to evaluate user functions along with their jacobians and calculate derivatives of solutions with respect to their parameters.

## Parallel branch

Idea: Do assembly of the system matrix and the solving of the system of linear equations on multiple threads parallel.

How to: Navigate to the "VoronoiFVM" directory (make sure that the parallel branch of [ExtendableSparse.jl](https://github.com/jotaraz/VoronoiFVM.jl/tree/parallel) is used) and open the Julia REPL with multiple threads, e.g. for 8 threads:

```
$ julia -t 8
julia> include("examples/Example_parallel.jl")
julia> using .Example_parallel
julia> Example_parallel.benchmark_one((100,100,100), 0, 1); # solving a PDE system on a 100x100x100 grid with 1 thread and ILUZero.jl
[ Info: >>> Timestep 1 | Runtime 37.47 | Ass.time 36.53 | Run-Ass 0.9342 | LinSolveTime 0.7191 | Allocs 1771317040
[ Info: >>> Timestep 2 | Runtime 29.37 | Ass.time 28.75 | Run-Ass 0.617 | LinSolveTime 0.5072 | Allocs 1022194464
[ Info: >>> Timestep 3 | Runtime 28.87 | Ass.time 28.25 | Run-Ass 0.6191 | LinSolveTime 0.511 | Allocs 1014160768
julia> Example_parallel.benchmark_one((100,100,100), 8, 3); # solving a PDE system on a 100x100x100 grid with 8 threads and parallel ILU decomposition
[ Info: >>> Timestep 1 | Runtime 84.05 | Ass.time 81.41 | Run-Ass 2.643 | LinSolveTime 2.447 | Allocs 2630868184
[ Info: >>> Timestep 2 | Runtime 65.54 | Ass.time 64.18 | Run-Ass 1.356 | LinSolveTime 1.267 | Allocs 813212576
[ Info: >>> Timestep 3 | Runtime 65.44 | Ass.time 64.19 | Run-Ass 1.249 | LinSolveTime 1.163 | Allocs 813212576
```

## Recent changes
Please look up the list of recent [changes](https://j-fu.github.io/VoronoiFVM.jl/stable/changes)

## Accompanying packages
- [ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl): convenient and efficient sparse matrix assembly
- [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl): unstructured grid management library
- [SimplexGridFactory.jl](https://github.com/j-fu/SimplexGridFactory.jl): unified high level  mesh generator interface
- [Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl):  Julia wrapper for the [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) triangle mesh generator by J. Shewchuk
- [TetGen.jl](https://github.com/JuliaGeometry/TetGen.jl):  Julia wrapper for the [TetGen](http://www.tetgen.org) tetrahedral mesh generator by H. Si.
- [GridVisualize.jl](https://github.com/j-fu/GridVisualize.jl): grid and function visualization related to ExtendableGrids.jl
- [PlutoVista.jl](https://github.com/j-fu/PlutoVista.jl): backend for [GridVisualize.jl](https://github.com/j-fu/GridVisualize.jl) for use in Pluto notebooks.

VoronoiFVM.jl and most of these packages are  part of the meta package [PDELib.jl](https://github.com/WIAS-BERLIN/PDELib.jl).



## Some alternatives
- [ExtendableFEM.jl](https://github.com/chmerdon/ExtendableFEM.jl): finite element library implementing gradient robust FEM
  from the same package base by Ch. Merdon
- [SkeelBerzins.jl](https://github.com/gregoirepourtier/SkeelBerzins.jl): a Julian variation on Matlab's `pdepe` API
- [Trixi.jl](https://github.com/trixi-framework/Trixi.jl):  numerical simulation framework for hyperbolic conservation laws 
- [GridAP.jl](https://github.com/gridap/Gridap.jl) Grid-based approximation of partial differential equations in Julia
- [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) Finite element toolbox for Julia
- [FinEtools.jl](https://github.com/PetrKryslUCSD/FinEtools.jl)  Finite element tools for Julia
- [FiniteVolumeMethod.jl](https://github.com/DanielVandH/FiniteVolumeMethod.jl/) Finite volumes with [Donald boxes](https://sciml.github.io/FiniteVolumeMethod.jl/dev/math/#Control-volumes)

## Some projects and packages using VoronoiFVM.jl

- [RfbScFVM: Performance prediction of flow battery vells](https://github.com/Isomorph-Electrochemical-Cells/RfbScFVM)
- [ChargeTransport.jl: Drift diffusion simulator for semiconductor devices](https://github.com/PatricioFarrell/ChargeTransport.jl)
- [MosLab.jl: From semiconductor to transistor level modeling in Julia](https://github.com/Rapos0/MOSLab.jl)
- [LiquidElectrolytes.jl: Generalized Nernst-Planck-Poisson model for liquid electrolytes](https://github.com/j-fu/LiquidElectrolytes.jl)


## Citation

If you use this package in your work, please cite it according to [CITATION.cff](https://raw.githubusercontent.com/j-fu/VoronoiFVM.jl/master/CITATION.cff)
