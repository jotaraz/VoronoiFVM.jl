
#https://github.com/j-fu/VoronoiFVM.jl/blob/7039d768ba26e80dbb1aa4fabcce5993486561a4/examples/Example207_NonlinearPoisson2D.jl

using PyPlot, GridVisualize, ExtendableGrids, LinearSolve

#2d
function reaction2(f, u, node)
	f[1] = u[1]^2
end
function flux2(f, u, edge)
	f[1] = 1e-2 * (u[1, 1]^2 - u[1, 2]^2)
end
function source2(f, node)
	x1 = node[1] - 0.5
	x2 = node[2] - 0.5
    f[1] = exp(-20.0 * (x1^2 + x2^2))
end
function storage2(f, u, node)
	f[1] = u[1]
end


# 3d
function reaction3(f, u, node)
	f[1] = u[1]^2
end
function flux3(f, u, edge)
	f[1] = u[1, 1]^2 - u[1, 2]^2 #1e-2 * (u[1, 1]^2 - u[1, 2]^2)
end
function source3(f, node)
	x1 = node[1] - 0.5
	x2 = node[2] - 0.5
	x3 = node[3] - 0.5
	f[1] = x1 * sin(5.0 * x2) * exp(x3)
end
function storage3(f, u, node)
	f[1] = u[1]
end

function example_time_normal(nm, dt, tpoints; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, do_plot = true, do_print_allocs=1, do_print_eaa=false, do_print_ts=true)
    dim = length(nm)
    grid = VoronoiFVM.ExtendableSparse.getgrid(nm)
    if dim == 2
        physics = VoronoiFVM.Physics(; reaction = reaction2,
                                    flux = flux2,
                                    source = source2, 
                                    storage = storage2)
    else
        physics = VoronoiFVM.Physics(; reaction = reaction3,
                                    flux = flux3,
                                    source = source3, 
                                    storage = storage3)
    end

	
    sys = VoronoiFVM.System(grid, physics; unknown_storage, assembly = assembly)
    enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    oldsol = unknowns(sys)
    oldsol .= 0.5
    solution = unknowns(sys)
    control = VoronoiFVM.NewtonControl()
    control.verbose = verbose
    control.reltol_linear = 1.0e-5
    control.method_linear = method_linear
    control.precon_linear = precon_linear
    control.log = true
    
    time = 0.0
    empedparam = 0.0
    params = zeros(0)
    
    sols = [zeros(num_nodes(grid)) for i=1:tpoints+1]
    sols[1] = copy(oldsol[1,:])
    
    if do_plot
    	p = GridVisualizer(;Plotter=PyPlot)
    end
    for i=1:tpoints
    	all = @allocated (t = @elapsed VoronoiFVM._solve_timestep!(solution, oldsol, sys, control, time, dt, empedparam, params; do_print_allocs, do_print_eaa))
    	tasm = sys.history.tasm
    	tlin = sys.history.tlinsolve
    	if do_print_ts
	    	@info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | LinSolveTime $(round(tlin, sigdigits=4)) | Allocs $all"
    	end
    	time += dt	
    	oldsol .= solution
    	sols[i+1] = copy(oldsol[1,:])
    	
    	if do_plot
    		scalarplot!(p[1, 1], grid, sols[i+1]; Plotter = PyPlot, clear = true, show = true, title="$time")
    	end
    end
	
	
	sols
end

function example_time_part(nm, dt, tpoints, nt; depth=2, verbose=false, unknown_storage=:sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, do_plot = true, do_print_allocs=1, do_print_eaa=false, do_print_ts=true)
    
    dim = length(nm)
    if dim == 2
        physics = VoronoiFVM.Physics(; reaction = reaction2,
                                    flux = flux2,
                                    source = source2, 
                                    storage = storage2)
    else
        physics = VoronoiFVM.Physics(; reaction = reaction3,
                                    flux = flux3,
                                    source = source3, 
                                    storage = storage3)
    end
	
	sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
    physics!(sys, physics)
    grid = sys.grid
    
    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    oldsol = unknowns(sys)
    oldsol .= 0.5
    solution = unknowns(sys)
    control = VoronoiFVM.NewtonControl()
    control.verbose = verbose
    control.reltol_linear = 1.0e-5
    control.method_linear = method_linear
    control.precon_linear = precon_linear
    control.log = true
    
    time = 0.0
    empedparam = 0.0
    params = zeros(0)
    
    sols = [zeros(num_nodes(grid)) for i=1:tpoints+1]
    sols[1] = copy(oldsol[1,:])
    
    if do_plot
    	p = GridVisualizer(;Plotter=PyPlot)
    end
    for i=1:tpoints
    	all = @allocated (t = @elapsed VoronoiFVM._solve_timestep_part!(solution, oldsol, sys, control, time, dt, empedparam, params; do_print_allocs, do_print_eaa))
    	tasm = sys.history.tasm
    	tlin = sys.history.tlinsolve
    	if do_print_ts
	    	@info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | LinSolveTime $(round(tlin, sigdigits=4)) | Allocs $all"
    	end
    	time += dt	
    	oldsol .= solution
    	sols[i+1] = copy(oldsol[1,:])
    	
    	if do_plot
    		scalarplot!(p[1, 1], grid, sols[i+1]; Plotter = PyPlot, clear = true, show = true, title="$time")
    	end
    end
	
	sols
end

function example_time_part_para(nm, dt, tpoints, nt; depth=2, verbose=false, unknown_storage=:sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, do_plot = true, do_print_allocs=1, 
              do_print_eaa=false, detail_allocs=false, do_print_ts=true, print_prep=false, check_partition=false)
    
    dim = length(nm)
    if dim == 2
        physics = VoronoiFVM.Physics(; reaction = reaction2,
                                    flux = flux2,
                                    source = source2, 
                                    storage = storage2)
    else
        physics = VoronoiFVM.Physics(; reaction = reaction3,
                                    flux = flux3,
                                    source = source3, 
                                    storage = storage3)
    end

	
	sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1], print_prep, check_partition)
    physics!(sys, physics)
    grid = sys.grid
    
    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    oldsol = unknowns(sys)
    oldsol .= 0.5
    solution = unknowns(sys)
    control = VoronoiFVM.NewtonControl()
    control.verbose = verbose
    control.reltol_linear = 1.0e-5
    control.method_linear = method_linear
    control.precon_linear = precon_linear
    control.log = true
    
    time = 0.0
    empedparam = 0.0
    params = zeros(0)
    
    sols = [zeros(num_nodes(grid)) for i=1:tpoints+1]
    sols[1] = copy(oldsol[1,:])

    if do_print_ts
    	@info "How many cells per partition on each level:", [length(sys.matrix.cellsforpart[(i-1)*nt+1]) for i=1:sys.matrix.depth+1]	
    end
    
    if do_plot
    	p = GridVisualizer(;Plotter=PyPlot)
    end
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    for i=1:tpoints
    	all = @allocated (t = @elapsed VoronoiFVM._solve_timestep_parallel!(solution, oldsol, sys, control, time, dt, empedparam, params; do_print_allocs, do_print_eaa, detail_allocs))
    	tasm = sys.history.tasm
    	tlin = sys.history.tlinsolve
    	if do_print_ts
	    	@info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | LinSolveTime $(round(tlin, sigdigits=4)) | Allocs $all"
    	end
    	time += dt	
    	oldsol .= solution
    	sols[i+1] = copy(oldsol[1,:])
    	
    	if do_plot
    		scalarplot!(p[1, 1], grid, sols[i+1]; Plotter = PyPlot, clear = true, show = true, title="$time")
    	end
    end
	
	
	sols
end

function example_time_part_para_outside(nm, dt, tpoints, nt; depth=2, verbose=false, unknown_storage=:sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, do_plot = true, do_print_allocs=1, do_print_eaa=false, detail_allocs=false, do_print_ts=true)
    
    dim = length(nm)
    if dim == 2
        physics = VoronoiFVM.Physics(; reaction = reaction2,
                                    flux = flux2,
                                    source = source2, 
                                    storage = storage2)
    else
        physics = VoronoiFVM.Physics(; reaction = reaction3,
                                    flux = flux3,
                                    source = source3, 
                                    storage = storage3)
    end
    
	
	sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
    physics!(sys, physics)
    grid = sys.grid
    
    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    oldsol = unknowns(sys)
    oldsol .= 0.5
    solution = unknowns(sys)
    control = VoronoiFVM.NewtonControl()
    control.verbose = verbose
    control.reltol_linear = 1.0e-5
    control.method_linear = method_linear
    control.precon_linear = precon_linear
    control.log = true
    
    time = 0.0
    empedparam = 0.0
    params = zeros(0)
    
    sols = [zeros(num_nodes(grid)) for i=1:tpoints+1]
    sols[1] = copy(oldsol[1,:])
    
    if do_plot
    	p = GridVisualizer(;Plotter=PyPlot)
    end
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    for i=1:tpoints
    	all = @allocated (t = @elapsed VoronoiFVM._solve_timestep_parallel_outside!(solution, oldsol, sys, control, time, dt, empedparam, params; do_print_allocs, do_print_eaa, detail_allocs))
    	tasm = sys.history.tasm
    	tlin = sys.history.tlinsolve
    	if do_print_ts
	    	@info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | LinSolveTime $(round(tlin, sigdigits=4)) | Allocs $all"
    	end
    	time += dt	
    	oldsol .= solution
    	sols[i+1] = copy(oldsol[1,:])
    	
    	if do_plot
    		scalarplot!(p[1, 1], grid, sols[i+1]; Plotter = PyPlot, clear = true, show = true, title="$time")
    	end
    end
	
	
	sols
end

#------------------------------------

function example_time_part_para_onetimestep(nm, dt, nt; depth=2, verbose=false, unknown_storage=:sparse,
    method_linear = KrylovJL_GMRES(), assembly = :cellwise,
    precon_linear = "", do_init = true, do_plot = true, do_print_allocs=1, do_print_eaa=true, detail_allocs=true, do_print_ts=true)

    if precon_linear == ""
        precon_linear = A -> VoronoiFVM.Identity()
    elseif precon_linear == "iluz"
        precon_linear = VoronoiFVM.ILUZeroPreconditioner()
    elseif precon_linear == "iluam"
        precon_linear = VoronoiFVM.ILUAMPreconditioner()
    elseif precon_linear == "piluam"
        precon_linear = VoronoiFVM.PILUAMPreconditioner()
    end

    @info precon_linear

    dim = length(nm)
    if dim == 2
        physics = VoronoiFVM.Physics(; reaction = reaction2,
                                flux = flux2,
                                source = source2, 
                                storage = storage2)
    else
        physics = VoronoiFVM.Physics(; reaction = reaction3,
                                flux = flux3,
                                source = source3, 
                                storage = storage3)
    end


    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
    physics!(sys, physics)
    grid = sys.grid

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    oldsol = unknowns(sys)
    oldsol .= 0.5
    solution = unknowns(sys)
    control = VoronoiFVM.NewtonControl()
    control.verbose = verbose
    control.reltol_linear = 1.0e-5
    control.method_linear = method_linear
    control.precon_linear = precon_linear
    control.log = true

    time = 0.0
    empedparam = 0.0
    params = zeros(0)

    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)

    VoronoiFVM._solve_timestep_parallel!(solution, oldsol, sys, control, time, dt, empedparam, params; do_print_allocs, do_print_eaa, detail_allocs)

end

#-----------------------------------

function validate_timesteps(
    nm, nt, depth, dt; 
    tpoints=3, do_print_allocs=0, do_print_eaa=false, do_print_ts=true, precon_linear=nothing, method_linear=nothing, 
    detail_allocs=false, print_prep=false, check_partition=false)

	#dim = length(nm)
    if method_linear === nothing
        method_linear = KrylovJL_GMRES()
    end
    if precon_linear === nothing
        precon_linear = VoronoiFVM.ILUZeroPreconditioner()
    end

    
	@info "Computing normal solution"
	if typeof(precon_linear) == VoronoiFVM.PILUAMPreconditioner
        sols_normal    = example_time_normal(   nm, dt, tpoints;            do_plot=false, do_print_allocs, do_print_eaa, do_print_ts, method_linear, precon_linear=VoronoiFVM.ILUAMPreconditioner())
    else
        sols_normal    = example_time_normal(   nm, dt, tpoints;            do_plot=false, do_print_allocs, do_print_eaa, do_print_ts, method_linear, precon_linear)
	end
    #if do_part
	#    @info "Computing partitioned solution"
	#    sols_part      = example_time_part(     nm, dt, tpoints, nt; depth, do_plot=false, do_print_allocs, do_print_eaa, do_print_ts, method_linear, precon_linear)
	#end
	@info "Computing parallelized solution"
	sols_part_para = example_time_part_para(nm, dt, tpoints, nt; depth, do_plot=false, do_print_allocs, do_print_eaa, do_print_ts, detail_allocs, method_linear, precon_linear, print_prep, check_partition)
	
	
	#if do_part
	#	d1 = absmax_entry_vecvec(sols_normal - sols_part)
	#	d2 = absmax_entry_vecvec(sols_normal - sols_part_para)
	#	return d1, d2
	#else
	d2 = absmax_entry_vecvec(sols_normal - sols_part_para)
	return d2
	#end
end

function absmax_entry_vecvec(A)
	maxs = maximum.(A) #[A[:,i] for i=1:size(A,2))
	maximum(maxs)
end


