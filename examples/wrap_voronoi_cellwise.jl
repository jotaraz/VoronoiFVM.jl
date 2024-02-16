#include("/home/johannes/Nextcloud/Documents/Uni/VIII/WIAS/juliaCode(anfang)/para/ExtendableSparseParallel/src/ESMP/ExtendableSparseParallel.jl")

#using .ExtendableSparseParallel


#https://github.com/j-fu/VoronoiFVM.jl/blob/7039d768ba26e80dbb1aa4fabcce5993486561a4/examples/Example207_NonlinearPoisson2D.jl

using PyPlot, GridVisualize, ExtendableGrids


function example3(nm; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)
    #n1,n2 = nm
    #h = 1.0 / convert(Float64, n)
    #X = collect(0.0:h:1.0)
    #Y = collect(0.0:h:1.0)

    grid = VoronoiFVM.ExtendableSparseParallel.getgrid(nm)

    eps = 1.0e-2

    physics = VoronoiFVM.Physics(; reaction = function (f, u, node)
                                     f[1] = u[1]^2
                                 end, flux = function (f, u, edge)
                                     f[1] = eps * (u[1, 1]^2 - u[1, 2]^2)
                                 end, source = function (f, node)
                                     x1 = node[1] - 0.5
                                     x2 = node[2] - 0.5
                                     f[1] = exp(-20.0 * (x1^2 + x2^2))
                                 end, storage = function (f, u, node)
                                     f[1] = u[1]
                                 end)
    sys = VoronoiFVM.System(grid, physics; unknown_storage, assembly = assembly)
    enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    inival = unknowns(sys)
    inival .= 0.5
	
	#=
	control = VoronoiFVM.NewtonControl()
    control.verbose = verbose
    control.reltol_linear = 1.0e-5
    control.method_linear = method_linear
    control.precon_linear = precon_linear
	#CommonSolve.solve(sys; inival=inival, control = VoronoiFVM.SolverControl(), tstep = tstep)
    #VoronoiFVM.solve(inival, sys; called_from_API = true, control=control, params=zeros(0), time=0.0, tstep=0.01)
    VoronoiFVM.fix_deprecations!(control)
    =#
    
    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    if do_init
	    VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble(sys,
		                               solution,
		                               oldsol,
		                               residual,
		                               0.0,
		                               0.01,
		                               0.0,
		                               zeros(0);
		                               edge_cutoff = 1e-5)) #control.edge_cutoff,)
    
    
    t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble(sys,
		                               solution,
		                               oldsol,
		                               residual,
		                               0.0,
		                               0.01,
		                               0.0,
		                               zeros(0);
		                               edge_cutoff = 1e-5))
    
    #=
    VoronoiFVM._solve_timestep!(solution,
                     inival,
                     sys,
                     control,
                     0.0,
                     0.01,
                     0,
                     zeros(0);
                     called_from_API = true,)
    
    VoronoiFVM.solve!(unknowns(system),
           inival,
           system;
           control = control,
           time = 0.0,
           tstep = 0.01,
           params = zeros(0),
           called_from_API = true,)
	=#
    sys.matrix, residual, (t1, t2), (na1, na2)
end



function example3_ESMP_part(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)
    #grid = VoronoiFVM.getgrid(nm)

	eps = 1.0e-2

    physics = VoronoiFVM.Physics(; reaction = function (f, u, node)
                                     f[1] = u[1]^2
                                 end, flux = function (f, u, edge)
                                     f[1] = eps * (u[1, 1]^2 - u[1, 2]^2)
                                 end, source = function (f, node)
                                     x1 = node[1] - 0.5
                                     x2 = node[2] - 0.5
                                     f[1] = exp(-20.0 * (x1^2 + x2^2))
                                 end, storage = function (f, u, node)
                                     f[1] = u[1]
                                 end)
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth;  unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    inival = unknowns(sys)
    inival .= 0.5
	

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble_part_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0);edge_cutoff = 1e-5))
	
	t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble_part_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0);edge_cutoff = 1e-5))
	
	
	#VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(sys.matrix; do_dense=true)
	
	sys.matrix, residual, (t1, t2), (na1, na2)
end

function example3_ESMP_part_para(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)
    #grid = VoronoiFVM.getgrid(nm)

	eps = 1.0e-2

    physics = VoronoiFVM.Physics(; reaction = function (f, u, node)
                                     f[1] = u[1]^2
                                 end, flux = function (f, u, edge)
                                     f[1] = eps * (u[1, 1]^2 - u[1, 2]^2)
                                 end, source = function (f, node)
                                     x1 = node[1] - 0.5
                                     x2 = node[2] - 0.5
                                     f[1] = exp(-20.0 * (x1^2 + x2^2))
                                 end, storage = function (f, u, node)
                                     f[1] = u[1]
                                 end)
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    inival = unknowns(sys)
    inival .= 0.5

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble_part_para_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5)) #control.edge_cutoff)
    
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t1, na1
	
	t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble_part_para_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5))
	
	
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t2, na2
	
	#VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(sys.matrix; do_dense=true)
	
	sys.matrix, residual, (t1, t2), (na1, na2)
end

function example3_ESMP_part_para_sepa(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)
    #grid = VoronoiFVM.getgrid(nm)

	eps = 1.0e-2

    physics = VoronoiFVM.Physics(; reaction = function (f, u, node)
                                     f[1] = u[1]^2
                                 end, flux = function (f, u, edge)
                                     f[1] = eps * (u[1, 1]^2 - u[1, 2]^2)
                                 end, source = function (f, node)
                                     x1 = node[1] - 0.5
                                     x2 = node[2] - 0.5
                                     f[1] = exp(-20.0 * (x1^2 + x2^2))
                                 end, storage = function (f, u, node)
                                     f[1] = u[1]
                                 end)
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    inival = unknowns(sys)
    inival .= 0.5

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble_part_para_ESMP_sepa(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5)) #control.edge_cutoff)
    
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t, na
	
	t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble_part_para_ESMP_sepa(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5))
	
	
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t, na
	
	#VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(sys.matrix; do_dense=true)
	
	sys.matrix, residual, (t1, t2), (na1, na2)
end

function example3_ESMP_part_para_flex(nm, nt, depth, fct; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)
    #grid = VoronoiFVM.getgrid(nm)

	eps = 1.0e-2

    physics = VoronoiFVM.Physics(; reaction = function (f, u, node)
                                     f[1] = u[1]^2
                                 end, flux = function (f, u, edge)
                                     f[1] = eps * (u[1, 1]^2 - u[1, 2]^2)
                                 end, source = function (f, node)
                                     x1 = node[1] - 0.5
                                     x2 = node[2] - 0.5
                                     f[1] = exp(-20.0 * (x1^2 + x2^2))
                                 end, storage = function (f, u, node)
                                     f[1] = u[1]
                                 end)
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    inival = unknowns(sys)
    inival .= 0.5

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = fct(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5))
    
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t1, na1
	
	t2 = @elapsed (na2 = fct(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5))
	
	
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t2, na2
	
	#VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(sys.matrix; do_dense=true)
	
	sys.matrix, residual, (t1, t2), (na1, na2)
end


function reorder(x, ni)
	y = Vector{typeof(x[1])}(undef, length(x)) #copy(x)
	for i=1:length(x)
		y[ni[i]] = x[i]
	end
	y

end

function validate_ESMP(nm, nt, depth, do_init; test_vec=true)
	ESM,r1,t1,n1     = example3(nm; do_init=do_init)
	CSC1             = SparseArrays.SparseMatrixCSC(ESM)
	
	#ESM2,t2,n21,n22  = example3_reorder(nm, nt, depth; do_init=do_init)
	#CSC2             = SparseArrays.SparseMatrixCSC(ESM2)
	
	#ESMP,t3,n31,n32  = example3_ESMP(nm, nt, depth; do_init=do_init)
	#CSC3             = ESMP.cscmatrix
	
	ESMP2,r4,t4,n4   = example3_ESMP_part(nm, nt, depth; do_init=do_init)
	CSC4     = ESMP2.cscmatrix
	
	ESMP3,r5,t5,n5   = example3_ESMP_part_para(nm, nt, depth; do_init=do_init)
	CSC5             = ESMP3.cscmatrix
	
	r1 = VoronoiFVM.values(r1)
	r4 = VoronoiFVM.values(r4)
	r5 = VoronoiFVM.values(r5)
	
	@info typeof(r1), typeof(r4), typeof(r5)
	
	if test_vec	
		nn = num_nodes(ESMP2.grid)
		b = rand(nn)
		v = CSC1*b
		
		br = reorder(b, ESMP2.new_indices)
		#vr2 = reorder(CSC2*br, ESMP.rev_new_indices)
		#vr3 = reorder(CSC3*br, ESMP.rev_new_indices)
		vr4 = reorder(CSC4*br, ESMP2.rev_new_indices)
		vr5 = reorder(CSC5*br, ESMP3.rev_new_indices)
		
		rr4 = reorder(r4, ESMP2.rev_new_indices)
		rr5 = reorder(r5, ESMP3.rev_new_indices)
		
		#@info "max diff 2: ", maximum(abs.(v-vr2))
		#@info "max diff 3: ", maximum(abs.(v-vr3))
		@warn "Differences in RHS"
		@info "max diff 4: ", maximum(abs.(r1-rr4))
		@info "max diff 5: ", maximum(abs.(r1-rr5))
		
		@warn "Differences in Matrix"
		@info "max diff 4: ", maximum(abs.(v-vr4))
		@info "max diff 5: ", maximum(abs.(v-vr5))
		#@info "max nz val: ", maximum(abs.(CSC1.nzval))
	end
	
	@warn "Times:"
	@info "Old Seq: ", t1
	@info "New Seq: ", t4
	@info "New Par: ", t5
	@warn "Allocations:"
	@info "Old Seq: ", n1
	@info "New Seq: ", n4
	@info "New Par: ", n5
	
	#CSC1, CSC2, CSC3, ESMP
end

#------------------------------------

function reaction(f, u, node)
	f[1] = u[1]^2
end

function flux(f, u, edge)
	f[1] = 1e-2 * (u[1, 1]^2 - u[1, 2]^2)
end

function source(f, node)
	x1 = node[1] - 0.5
	x2 = node[2] - 0.5
    f[1] = exp(-20.0 * (x1^2 + x2^2))
end

function storage(f, u, node)
	f[1] = u[1]
end

function example_time_normal(nm, dt, tpoints; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, do_plot = true, do_print_allocs=1, do_print_eaa=false, do_print_ts=true)
    grid = VoronoiFVM.ExtendableSparseParallel.getgrid(nm)

    #eps = 1.0e-2
	physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
	#=
    physics = VoronoiFVM.Physics(; reaction = function (f, u, node)
                                     f[1] = u[1]^2
                                 end, flux = function (f, u, edge)
                                     f[1] = eps * (u[1, 1]^2 - u[1, 2]^2)
                                 end, source = function (f, node)
                                     x1 = node[1] - 0.5
                                     x2 = node[2] - 0.5
                                     f[1] = exp(-20.0 * (x1^2 + x2^2))
                                 end, storage = function (f, u, node)
                                     f[1] = u[1]
                                 end)
    =#
    
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
    	if do_print_ts
	    	@info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | Allocs $all"
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
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
	
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
    	if do_print_ts
	    	@info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | Allocs $all"
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
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, do_plot = true, do_print_allocs=1, do_print_eaa=false, detail_allocs=false, do_print_ts=true)
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
	
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
    	all = @allocated (t = @elapsed VoronoiFVM._solve_timestep_parallel!(solution, oldsol, sys, control, time, dt, empedparam, params; do_print_allocs, do_print_eaa, detail_allocs))
    	tasm = sys.history.tasm
    	if do_print_ts
	    	@info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | Allocs $all"
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



function validate_timesteps(nm, nt, depth, dt; tpoints=5, do_print_allocs=0, do_print_eaa=false, do_print_ts=false, do_part=true)
	@info "Computing normal solution"
	sols_normal    = example_time_normal(   nm, dt, tpoints;            do_plot=false, do_print_allocs, do_print_eaa, do_print_ts)
	if do_part
	@info "Computing partitioned solution"
	sols_part      = example_time_part(     nm, dt, tpoints, nt; depth, do_plot=false, do_print_allocs, do_print_eaa, do_print_ts)
	end
	@info "Computing parallelized solution"
	sols_part_para = example_time_part_para(nm, dt, tpoints, nt; depth, do_plot=false, do_print_allocs, do_print_eaa, do_print_ts, detail_allocs=false)
	
	
	if do_part
		d1 = absmax_entry_vecvec(sols_normal - sols_part)
		d2 = absmax_entry_vecvec(sols_normal - sols_part_para)
		return d1, d2
	else
		d2 = absmax_entry_vecvec(sols_normal - sols_part_para)
		return d2
	end
end

function absmax_entry_vecvec(A)
	maxs = maximum.(A) #[A[:,i] for i=1:size(A,2))
	maximum(maxs)
end


