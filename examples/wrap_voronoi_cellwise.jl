#include("/home/johannes/Nextcloud/Documents/Uni/VIII/WIAS/juliaCode(anfang)/para/ExtendableSparseParallel/src/ESMP/ExtendableSparseParallel.jl")

#using .ExtendableSparseParallel


#https://github.com/j-fu/VoronoiFVM.jl/blob/7039d768ba26e80dbb1aa4fabcce5993486561a4/examples/Example207_NonlinearPoisson2D.jl

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
    
    t = @elapsed ((n1, n2) = VoronoiFVM.eval_and_assemble(sys,
		                               solution,
		                               oldsol,
		                               residual,
		                               0.0,
		                               0.01,
		                               0.0,
		                               zeros(0);
		                               edge_cutoff = 1e-5)) #control.edge_cutoff,)
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
    sys.matrix, t, n1, n2
end




function example3_ESMP(nm, nt, depth; verbose = false, unknown_storage = :sparse,
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
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

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
    
    VoronoiFVM.fix_deprecations!(control)
    =#

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t = @elapsed ((n1, n2) = VoronoiFVM.eval_and_assemble_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0);edge_cutoff = 1e-5))
	
	VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(sys.matrix; do_dense=true)
	
	sys.matrix, t, n1, n2
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
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

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
    
    VoronoiFVM.fix_deprecations!(control)
    =#

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t = @elapsed ((n1, n2) = VoronoiFVM.eval_and_assemble_part_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0);edge_cutoff = 1e-5))
	
	VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(sys.matrix; do_dense=true)
	
	sys.matrix, t, n1, n2
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
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

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
    
    VoronoiFVM.fix_deprecations!(control)
    =#

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t = @elapsed ((n1, n2) = VoronoiFVM.eval_and_assemble_part_para_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0);edge_cutoff = 1e-5)) #control.edge_cutoff)
	
	VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(sys.matrix; do_dense=true)
	
	sys.matrix, t, n1, n2
end



function example3_reorder(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init=true)
    grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts = VoronoiFVM.ExtendableSparseParallel.preparatory_multi_ps_less_reverse(nm, nt, depth, Int64)          
    
    #grid = VoronoiFVM.ExtendableSparseParallel.getgrid(nm)

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
    sys = VoronoiFVM.System(grid, physics; unknown_storage, assembly = assembly, species=[1])
    #enable_species!(sys, 1, [1])

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
    
    t = @elapsed ((n1, n2) = VoronoiFVM.eval_and_assemble_ESMP(sys,
		                         solution,
		                         oldsol,
		                         residual,
		                         0.0,
		                         0.01,
		                         0.0,
		                         zeros(0);
		                         edge_cutoff = 1e-5, #control.edge_cutoff,
		                         new_ind = ni))
    
    sys.matrix, t, n1, n2
end



function reorder(x, ni)
	y = copy(x)
	for i=1:length(x)
		y[ni[i]] = x[i]
	end
	y

end

function validate_ESMP(nm, nt, depth, do_init; test_vec=true)
	ESM,t1,n11,n12   = example3(nm; do_init=do_init)
	CSC1             = SparseArrays.SparseMatrixCSC(ESM)
	
	#ESM2,t2,n21,n22  = example3_reorder(nm, nt, depth; do_init=do_init)
	#CSC2             = SparseArrays.SparseMatrixCSC(ESM2)
	
	#ESMP,t3,n31,n32  = example3_ESMP(nm, nt, depth; do_init=do_init)
	#CSC3             = ESMP.cscmatrix
	
	ESMP2,t4,n41,n42 = example3_ESMP_part(nm, nt, depth; do_init=do_init)
	CSC4     = ESMP2.cscmatrix
	
	ESMP3,t5,n51,n52 = example3_ESMP_part_para(nm, nt, depth; do_init=do_init)
	CSC5             = ESMP3.cscmatrix
	
	if test_vec	
		nn = num_nodes(ESMP2.grid)
		b = rand(nn)
		v = CSC1*b
		
		br = reorder(b, ESMP2.new_indices)
		#vr2 = reorder(CSC2*br, ESMP.rev_new_indices)
		#vr3 = reorder(CSC3*br, ESMP.rev_new_indices)
		vr4 = reorder(CSC4*br, ESMP2.rev_new_indices)
		vr5 = reorder(CSC5*br, ESMP3.rev_new_indices)
		
		#@info "max diff 2: ", maximum(abs.(v-vr2))
		#@info "max diff 3: ", maximum(abs.(v-vr3))
		@info "max diff 4: ", maximum(abs.(v-vr4))
		@info "max diff 5: ", maximum(abs.(v-vr5))
		@info "max nz val: ", maximum(abs.(CSC1.nzval))
	end
	
	@info "Times:"
	@info "Old Seq: ", t1, n11, n12
	@info "New Seq: ", t4, n41, n42
	@info "New Par: ", t5, n51, n52
	
	#CSC1, CSC2, CSC3, ESMP
end


