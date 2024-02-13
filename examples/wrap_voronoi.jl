#include("/home/johannes/Nextcloud/Documents/Uni/VIII/WIAS/juliaCode(anfang)/para/ExtendableSparseParallel/src/ESMP/ExtendableSparseParallel.jl")

#using .ExtendableSparseParallel


#https://github.com/j-fu/VoronoiFVM.jl/blob/7039d768ba26e80dbb1aa4fabcce5993486561a4/examples/Example207_NonlinearPoisson2D.jl

m = 2.0

function flux!(f, u, edge)
    f[1] = u[1, 1]^m - u[1, 2]^m
end

function storage!(f, u, node)
    f[1] = u[1]
end

function barenblatt(x, t, m)
    tx = t^(-1.0 / (m + 1.0))
    xx = x * tx
    xx = xx * xx
    xx = 1 - xx * (m - 1) / (2.0 * m * (m + 1))
    if xx < 0.0
        xx = 0.0
    end
    return tx * xx^(1.0 / (m - 1.0))
end

function test_VoronoiFVM_ESMP(nm, nt, depth)
	eps = 1e-2
	
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
	
	
	system = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
	
	physics!(system, physics)
	
	
	boundary_dirichlet!(system, 1, 2, 0.1)
    boundary_dirichlet!(system, 1, 4, 0.1)
	
    #enable_species!(system, 1, [1])
	
	inival = (unknowns(system)*0.0).+0.5
	oldval = unknowns(system)*0.0.+0.5
	rhs    = unknowns(system)*0.0
	time = 0
	tstep = 1e-2
	λ = 0.0
	params = zeros(0)
	
	VoronoiFVM._complete_nomatrix!(system; create_newtonvectors = true)
	
	@warn "num params:", system.num_parameters
	
	VoronoiFVM.eval_and_assemble_ESMP(system, inival, oldval, rhs, time, tstep, λ, params)
	
	VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(system.matrix; do_dense=true)
	
	
	system.matrix
end


function test_VoronoiFVM_normal(nm)
	eps = 1e-2
	
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
	
	grid = VoronoiFVM.ExtendableSparseParallel.getgrid(nm)
	system = VoronoiFVM.System(grid; species = [1])
	
	physics!(system, physics)
	
	
	boundary_dirichlet!(system, 1, 2, 0.1)
    boundary_dirichlet!(system, 1, 4, 0.1)
	
    #enable_species!(system, 1, [1])
	
	inival = (unknowns(system)*0.0).+0.5
	oldval = unknowns(system)*0.0.+0.5
	rhs    = unknowns(system)*0.0
	time = 0
	tstep = 1e-2
	λ = 0.0
	params = zeros(0)
	
	VoronoiFVM._complete!(system; create_newtonvectors = true)
	
	@warn "num params:", system.num_parameters
	
	VoronoiFVM.eval_and_assemble(system, inival, oldval, rhs, time, tstep, λ, params)
	
	#VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(system.matrix; do_dense=true)
	
	
	system.matrix
end


function example(; n = 10, Plotter = nothing, verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :edgewise,
              precon_linear = A -> VoronoiFVM.Identity())
    h = 1.0 / convert(Float64, n)
    X = collect(0.0:h:1.0)
    Y = collect(0.0:h:1.0)

    grid = VoronoiFVM.Grid(X, Y)

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

    control = VoronoiFVM.NewtonControl()
    control.verbose = verbose
    control.reltol_linear = 1.0e-5
    control.method_linear = method_linear
    control.precon_linear = precon_linear
    tstep = 0.01
    time = 0.0
    u15 = 0
    p = GridVisualizer(; Plotter = Plotter)
    while time < 1.0
        time = time + tstep
        U = solve(sys; inival, control, tstep)
        u15 = U[15]
        inival .= U

        scalarplot!(p[1, 1], grid, U[1, :]; Plotter = Plotter, clear = true, show = true)
        tstep *= 1.0
    end
    return u15
end



function example2(; n = 10, Plotter = nothing, verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :edgewise,
              precon_linear = A -> VoronoiFVM.Identity())
    h = 1.0 / convert(Float64, n)
    X = collect(0.0:h:1.0)
    Y = collect(0.0:h:1.0)

    grid = VoronoiFVM.Grid(X, Y)

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
	
	control = VoronoiFVM.NewtonControl()
    control.verbose = verbose
    control.reltol_linear = 1.0e-5
    control.method_linear = method_linear
    control.precon_linear = precon_linear
	#CommonSolve.solve(sys; inival=inival, control = VoronoiFVM.SolverControl(), tstep = tstep)
    #VoronoiFVM.solve(inival, sys; called_from_API = true, control=control, params=zeros(0), time=0.0, tstep=0.01)
    VoronoiFVM.fix_deprecations!(control)
    solution = unknowns(sys)
    VoronoiFVM._solve_timestep!(solution,
                     inival,
                     sys,
                     control,
                     0.0,
                     0.01,
                     0,
                     zeros(0);
                     called_from_API = true,)
    
    #=
    VoronoiFVM.solve!(unknowns(system),
           inival,
           system;
           control = control,
           time = 0.0,
           tstep = 0.01,
           params = zeros(0),
           called_from_API = true,)
	=#
    solution
end



function example3(nm; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :edgewise,
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
    
    t = @elapsed VoronoiFVM.eval_and_assemble(sys,
		                               solution,
		                               oldsol,
		                               residual,
		                               0.0,
		                               0.01,
		                               0.0,
		                               zeros(0);
		                               edge_cutoff = 1e-5) #control.edge_cutoff,)
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
    sys.matrix, t
end




function example3_ESMP(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :edgewise,
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
    
    t = @elapsed VoronoiFVM.eval_and_assemble_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0);edge_cutoff = 1e-5) #control.edge_cutoff)
	
	VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(sys.matrix; do_dense=true)
	
	sys.matrix, t
end


function example3_reorder(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :edgewise,
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
    
    t = @elapsed VoronoiFVM.eval_and_assemble_ESMP(sys,
		                         solution,
		                         oldsol,
		                         residual,
		                         0.0,
		                         0.01,
		                         0.0,
		                         zeros(0);
		                         edge_cutoff = 1e-5, #control.edge_cutoff,
		                         new_ind = ni)
    
    sys.matrix, t
end



function reorder(x, ni)
	y = copy(x)
	for i=1:length(x)
		y[ni[i]] = x[i]
	end
	y

end

function validate_ESMP(nm, nt, depth, do_init; test_vec=true)
	ESM,t1  = example3(nm; do_init=do_init)
	CSC1    = SparseArrays.SparseMatrixCSC(ESM)
	
	ESM2,t2 = example3_reorder(nm, nt, depth; do_init=do_init)
	CSC2    = SparseArrays.SparseMatrixCSC(ESM2)
	
	ESMP,t3 = example3_ESMP(nm, nt, depth; do_init=do_init)
	CSC3    = ESMP.cscmatrix
	
	if test_vec	
		nn = num_nodes(ESMP.grid)
		b = rand(nn)
		v = CSC1*b
		
		br = reorder(b, ESMP.new_indices)
		vr2 = reorder(CSC2*br, ESMP.rev_new_indices)
		vr3 = reorder(CSC3*br, ESMP.rev_new_indices)
		
		@info "max diff 1: ", maximum(abs.(v-vr2))
		@info "max diff 3: ", maximum(abs.(v-vr3))
		@info "max nz val: ", maximum(abs.(CSC2.nzval))
	end
	
	@info t1, t2, t3

	CSC1, CSC2, CSC3, ESMP
end


function direct_compare(nm, nt, depth)
	grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts = VoronoiFVM.ExtendableSparseParallel.preparatory_multi_ps_less_reverse(nm, nt, depth, Int64)          
    nn = VoronoiFVM.ExtendableGrids.num_nodes(grid)
    
    A = ExtendableSparseMatrix{Float64, Int64}(nn, nn)
    
    
    csc = spzeros(Float64, Int64, nn, nn)
	lnk = [VoronoiFVM.ExtendableSparseParallel.SuperSparseMatrixLNK{Float64, Int64}(nn, nnts[tid]) for tid=1:nt]
	B = VoronoiFVM.ExtendableSparseParallel.ExtendableSparseMatrixParallel{Float64, Int64}(csc, lnk, grid, nnts, s, onr, cfp, gi, ni, rni, starts, nt, depth)
	
	nj = B.new_indices
	
	cellids = 1:VoronoiFVM.ExtendableGrids.num_cells(grid)#[1, 3, 10]
	
	for cell in cellids
		for i in grid[CellNodes][:,cell]
			for j in grid[CellNodes][:,cell]
				v = rand()
				VoronoiFVM._addnz(A, ni[i], ni[j], v, 0.5)
				VoronoiFVM._addnz(B, nj[i], nj[j], v, 0.5)
			end
		end
	end
	
	
	CA = SparseArrays.SparseMatrixCSC(A)

	VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(B; do_dense=true)
	CB = B.cscmatrix
	
	VoronoiFVM.ExtendableSparseParallel.compare_matrices_light(CA, CB)
	
	CA, CB
end



function validate_ESMP2(nm, nt, depth)
	ESM  = test_VoronoiFVM_normal(nm)
	CSC2 = SparseArrays.SparseMatrixCSC(ESM)
	
	
	ESMP = test_VoronoiFVM_ESMP(nm, nt, depth)
	CSC1 = ESMP.cscmatrix
	
	
	
	nn = num_nodes(ESMP.grid)
	b = rand(nn)
	v = CSC2*b
	
	br = reorder(b, ESMP.new_indices)
	vr1 = reorder(CSC1*br, ESMP.rev_new_indices)
	
	@info "max diff 1: ", maximum(abs.(v-vr1))



	CSC1, CSC2, ESMP
end



function test2()
	h0 = 0.005 / 2.0^0
    h1 = 0.1 / 2.0^0

    X = VoronoiFVM.geomspace(0, 1.0, h0, h1)
    grid = VoronoiFVM.Grid(X)

	data = (R = 0.5, D = 2.0, C = 1.0)

    # Declare constitutive functions
    flux = function (f, u, edge, data)
        f[1] = data.D * (u[1, 1] - u[1, 2])
    end

    storage = function (f, u, node, data)
        f[1] = data.C * u[1]
    end

    reaction = function (f, u, node, data)
        f[1] = data.R * u[1]
    end
    
    excited_bc = 1
    excited_bcval = 1.0
    excited_spec = 1
    meas_bc = 2
    
    physics = VoronoiFVM.Physics(; data = data,
                                 flux = flux,
                                 storage = storage,
                                 reaction = reaction)
                                 
    sys = VoronoiFVM.System(grid, physics; unknown_storage = :sparse, assembly = :edgewise)

    enable_species!(sys, excited_spec, [1])
end
