

using ThreadPinning; pinthreads(:cores)

using LinearSolve
using AMGCLWrap
using ExtendableSparse


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

function setup(nm; seq=true, unknown_storage=:sparse, assembly=:cellwise)
	
	#if seq	
		grid = VoronoiFVM.ExtendableSparseParallel.getgrid(nm)

		physics = VoronoiFVM.Physics(; reaction = reaction,
									   flux = flux,
									   source = source, 
									   storage = storage)
									   
		sys = VoronoiFVM.System(grid, physics; unknown_storage, assembly = assembly)
		enable_species!(sys, 1, [1])
		boundary_dirichlet!(sys, 1, 2, 0.1)
		boundary_dirichlet!(sys, 1, 4, 0.1)

		inival = unknowns(sys)
		inival .= 0.5
		
		
		solution = unknowns(sys)
		oldsol = inival
		
		VoronoiFVM._complete!(sys; create_newtonvectors = true)
		
		solution .= oldsol
		
		residual = sys.residual
		update = sys.update
		VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
		
		VoronoiFVM.eval_and_assemble(sys,
		                           solution,
		                           oldsol,
		                           residual,
		                           0.0,
		                           0.01,
		                           0.0,
		                           zeros(0);
		                           edge_cutoff = 1e-5)
	#end    
	
	
	sys #.matrix, sys.residual, sys.update
end



function run_tests(nm, strategies; 
					method_linear = nothing, 
					precon_linear = A -> VoronoiFVM.Identity(),
					verbose=false)
	system = setup(nm)
	
	for (i,strat) in enumerate(strategies)
		@info i
		#system.update .= 0.0
		
		control = VoronoiFVM.NewtonControl()
		control.verbose = verbose
		control.reltol_linear = 1.0e-5
		control.method_linear = method_linear
		control.precon_linear = precon_linear
		control.log = true
		nlhistory = NewtonSolverHistory()
		
		method_linear = system.matrixtype == :sparse ? control.method_linear : nothing;
		
		@info system.matrixtype
		
	    if isnothing(method_linear) &&  system.matrixtype == :sparse
	        #if Tv != Float64
	        #    method_linear = SparspakFactorization()
	        #else
	        #if dim_space(system.grid)==1
	            method_linear = KLUFactorization()
	        #elseif dim_space(system.grid)==2
	        #    method_linear = SparspakFactorization()
	        #else
	        #    method_linear = UMFPACKFactorization() # seems to do the best pivoting
	        #end
	    end
			
		t = @elapsed VoronoiFVM._solve_linear!(values(system.update),
                                     system,
                                     nlhistory,
                                     control,
                                     method_linear,
                                     system.matrix,
                                     values(system.residual))
                                     
        @info i, t
		
	end

end




function rt2(nm, dt, nt; depth=2, verbose=false, unknown_storage=:sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, do_plot = true, do_print_allocs=1, do_print_eaa=false, detail_allocs=false, do_print_ts=true, num=5)
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
	
	if nt != 0
		sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
		grid = sys.grid
    else
    	grid = VoronoiFVM.ExtendableSparseParallel.getgrid(nm)
		sys = VoronoiFVM.System(grid; species = [1])
	end
    physics!(sys, physics)
    
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
    
    if nt != 0
	    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    else
        VoronoiFVM._complete!(sys; create_newtonvectors = true)
    end
    
    system = sys
    called_from_API = false
    nlhistory = NewtonSolverHistory()
    
    solution .= oldsol
	residual = system.residual
	update   = system.update
	VoronoiFVM._initialize!(solution, system; time, λ = empedparam, params)
    
    method_linear = system.matrixtype == :sparse ? control.method_linear : nothing;
    if isnothing(method_linear) &&  system.matrixtype == :sparse
        #if Tv != Float64
        #    method_linear = SparspakFactorization()
        #else
        if VoronoiFVM.dim_space(system.grid)==1
            method_linear = KLUFactorization()
        elseif VoronoiFVM.dim_space(system.grid)==2
            method_linear = SparspakFactorization()
        else
            method_linear = UMFPACKFactorization() # seems to do the best pivoting
        end
    end
    
    @info precon_linear, typeof(system.matrix)
    


    if nt != 0
		VoronoiFVM.eval_and_assemble_part_para_ESMP_fcts(system,
		                                       solution,
		                                       oldsol,
		                                       residual,
		                                       time,
		                                       dt,
		                                       empedparam,
		                                       params;
		                                       edge_cutoff = control.edge_cutoff,
		                                       detail_allocs)
	else
		VoronoiFVM.eval_and_assemble(system,
                                   solution,
                                   oldsol,
                                   residual,
                                   time,
                                   dt,
                                   empedparam,
                                   params;
                                   edge_cutoff = control.edge_cutoff)
	end	                                                                                                            
                                           
	t1 = @elapsed VoronoiFVM._solve_linear!(values(update[1,:]),
		             system,
		             nlhistory,
		             control,
		             method_linear,
		             system.matrix,
		             values(residual)[1,:])
	
	t = zeros(num)		
	             
	for i=1:num
		update[1,:] .= 0.0
		t[i] = @elapsed VoronoiFVM._solve_linear!(values(update[1,:]),
												 system,
												 nlhistory,
												 control,
												 method_linear,
				
												 system.matrix,
												 values(residual)[1,:])
	end
	update[1,:] .= 0.0
	all = @allocated VoronoiFVM._solve_linear!(values(update[1,:]),
												system,
												nlhistory,
												control,
												method_linear,
												system.matrix,
												values(residual)[1,:]) 					 
    t1, minimum(t), sum(t)/num, maximum(t), (all*1.0)
end	
    
function loop1(nm, nt; num=5, method_linear = KrylovJL_GMRES())
	strats = [A -> VoronoiFVM.Identity(), VoronoiFVM.UMFPACKFactorization(), VoronoiFVM.ILUZeroPreconditioner(), VoronoiFVM.ExtendableSparse.ILUAMPreconditioner(), VoronoiFVM.ExtendableSparse.PILUAMPreconditioner(), AMGCL_AMGPreconditioner]
	@info method_linear
	for strat in strats
		@info rt2(nm, 0.01, nt; method_linear, precon_linear=strat, num)
	end
	
end
    
function loop2(nm, nt; num=5, method_linear = KrylovJL_GMRES())
	strats = [VoronoiFVM.UMFPACKFactorization(), VoronoiFVM.ILUZeroPreconditioner(), VoronoiFVM.ExtendableSparse.ILUAMPreconditioner(), VoronoiFVM.ExtendableSparse.PILUAMPreconditioner()]
	@info method_linear
	for strat in strats
		@info rt2(nm, 0.01, nt; method_linear, precon_linear=strat, num)
	end
	
end

function loop3(nm, nt; num=5, method_linear = KrylovJL_GMRES())
	strats = [VoronoiFVM.SparspakFactorization(), VoronoiFVM.ILUZeroPreconditioner(), VoronoiFVM.ExtendableSparse.ILUAMPreconditioner(), VoronoiFVM.ExtendableSparse.PILUAMPreconditioner()]
	@info method_linear
	for strat in strats
		@info rt2(nm, 0.01, nt; method_linear, precon_linear=strat, num)
	end
	
end

function loop4(nm, nt; num=5, method_linear = KrylovJL_GMRES())
	strats = [VoronoiFVM.ILUZeroPreconditioner(), VoronoiFVM.ExtendableSparse.ILUAMPreconditioner(), VoronoiFVM.ExtendableSparse.PILUAMPreconditioner()]
	@info method_linear
	for strat in strats
		@info rt2(nm, 0.01, nt; method_linear, precon_linear=strat, num)
	end
	
end
