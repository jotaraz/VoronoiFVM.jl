

using ThreadPinning; pinthreads(:cores)

using LinearAlgebra
using LinearSolve
#using AMGCLWrap
using ExtendableSparse
#using ProfileView
using Base.Threads


BLAS.set_num_threads(nthreads())


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



function full_rt2(nm, dt, nt; depth=2, verbose=false, unknown_storage=:sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_print_allocs=1, do_print_eaa=false, detail_allocs=false, do_print_ts=true, num=5)
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
	
	if nt != 0
		sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; species = [1])
		grid = sys.grid
    else
    	grid = VoronoiFVM.ExtendableSparse.getgrid(nm)
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
    
    #@info precon_linear, typeof(system.matrix)
    
    oldsol = unknowns(sys)
    oldsol .= 0.5
    solution = unknowns(sys)

    if nt != 0
        VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
        
        for i=1:num
            all = @allocated (t = @elapsed VoronoiFVM._solve_timestep_parallel!(solution, oldsol, sys, control, time, dt, empedparam, params; do_print_allocs, do_print_eaa, detail_allocs))
            tasm = sys.history.tasm
            tlin = sys.history.tlinsolve
            if do_print_ts
                @info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | LinSolveTime $(round(tlin, sigdigits=4)) | Allocs $all"
            end
            time += dt	
            oldsol .= solution
        end
    else
        VoronoiFVM._complete!(sys; create_newtonvectors = true)
        
        for i=1:num
            all = @allocated (t = @elapsed VoronoiFVM._solve_timestep!(solution, oldsol, sys, control, time, dt, empedparam, params; do_print_allocs, do_print_eaa))
            tasm = sys.history.tasm
            tlin = sys.history.tlinsolve
            if do_print_ts
                @info ">>> Timestep $i | Runtime $(round(t, sigdigits=4)) | Ass.time $(round(tasm, sigdigits=4)) | Run-Ass $(round(t-tasm, sigdigits=4)) | LinSolveTime $(round(tlin, sigdigits=4)) | Allocs $all"
            end
            time += dt	
            oldsol .= solution
        end
    end

    return oldsol

end	


function benchmark_one(nm, nt, p; num=5, method_linear = KrylovJL_GMRES(), dpa=1, dt=0.01)
	strats = [VoronoiFVM.ILUZeroPreconditioner(), VoronoiFVM.ExtendableSparse.ILUAMPreconditioner(), VoronoiFVM.ExtendableSparse.PILUAMPreconditioner()]
	#@info method_linear
	full_rt2(nm, dt, nt; method_linear, precon_linear=strats[p], num, do_print_allocs=dpa)
	
end

function validate(nm, nt, p; num=5, dt=.01, method_linear = KrylovJL_GMRES(), dpa=1, nt0=0, p0=1)
    strats = [VoronoiFVM.ILUZeroPreconditioner(), VoronoiFVM.ExtendableSparse.ILUAMPreconditioner(), VoronoiFVM.ExtendableSparse.PILUAMPreconditioner()]
	
    seq_res = full_rt2(nm, dt, nt0; method_linear, precon_linear=strats[p0], num, do_print_allocs=dpa)
    par_res = full_rt2(nm, dt, nt; method_linear, precon_linear=strats[p], num, do_print_allocs=dpa)

    maximum(abs.(seq_res-par_res))

end


function loop_full_now(nm, nt, p; num=5, method_linear = KrylovJL_GMRES(), dpa=1)
	strats = [VoronoiFVM.ILUZeroPreconditioner(), VoronoiFVM.ExtendableSparse.ILUAMPreconditioner(), VoronoiFVM.ExtendableSparse.PILUAMPreconditioner()]
	#@info method_linear
	full_rt2(nm, 0.01, nt; method_linear, precon_linear=strats[p], num, do_print_allocs=dpa)
	
end

function test(; tol=1e-8, nt=nothing, nm=(300,300), dt=.01, num=5, do_print_ts=true)
    if nt === nothing
        nt = nthreads()
    end
    method_linear = KrylovJL_GMRES()
    strats = [VoronoiFVM.ILUZeroPreconditioner(), VoronoiFVM.ExtendableSparse.ILUAMPreconditioner(), VoronoiFVM.ExtendableSparse.PILUAMPreconditioner()]
	
    nts = [0, nt, nt, nt]
    ps  = [1, 1, 2, 3]
    numcases = length(nts)
    results = [full_rt2(nm, dt, nts[i]; method_linear, precon_linear=strats[ps[i]], num, do_print_allocs=0, do_print_ts) for i=1:numcases]

    diffs = [maximum(abs.(results[i]-results[i+1])) for i=1:numcases-1]
    
    if maximum(diffs) < tol
        @warn "Test successful, all differences < $tol"
    else
        @warn "Test not successful: $(round.(diffs,sigdigits=3)), where #threads = $nts, and preconditioners = ILUZero, ILUZero, ILUAM, PILUAM"
    end
end