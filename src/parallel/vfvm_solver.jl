#
# See
# https://discourse.julialang.org/t/is-it-possible-to-detect-if-julia-is-ahead-of-time-precompiling/78631
#
is_precompiling() = ccall(:jl_generating_output, Cint, ()) == 1

"""
$(SIGNATURES)

Solve time step problem. This is the core routine
for implicit Euler and stationary solve.
"""
function _solve_timestep!(solution::AbstractMatrix{Tv}, # old time step solution resp. initial value
                          oldsol::AbstractMatrix{Tv}, # old time step solution resp. initial value
                          system::AbstractSystem{Tv, Tc, Ti, Tm}, # Finite volume system
                          control::SolverControl,
                          time,
                          tstep,
                          embedparam,
                          params;
                          called_from_API = false,) where {Tv, Tc, Ti, Tm}
    _complete!(system; create_newtonvectors = true)
    nlhistory = NewtonSolverHistory()
    tasm = 0.0
    tlinsolve = 0.0
    t = @elapsed begin
        solution .= oldsol
        residual = system.residual
        update = system.update
        _initialize!(solution, system; time, λ = embedparam, params)

        method_linear = system.matrixtype == :sparse ? control.method_linear : nothing;
        if isnothing(method_linear) &&  system.matrixtype == :sparse
            if Tv != Float64
                method_linear = SparspakFactorization()
            elseif dim_space(system.grid)==1
                method_linear = KLUFactorization()
            elseif dim_space(system.grid)==2
                method_linear = SparspakFactorization()
            else
                method_linear = UMFPACKFactorization() # seems to do the best pivoting
            end
        end

        oldnorm = 1.0
        converged = false
        damp = 1.0
        if !system.is_linear
            if doprint(control, 'n')
                println("\n  [n]ewton: #it(lin)  |update| cont3tion   |round| #rd")
            end
            damp = control.damp_initial
            rnorm = control.rnorm(solution)
        end

        nlu_reuse = 0
        nround = 0
        tolx = 0.0
        ncalloc = 0
        nballoc = 0
        neval = 0
        niter = 1

        while niter <= control.maxiters
            # Create Jacobi matrix and RHS for Newton iteration
            try
                tasm += @elapsed nca, nba, nev = eval_and_assemble(system,
                                                                   solution,
                                                                   oldsol,
                                                                   residual,
                                                                   time,
                                                                   tstep,
                                                                   embedparam,
                                                                   params;
                                                                   edge_cutoff = control.edge_cutoff,)
                ncalloc += nca
                nballoc += nba
                neval += nev
            catch err
                if (control.handle_exceptions)
                    _print_error(err, stacktrace(catch_backtrace()))
                    throw(AssemblyError())
                else
                    rethrow(err)
                end
            end

            tlinsolve += @elapsed _solve_linear!(values(update),
                                                 system,
                                                 nlhistory,
                                                 control,
                                                 method_linear,
                                                 system.matrix,
                                                 values(residual))

            values(solution) .-= damp * values(update)

            # "incremental collection may only sweep   so-called young objects"
            GC.gc(false)

            if system.is_linear
                converged = true
                break
            end

            damp = min(damp * control.damp_growth, 1.0)
            norm = control.unorm(update)
            if tolx == 0.0
                tolx = norm * control.reltol
            end
            dnorm = 1.0
            rnorm_new = control.rnorm(solution)
            if rnorm > 1.0e-50
                dnorm = abs((rnorm - rnorm_new) / rnorm)
            end

            if dnorm < control.tol_round
                nround = nround + 1
            else
                nround = 0
            end

            if control.log
                push!(nlhistory.l1normdiff, dnorm)
                push!(nlhistory.updatenorm, norm)
            end
            if doprint(control, 'n')
                if control.reltol_linear < 1.0
                    itstring = @sprintf("  [n]ewton: % 3d(% 3d)", niter, nlhistory.nlin)
                else
                    itstring = @sprintf("it=% 3d", niter)
                end
                if control.max_round > 0
                    @printf("%s %.3e %.3e %.3e % 2d\n",
                            itstring,
                            norm,
                            norm/oldnorm,
                            dnorm,
                            nround)
                else
                    @printf("%s %.3e %.3e\n", itstring, norm, norm/oldnorm)
                end
            end
            if niter > 1 && norm / oldnorm > 1.0 / control.tol_mono
                converged = false
                break
            end

            if norm < control.abstol || norm < tolx
                converged = true
                break
            end
            oldnorm = norm
            rnorm = rnorm_new

            if nround > control.max_round
                converged = true
                break
            end
            niter = niter + 1
        end
        if !converged
            throw(ConvergenceError())
        end
    end
    if control.log
        nlhistory.time = t
        nlhistory.tlinsolve = tlinsolve
        nlhistory.tasm = tasm
    end

    if ncalloc + nballoc > 0 && doprint(control, 'a') && !is_precompiling()
        @warn "[a]llocations in assembly loop: cells: $(ncalloc÷neval), bfaces: $(nballoc÷neval)"
    end

    if doprint(control, 'n') && !system.is_linear
        println("  [n]ewton: $(round(t,sigdigits=3)) seconds asm: $(round(100*tasm/t,sigdigits=3))%, linsolve: $(round(100*tlinsolve/t,sigdigits=3))%")
    end

    if doprint(control, 'l') && system.is_linear
        println("  [l]inear($(nameof(typeof(method_linear)))): $(round(t,sigdigits=3)) seconds")
    end

    system.history = nlhistory
end

################################################################
"""
````
solve!(solution, inival, system; 
    control=SolverControl(), 
    tstep=Inf)
````
Mutating version of [`solve(inival,system)`](@ref)
"""
function VoronoiFVM.solve!(solution, # Solution
                           inival,   # Initial value 
                           system::VoronoiFVM.AbstractSystem;     # Finite volume system
                           control = SolverControl(),      # Newton solver control information
                           time = Inf,
                           tstep = Inf,                # Time step size. Inf means  stationary solution
                           embedparam = 0.0,
                           params = zeros(0),
                           called_from_API = false,)
    fix_deprecations!(control)
    if !called_from_API && doprint(control, 'd')
        @warn "[d]eprecated: solve(inival,solution, system; kwargs...)"
    end
    _solve_timestep!(solution,
                     inival,
                     system,
                     control,
                     time,
                     tstep,
                     embedparam,
                     params;
                     called_from_API = true,)
    return solution
end

################################################################
"""
````
    solve(inival, system; control=SolverControl(),params, tstep=Inf)
````
Alias for [`solve(system::VoronoiFVM.AbstractSystem; kwargs...)`](@ref) with the corresponding keyword arguments.

Solve stationary problem(if `tstep==Inf`) or one step implicit Euler step using Newton's method with `inival` as initial
value. Returns a solution array.
"""
function CommonSolve.solve(inival,   # Initial value 
                           system::AbstractSystem;     # Finite volume system
                           control = SolverControl(),      # Newton solver control information
                           time = Inf,
                           tstep = Inf,                # Time step size. Inf means  stationary solution
                           params = zeros(0),
                           called_from_API = false,)
    fix_deprecations!(control)
    if !called_from_API && doprint(control, 'd')
        @warn "[d]eprecated: solve(inival,system; kwargs...)"
    end

    solve!(unknowns(system),
           inival,
           system;
           control = control,
           time = time,
           tstep = tstep,
           params = params,
           called_from_API = true,)
end

"""
        solve(inival, system, times; kwargs...)

Alias for [`solve(system::VoronoiFVM.AbstractSystem; kwargs...)`](@ref) with the corresponding keyword arguments.

"""
function CommonSolve.solve(inival,
                           system::VoronoiFVM.AbstractSystem,
                           lambdas;
                           control = SolverControl(),
                           transient = true, # choose between transient and stationary (embedding) case
                           time = 0.0,
                           params = zeros(0),
                           called_from_API = false,
                           kwargs...,)
    fix_deprecations!(control)
    if !called_from_API && doprint(control, 'd')
        @warn "[d]eprecated: solve(inival,system,times;kwargs...)"
    end
    # rounding in output
    rd(x) = round(x; sigdigits = 5)

    # Set initial value of Δλ etc
    if transient # λ is time
        λstr = "t"
        Δλ = control.Δt
        Δλ_min = control.Δt_min
        Δλ_max = control.Δt_max
        Δλ_grow = control.Δt_grow
        Δλ_decrease = control.Δt_decrease
    else  # λ is embedding parameter
        λstr = "p"
        Δλ = control.Δp
        Δλ_min = control.Δp_min
        Δλ_max = control.Δp_max
        Δλ_grow = control.Δp_grow
        Δλ_decrease = control.Δp_decrease
    end
    Δu_opt = control.Δu_opt
    Δu_max_factor = control.Δu_max_factor

    allhistory = TransientSolverHistory()

    solution = copy(inival)
    oldsolution = copy(inival) # we need a copy as it is later overwritten

    # Initialize Dirichlet boundary values
    _initialize_dirichlet!(solution, system; time, λ = Float64(lambdas[1]), params)

    # If not transient, solve for first embedding lambdas[1]
    t0 = @elapsed if !transient
        control.pre(solution, Float64(lambdas[1]))
        solution = solve!(solution,
                          oldsolution,
                          system;
                          called_from_API = true,
                          control = control,
                          time = time,
                          tstep = Inf,
                          embedparam = Float64(lambdas[1]),
                          params = params,
                          kwargs...,)

        control.post(solution, oldsolution, lambdas[1], 0)

        if control.log
            push!(allhistory, system.history)
            push!(allhistory.times, lambdas[1])
            Δu = control.delta(system, solution, oldsolution, lambdas[1], 0)
            push!(allhistory.updates, Δu)
        end
        oldsolution .= solution
    end

    # Initialize transient solution struct
    tsol = TransientSolution(Float64(lambdas[1]), solution; in_memory = control.in_memory)

    if doprint(control, 'e')
        println("[e]volution: start in $(extrema(lambdas))")
    end

    λ0 = 0
    istep = 0
    solved = false
    # Outer loop over embedding params/ time values
    t1 = @elapsed for i = 1:(length(lambdas) - 1)
        Δλ = max(Δλ, Δλ_min)
        λstart = lambdas[i]
        λend = lambdas[i + 1]
        λ = Float64(λstart)

        # Inner loop between two embedding params/time values
        while λ < λend
            solved = false
            λ0 = λ
            Δu = 0.0

            # Try to solve, possibly with stepsize decrease
            while !solved
                solved = true
                forced = false
                errored = false
                try # check for non-converging newton
                    λ = λ0 + Δλ
                    control.pre(solution, λ)

                    if transient # λ is time
                        _time = λ
                        _tstep = Δλ
                        _embedparam = 0.0
                    else # λ is embedding parameter
                        _time = time
                        _tstep = Inf
                        _embedparam = λ
                    end

                    solution = solve!(solution,
                                      oldsolution,
                                      system;
                                      called_from_API = true,
                                      control = control,
                                      time = _time,
                                      tstep = _tstep,
                                      embedparam = _embedparam,
                                      params = params,
                                      kwargs...,)
                catch err
                    err = "Problem at $(λstr)=$(λ|>rd), Δ$(λstr)=$(Δλ|>rd):\n$(err)"
                    if (control.handle_exceptions)
                        _print_error(err, stacktrace(catch_backtrace()))
                    else
                        rethrow(err)
                    end
                    solved = false
                    errored = true
                end
                if solved
                    Δu = control.delta(system, solution, oldsolution, λ, Δλ)
                    if Δu > Δu_max_factor * Δu_opt
                        solved = false
                    end
                end
                if !solved
                    if Δλ ≈ Δλ_min
                        if !(control.force_first_step && istep == 0)
                            err = """
    At $(λstr)=$(λ|>rd): Δ$(λstr)_min=$(Δλ_min|>rd) reached while Δu/Δu_opt=$(Δu/Δu_opt|>rd).
    Returning prematurely before $(λstr)[end]=$(lambdas[end]|>rd) 
    """
                            if control.handle_exceptions
                                @warn err
                            else
                                throw(ErrorException(err))
                            end
                            break # give up lowering stepsize, break out if "while !solved" loop
                        elseif !errored
                            if doprint(control, 'e')
                                println("[e]volution:  forced first timestep: Δu/Δu_opt=$(Δu/Δu_opt|>rd)")
                            end
                            forced = true
                            solved = true
                        else
                            err = "Convergence problem in first timestep"
                            if control.handle_exceptions
                                @warn err
                            else
                                throw(ErrorException(err))
                            end

                            break
                        end
                    else
                        # reduce time step 
                        Δλ = max(Δλ_min, Δλ * Δλ_decrease)
                        if doprint(control, 'e')
                            @printf("[e]volution:  Δu/Δu_opt=%.3e => retry: Δ%s=%.3e\n", Δu/Δu_opt, λstr, Δλ)
                        end
                    end
                end
            end # while !solved

            # Advance step
            if solved
                istep = istep + 1
                if doprint(control, 'e')
                    @printf("[e]volution: step=%d %s=%.3e Δ%s=%.3e Δu=%.3e\n",
                            istep,
                            λstr,
                            λ,
                            λstr,
                            Δλ,
                            Δu)
                end
                if control.log
                    push!(allhistory, system.history)
                    push!(allhistory.updates, Δu)
                    push!(allhistory.times, λ)
                end
                if control.store_all
                    append!(tsol, λ, solution)
                end
                control.post(solution, oldsolution, λ, Δλ)
                oldsolution .= solution

                # Adjust last timesteps, so there will be no accidental
                # very small timestep
                steps_to_go = ceil((λend - λ) / Δλ)
                λ_predict = λend - λ
                if steps_to_go < control.num_final_steps
                    λ_predict = (λend - λ) / steps_to_go
                end

                # see fixed_timesteps!()
                if Δλ_max ≈ Δλ_min
                    λ_predict = Δλ_max
                end

                if λ < λend
                    #  ### account for close last timestep
                    Δλ = min(Δλ_max,
                             Δλ * Δλ_grow,
                             Δλ * Δu_opt / (Δu + 1.0e-14),
                             λ_predict,
                             λend - λ)
                end
            else
                break # break out of inner loop overt timestep
            end # if solved            
        end # while λ<λ_end

        if !control.store_all # store last solution obtained
            append!(tsol, λ0, solution)
        end

        control.sample(solution, λ0)

        if solved
            if !(λ ≈ lambdas[i + 1]) # check end of interval has been reached in inner loop
                @warn "λ=$(λ), lambdas[i+1]=$(lambdas[i+1])"
            end
        else
            break # emergency exit
        end
    end # for i = 1:(length(lambdas)-1), end outer loop

    if doprint(control, 'e')
        println("[e]volution:  $(round(t0+t1,sigdigits=3)) seconds")
    end

    system.history = allhistory
    tsol.history = allhistory
    return tsol
end

"""
    evaluate_residual_and_jacobian(system,u;
                                   t=0.0, tstep=Inf,embed=0.0)

Evaluate residual and jacobian at solution value u.
Returns a solution vector containing a copy of  residual, and an ExendableSparseMatrix
containing a copy of the linearization at u.

"""
function evaluate_residual_and_jacobian(sys, u; t = 0.0, tstep = Inf, embed = 0.0)
    _complete!(sys; create_newtonvectors = true)

    eval_and_assemble(sys, u, u, sys.residual, t, tstep, embed, zeros(0))
    copy(sys.residual), copy(flush!(sys.matrix))
end

module NoModule end

#####################################################################
"""
    solve(system; kwargs...)
    
Built-in solution method for [`VoronoiFVM.System`](@ref).  
    
Keyword arguments:
- General for all solvers 
   - `inival` (default: 0) : Array created via [`unknowns`](@ref) or  number giving the initial value.
   - `control` (default: nothing): Pass instance of [`SolverControl`](@ref)
   -  All elements of [`SolverControl`](@ref) can be used as kwargs. Eventually overwrites values given via `control`
   - `params`: Parameters (Parameter handling is experimental and may change)
    
- __Stationary solver__:
  Invoked if neither `times` nor `embed`, nor `tstep` are given as keyword argument.
  - `time` (default: `0.0`): Set time value.
  Returns a [`DenseSolutionArray`](@ref) or [`SparseSolutionArray`](@ref)

- __Embedding (homotopy) solver__: Invoked if `embed` kwarg is given.
  Use homotopy embedding + damped Newton's method  to 
  solve stationary problem or to solve series of parameter dependent problems.
  Parameter step control is performed according to solver control data.  kwargs and default values are:
  - `embed` (default: `nothing` ): vector of parameter values to be reached exactly
  In addition,  all kwargs of the implicit Euler solver (besides `times`) are handled.  
  Returns a transient solution object `sol` containing the stored solution(s),  see [`TransientSolution`](@ref).
  
- __Implicit Euler transient solver__: Invoked if `times` kwarg is given.
  Use implicit Euler method  + damped   Newton's method  to 
  solve time dependent problem. Time step control is performed
  according to solver control data.  kwargs and default values are:
  - `times` (default: `nothing` ): vector of time values to be reached exactly
  - `pre` (default: `(sol,t)->nothing` ):  callback invoked before each time step
  - `post`  (default:  `(sol,oldsol, t, Δt)->nothing` ): callback invoked after each time step
  - `sample` (default:  `(sol,t)->nothing` ): callback invoked after timestep for all times in `times[2:end]`.
  - `delta` (default:  `(system, u,v,t, Δt)->norm(sys,u-v,Inf)` ):  Value  used to control the time step size `Δu`
  If `control.handle_error` is true, if time step solution  throws an error,
  stepsize  is lowered, and  step solution is called again with a smaller time value.
  If `control.Δt<control.Δt_min`, solution is aborted with error.
  Returns a transient solution object `sol` containing the stored solution,  see [`TransientSolution`](@ref).
  
- __Implicit Euler timestep solver__.  Invoked if `tstep` kwarg is given. Solve one time step of the implicit Euler method.
  - `time` (default: `0`): Set time value. 
  - `tstep`: time step
  Returns a [`DenseSolutionArray`](@ref) or [`SparseSolutionArray`](@ref)
"""
function CommonSolve.solve(sys::VoronoiFVM.AbstractSystem;
                           inival = 0,
                           params = zeros(0),
                           control = VoronoiFVM.SolverControl(),
                           time = 0.0,
                           tstep = Inf,
                           kwargs...,)
    fix_deprecations!(control)

    if isa(inival, Number)
        inival = unknowns(sys; inival = inival)
    elseif !VoronoiFVM.isunknownsof(inival, sys)
        @error "wrong type of inival: $(typeof(inival))"
    end

    for pair in kwargs
        if first(pair) != :times &&
           first(pair) != :embed &&
           hasfield(SolverControl, first(pair))
            setproperty!(control, first(pair), last(pair))
        end
    end

    sys.linear_cache = nothing

    if haskey(kwargs, :times) && !isnothing(kwargs[:times])
        solve(inival,
              sys,
              kwargs[:times];
              control,
              transient = true,
              params,
              time = kwargs[:times][1],
              called_from_API = true,)
    elseif haskey(kwargs, :embed) && !isnothing(kwargs[:embed])
        solve(inival,
              sys,
              kwargs[:embed];
              called_from_API = true,
              transient = false,
              control,
              params,
              time,)
    else
        solve(inival, sys; called_from_API = true, control, params, time, tstep)
    end
end
