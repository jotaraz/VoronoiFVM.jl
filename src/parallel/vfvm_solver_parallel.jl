function _solve_timestep_parallel!(solution::AbstractMatrix{Tv}, # old time step solution resp. initial value
                          oldsol::AbstractMatrix{Tv}, # old time step solution resp. initial value
                          system::AbstractSystem{Tv, Tc, Ti, Tm}, # Finite volume system
                          control::SolverControl,
                          time,
                          tstep,
                          embedparam,
                          params;
                          called_from_API = false, do_print_allocs=1, do_print_eaa=false, detail_allocs=false) where {Tv, Tc, Ti, Tm}
    allo1 = 0 #@allocated (_complete_nomatrix!(system; create_newtonvectors = true))
    allo2 = @allocated (nlhistory = NewtonSolverHistory())
    tasm = 0.0
    dtasm = 0.0
    tlinsolve = 0.0
    t = 0
    allo3 = @allocated begin
		    solution .= oldsol
		    residual = system.residual
		    update   = system.update
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
		    nasm = 0
		    ncalloc = 0
		    nballoc = 0
		    nfalloc = 0
		    nreorder = 0
		    nlinsolve = 0
		    nGC = 0
		    nconv = 0
		    neval = 0
		    niter = 1
		end
	
	allo4 = @allocated begin
	    while niter <= control.maxiters
	        # Create Jacobi matrix and RHS for Newton iteration
	        try
	        	nasm += @allocated (dtasm = @elapsed nca, nba, nev, nfa = eval_and_assemble_part_para_ESMP_fcts(system,
	                                                               solution,
	                                                               oldsol,
	                                                               residual,
	                                                               time,
	                                                               tstep,
	                                                               embedparam,
	                                                               params;
	                                                               edge_cutoff = control.edge_cutoff,
	                                                               detail_allocs))
	            tasm += dtasm
	            ncalloc += nca
	            nballoc += nba
	            nfalloc += nfa
	            neval += nev
	        catch err
	            if (control.handle_exceptions)
	                _print_error(err, stacktrace(catch_backtrace()))
	                throw(AssemblyError())
	            else
	                rethrow(err)
	            end
	        end

			#@info size(values(update)), size(values(residual))
	        nlinsolve += @allocated (tlinsolve += @elapsed _solve_linear!(values(update),
	                                             system,
	                                             nlhistory,
	                                             control,
	                                             method_linear,
	                                             system.matrix,
	                                             values(residual)))
			
		    nreorder += @allocated (update = reorder(update, system.matrix.rev_new_indices))
	        values(solution) .-= damp * values(update)
	        # "incremental collection may only sweep   so-called young objects"
	        nGC += @allocated GC.gc(false)

	        if system.is_linear
	            converged = true
	            break
	        end
			
			nconv += @allocated begin
			    damp = min(damp * control.damp_growth, 1.0)
			    norm = control.unorm(update)
			    if do_print_eaa
			    	@info "tasm $(round(dtasm, sigdigits=4)), time $time, niter $niter, norm $(round(norm, sigdigits=4))" 
			    end
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
	
	if do_print_allocs == 0
	
	elseif do_print_allocs == 1
		if ncalloc + nballoc > 0 && doprint(control, 'a') && !is_precompiling()
			@warn "[a]llocations in assembly loop: cells: $(ncalloc÷neval), bfaces: $(nballoc÷neval)"
		    @info "linsolve $(nlinsolve÷neval), flush $(nfalloc÷neval), tot.asm: $(nasm÷neval), conv: $(nconv÷neval), reord.: $(nreorder÷neval), nGC $nGC"
		    @info "history $allo2, pre $allo3, loop $allo4 | #rounds $neval"
		    #@warn "[a]llocations in assembly loop: $allo2 $allo3 $allo4 | $neval"
		    #@info "cells: $(ncalloc÷neval), bfaces: $(nballoc÷neval), asm(tot): $(nasm÷neval), convergence: $(nconv÷neval), reord.: $(nreorder÷neval), nGC $nGC, linsolve $(nlinsolve÷neval)"
		end
	else
		@warn "[a]llocations in assembly loop: cells: $(ncalloc÷neval), bfaces: $(nballoc÷neval)"
		@info "linsolve $(nlinsolve÷neval), flush $(nfalloc÷neval), tot.asm: $(nasm÷neval), conv: $(nconv÷neval), reord.: $(nreorder÷neval), nGC $nGC"
	    @info "history $allo2, pre $allo3, loop $allo4 | #rounds $neval"
		#@warn "[a]llocations in assembly loop: $allo2 $allo3 $allo4 | $neval"
		#@info "cells: $(ncalloc÷neval), bfaces: $(nballoc÷neval), asm(tot): $(nasm÷neval), convergence: $(nconv÷neval), reord.: $(nreorder÷neval), nGC $nGC, linsolve $(nlinsolve÷neval)"
	end

    if doprint(control, 'n') && !system.is_linear
        println("  [n]ewton: $(round(t,sigdigits=3)) seconds asm: $(round(100*tasm/t,sigdigits=3))%, linsolve: $(round(100*tlinsolve/t,sigdigits=3))%")
    end

    if doprint(control, 'l') && system.is_linear
        println("  [l]inear($(nameof(typeof(method_linear)))): $(round(t,sigdigits=3)) seconds")
    end

    system.history = nlhistory
    #@info allo1, allo2, allo3
end

function reorder(x, ni)
	y = copy(x)
	for i=1:length(x)
		y[ni[i]] = x[i]
	end
	y

end
