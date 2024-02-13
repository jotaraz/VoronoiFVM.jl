function eval_and_assemble_part_para_ESMP(
    system::System{Tv,Tc,Ti,Tm,TSpecMat,TSolArray},
    U::AbstractMatrix{Tv}, # Actual solution iteration
    UOld::AbstractMatrix{Tv}, # Old timestep solution
    F::AbstractMatrix{Tv},# Right hand side
    time,
    tstep,# time step size. Inf means stationary solution
    λ,
    params::AbstractVector;
    edge_cutoff = 0.0) where {Tv,Tc,Ti,Tm,TSpecMat,TSolArray}
    
    if system.assembly_type != :cellwise
    	@warn "Assembly has to be `:cellwise`"
    	return
    end
    _complete!(system) # needed here as well for test function system which does not use newton

    

	# Reset matrix + rhs
    reset!(system.matrix)
    new_ind = system.matrix.new_indices
    cfp     = system.matrix.cellsforpart
    nt      = system.matrix.nt
    depth   = system.matrix.depth
    
    grid = system.grid
    physics = system.physics
    nodes = [Node(system, time, λ, params) for tid=1:nt]
    edges = [Edge(system, time, λ, params) for tid=1:nt]
    nspecies::Int = num_species(system)

    
    F .= 0.0
    nparams::Int = system.num_parameters

    dudp = system.dudp

    for iparam = 1:nparams
        dudp[iparam] .= 0.0
    end

    # Arrays for gathering solution data
    UKs    = [Array{Tv,1}(undef, nspecies + nparams) for tid=1:nt]
    UKOlds = [Array{Tv,1}(undef, nspecies + nparams) for tid=1:nt]
    UKLs   = [Array{Tv,1}(undef, 2 * nspecies + nparams) for tid=1:nt]

    @assert length(params) == nparams
    if nparams > 0
        for tid=1:nt
		    UKs[tid][(nspecies+1):end]    .= params
		    UKOlds[tid][(nspecies+1):end] .= params
		    UKLs[tid][(2*nspecies+1):end] .= params
    	end
    end

    # Inverse of timestep
    # According to Julia documentation, 1/Inf=0 which
    # comes handy to write compact code here for the
    # case of stationary problems.
    tstepinv = 1.0 / tstep
    

    #
    # These wrap the different physics functions.
    #
    src_evaluators     = [ResEvaluator(physics, :source, UKs[i], nodes[i], nspecies) for i=1:nt]
    rea_evaluators     = [ResJacEvaluator(physics, :reaction, UKs[i], nodes[i], nspecies) for i=1:nt]
    stor_evaluators    = [ResJacEvaluator(physics, :storage, UKs[i], nodes[i], nspecies) for i=1:nt]
    oldstor_evaluators = [ResEvaluator(physics, :storage, UKs[i], nodes[i], nspecies) for i=1:nt]
    flux_evaluators    = [ResJacEvaluator(physics, :flux, UKLs[i], edges[i], nspecies) for i=1:nt]
    erea_evaluators    = [ResJacEvaluator(physics, :edgereaction, UKLs[i], edges[i], nspecies) for i=1:nt]
    outflow_evaluators = [ResJacEvaluator(physics, :boutflow, UKLs[i], edges[i], nspecies) for i=1:nt]
    
    
    

    ncalloc = @allocated begin
    	for level=1:depth
    		@threads for tid=1:nt
    			for item in cfp[(level-1)*nt+tid]
    				for inode in noderange(system.assembly_data, item)
						_fill!(nodes[tid], system.assembly_data, inode, item)
						@views UKs[tid][1:nspecies] .= U[:, nodes[tid].index]
						@views UKOlds[tid][1:nspecies] .= UOld[:, nodes[tid].index]

						evaluate!(src_evaluators[tid])
						src = res(src_evaluators[tid])

						evaluate!(rea_evaluators[tid], UKs[tid])
						res_react = res(rea_evaluators[tid])
						jac_react = jac(rea_evaluators[tid])

						evaluate!(stor_evaluators[tid], UKs[tid])
						res_stor = res(stor_evaluators[tid])
						jac_stor = jac(stor_evaluators[tid])

						evaluate!(oldstor_evaluators[tid], UKOlds[tid])
						oldstor = res(oldstor_evaluators[tid])

						@inline function asm_res(idof, ispec)
						    _add(
						        F,
						        idof,
						        nodes[tid].fac * (
						            res_react[ispec] - src[ispec] +
						            (res_stor[ispec] - oldstor[ispec]) * tstepinv
						        ),
						    )
						end

						@inline function asm_jac(idof, jdof, ispec, jspec)
						    _addnz(
						        system.matrix,
						        idof,
						        jdof,
						        tid,
						        jac_react[ispec, jspec] + jac_stor[ispec, jspec] * tstepinv,
						        nodes[tid].fac,
						    )
						end

						@inline function asm_param(idof, ispec, iparam)
						    jparam = nspecies + iparam
						    dudp[iparam][ispec, idof] +=
						        (jac_react[ispec, jparam] + jac_stor[ispec, jparam] * tstepinv) *
						        nodes[tid].fac
						end

						assemble_res_jac_reordered(nodes[tid], system, asm_res, asm_jac, asm_param, new_ind)
					end
				end
			end
		end
		
		for item in cfp[depth*nt+1]
    		for inode in noderange(system.assembly_data, item)
				_fill!(nodes[1], system.assembly_data, inode, item)
				@views UKs[1][1:nspecies] .= U[:, nodes[1].index]
				@views UKOlds[1][1:nspecies] .= UOld[:, nodes[1].index]

				evaluate!(src_evaluators[1])
				src = res(src_evaluators[1])

				evaluate!(rea_evaluators[1], UKs[1])
				res_react = res(rea_evaluators[1])
				jac_react = jac(rea_evaluators[1])

				evaluate!(stor_evaluators[1], UKs[1])
				res_stor = res(stor_evaluators[1])
				jac_stor = jac(stor_evaluators[1])

				evaluate!(oldstor_evaluators[1], UKOlds[1])
				oldstor = res(oldstor_evaluators[1])

				@inline function asm_res(idof, ispec)
					_add(
					    F,
					    idof,
					    nodes[1].fac * (
					        res_react[ispec] - src[ispec] +
					        (res_stor[ispec] - oldstor[ispec]) * tstepinv
					    ),
					)
				end

				@inline function asm_jac(idof, jdof, ispec, jspec)
					_addnz(
					    system.matrix,
					    idof,
					    jdof,
					    1,
					    jac_react[ispec, jspec] + jac_stor[ispec, jspec] * tstepinv,
					    nodes[1].fac,
					)
				end

				@inline function asm_param(idof, ispec, iparam)
					jparam = nspecies + iparam
					dudp[iparam][ispec, idof] +=
					    (jac_react[ispec, jparam] + jac_stor[ispec, jparam] * tstepinv) *
					    nodes[1].fac
				end

				assemble_res_jac_reordered(nodes[1], system, asm_res, asm_jac, asm_param, new_ind)
			end
		end
	end
		
    if isnontrivial(outflow_evaluators[1])
    end


    ncalloc += @allocated begin
    	for level=1:depth
    		@threads for tid=1:nt
    			for item in cfp[(level-1)*nt+tid]
    				for iedge in edgerange(system.assembly_data, item)
						_fill!(edges[tid], system.assembly_data, iedge, item)

						@views UKLs[tid][1:nspecies] .= U[:, edges[tid].node[1]]
						@views UKLs[tid][(nspecies+1):(2*nspecies)] .= U[:, edges[tid].node[2]]

						evaluate!(flux_evaluators[tid], UKLs[tid])
						res_flux = res(flux_evaluators[tid])
						jac_flux = jac(flux_evaluators[tid])

						@inline function asm_res(idofK, idofL, ispec)
						    val = edges[tid].fac * res_flux[ispec]
						    _add(F, idofK, val)
						    _add(F, idofL, -val)
						end

						@inline function asm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
						    _addnz(system.matrix, idofK, jdofK, tid, +jac_flux[ispec, jspec], edges[tid].fac)
						    _addnz(system.matrix, idofL, jdofK, tid, -jac_flux[ispec, jspec], edges[tid].fac)
						    _addnz(
						        system.matrix,
						        idofK,
						        jdofL,
						        tid,
						        +jac_flux[ispec, jspec+nspecies],
						        edges[tid].fac,
						    )
						    _addnz(
						        system.matrix,
						        idofL,
						        jdofL,
						        tid,
						        -jac_flux[ispec, jspec+nspecies],
						        edges[tid].fac,
						    )
						end

						@inline function asm_param(idofK, idofL, ispec, iparam)
						    jparam = 2 * nspecies + iparam
						    dudp[iparam][ispec, idofK] += edges[tid].fac * jac_flux[ispec, jparam]
						    dudp[iparam][ispec, idofL] -= edges[tid].fac * jac_flux[ispec, jparam]
						end

						assemble_res_jac_reordered(edges[tid], system, asm_res, asm_jac, asm_param, new_ind)

						##################################################################################
						if isnontrivial(erea_evaluators[tid])
						    evaluate!(erea_evaluators[tid], UKLs[tid])
						    res_erea = res(erea_evaluators[tid])
						    jac_erea = jac(erea_evaluators[tid])

						    @inline function ereaasm_res(idofK, idofL, ispec)
						        val = edges[tid].fac * res_erea[ispec]
						        _add(F, idofK, val)
						        _add(F, idofL, val)
						    end

						    @inline function ereaasm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
						        _addnz(system.matrix, idofK, jdofK, tid, +jac_erea[ispec, jspec], edges[tid].fac)
						        _addnz(system.matrix, idofL, jdofK, tid, -jac_erea[ispec, jspec], edges[tid].fac)
						        _addnz(
						            system.matrix,
						            idofK,
						            jdofL,
						            tid,
						            -jac_erea[ispec, jspec+nspecies],
						            edges[tid].fac,
						        )
						        _addnz(
						            system.matrix,
						            idofL,
						            jdofL,
						            tid,
						            +jac_erea[ispec, jspec+nspecies],
						            edge.fac,
						        )
						    end

						    @inline function ereaasm_param(idofK, idofL, ispec, iparam)
						        jparam = 2 * nspecies + iparam
						        dudp[iparam][ispec, idofK] += edges[tid].fac * jac_erea[ispec, jparam]
						        dudp[iparam][ispec, idofL] += edges[tid].fac * jac_erea[ispec, jparam]
						    end

						    assemble_res_jac_reordered(edges[tid], system, ereaasm_res, ereaasm_jac, ereaasm_param, new_ind)
						end

						##################################################################################
						if isnontrivial(outflow_evaluators[tid]) && hasoutflownode(edges[tid])
						    outflownode!(edges[tid])
						    evaluate!(outflow_evaluators[tid], UKLs[tid])
						    res_outflow = res(outflow_evaluators[tid])
						    jac_outflow = jac(outflow_evaluators[tid])
						    
						    @inline function outflowasm_res(idofK, idofL, ispec)
						        val = edges[tid].fac * res_outflow[ispec]

						        if isoutflownode(edge,1)
						            _add(F, idofK, val)
						        end
						        
						        if isoutflownode(edge,2)
						            _add(F, idofL, -val)
						        end
						    end

						    @inline function outflowasm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
						        if isoutflownode(edge, 1)
						            _addnz(
						                system.matrix,
						                idofK,
						                jdofK,
						                tid,
						                +jac_outflow[ispec, jspec],
						                edges[tid].fac
						            )
						            _addnz(
						                system.matrix,
						                idofK,
						                jdofL,
						                tid,
						                jac_outflow[ispec, jspec+nspecies],
						                edges[tid].fac
						            )
						        end

						        if isoutflownode(edge, 2)
						            _addnz(
						                system.matrix,
						                idofL,
						                jdofK,
						                tid,
						                -jac_outflow[ispec, jspec],
						                edges[tid].fac
						            )
						            _addnz(
						                system.matrix,
						                idofL,
						                jdofL,
						                tid,
						                -jac_outflow[ispec, jspec+nspecies],
						                edges[tid].fac
						            )
						        end
						    end

						    @inline function outflowasm_param(idofK, idofL, ispec, iparam)
						        jparam = 2 * nspecies + iparam
						        if isoutflownode(edge, 1)
						            dudp[iparam][ispec, idofK] += edges[tid].fac * jac_outflow[ispec, jparam]
						        end
						        if isoutflownode(edge, 2)
						            dudp[iparam][ispec, idofL] += edges[tid].fac * jac_outflow[ispec, jparam]
						        end
						    end

						    assemble_res_jac_reordered(
						        edges[tid],
						        system,
						        outflowasm_res,
						        outflowasm_jac,
						        outflowasm_param,
						        new_ind
						    )
						end
					end
				end
    		end
    	end
    end
    
    for item in cfp[depth*nt+1]
		for iedge in edgerange(system.assembly_data, item)
			_fill!(edges[1], system.assembly_data, iedge, item)

			@views UKLs[1][1:nspecies] .= U[:, edges[1].node[1]]
			@views UKLs[1][(nspecies+1):(2*nspecies)] .= U[:, edges[1].node[2]]

			evaluate!(flux_evaluators[1], UKLs[1])
			res_flux = res(flux_evaluators[1])
			jac_flux = jac(flux_evaluators[1])

			@inline function asm_res(idofK, idofL, ispec)
			    val = edges[1].fac * res_flux[ispec]
			    _add(F, idofK, val)
			    _add(F, idofL, -val)
			end

			@inline function asm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
			    _addnz(system.matrix, idofK, jdofK, 1, +jac_flux[ispec, jspec], edges[1].fac)
			    _addnz(system.matrix, idofL, jdofK, 1, -jac_flux[ispec, jspec], edges[1].fac)
			    _addnz(
			        system.matrix,
			        idofK,
			        jdofL,
			        1,
			        +jac_flux[ispec, jspec+nspecies],
			        edges[1].fac,
			    )
			    _addnz(
			        system.matrix,
			        idofL,
			        jdofL,
			        1,
			        -jac_flux[ispec, jspec+nspecies],
			        edges[1].fac,
			    )
			end

			@inline function asm_param(idofK, idofL, ispec, iparam)
			    jparam = 2 * nspecies + iparam
			    dudp[iparam][ispec, idofK] += edges[1].fac * jac_flux[ispec, jparam]
			    dudp[iparam][ispec, idofL] -= edges[1].fac * jac_flux[ispec, jparam]
			end

			assemble_res_jac_reordered(edges[1], system, asm_res, asm_jac, asm_param, new_ind)

			##################################################################################
			if isnontrivial(erea_evaluators[1])
			    evaluate!(erea_evaluators[1], UKLs[1])
			    res_erea = res(erea_evaluators[1])
			    jac_erea = jac(erea_evaluators[1])

			    @inline function ereaasm_res(idofK, idofL, ispec)
			        val = edges[1].fac * res_erea[ispec]
			        _add(F, idofK, val)
			        _add(F, idofL, val)
			    end

			    @inline function ereaasm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
			        _addnz(system.matrix, idofK, jdofK, 1, +jac_erea[ispec, jspec], edges[1].fac)
			        _addnz(system.matrix, idofL, jdofK, 1, -jac_erea[ispec, jspec], edges[1].fac)
			        _addnz(
			            system.matrix,
			            idofK,
			            jdofL,
			            1,
			            -jac_erea[ispec, jspec+nspecies],
			            edges[1].fac,
			        )
			        _addnz(
			            system.matrix,
			            idofL,
			            jdofL,
			            1,
			            +jac_erea[ispec, jspec+nspecies],
			            edges[1].fac,
			        )
			    end

			    @inline function ereaasm_param(idofK, idofL, ispec, iparam)
			        jparam = 2 * nspecies + iparam
			        dudp[iparam][ispec, idofK] += edges[1].fac * jac_erea[ispec, jparam]
			        dudp[iparam][ispec, idofL] += edges[1].fac * jac_erea[ispec, jparam]
			    end

			    assemble_res_jac_reordered(edges[1], system, ereaasm_res, ereaasm_jac, ereaasm_param, new_ind)
			end

			##################################################################################
			if isnontrivial(outflow_evaluators[1]) && hasoutflownode(edges[1])
			    outflownode!(edges[1])
			    evaluate!(outflow_evaluators[1], UKLs[1])
			    res_outflow = res(outflow_evaluators[1])
			    jac_outflow = jac(outflow_evaluators[1])
			    
			    @inline function outflowasm_res(idofK, idofL, ispec)
			        val = edges[1].fac * res_outflow[ispec]

			        if isoutflownode(edge,1)
			            _add(F, idofK, val)
			        end
			        
			        if isoutflownode(edge,2)
			            _add(F, idofL, -val)
			        end
			    end

			    @inline function outflowasm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
			        if isoutflownode(edge, 1)
			            _addnz(
			                system.matrix,
			                idofK,
			                jdofK,
			                1,
			                +jac_outflow[ispec, jspec],
			                edges[1].fac
			            )
			            _addnz(
			                system.matrix,
			                idofK,
			                jdofL,
			                1,
			                jac_outflow[ispec, jspec+nspecies],
			                edges[1].fac
			            )
			        end

			        if isoutflownode(edge, 2)
			            _addnz(
			                system.matrix,
			                idofL,
			                jdofK,
			                1,
			                -jac_outflow[ispec, jspec],
			                edges[1].fac
			            )
			            _addnz(
			                system.matrix,
			                idofL,
			                jdofL,
			                1,
			                -jac_outflow[ispec, jspec+nspecies],
			                edges[1].fac
			            )
			        end
			    end

			    @inline function outflowasm_param(idofK, idofL, ispec, iparam)
			        jparam = 2 * nspecies + iparam
			        if isoutflownode(edge, 1)
			            dudp[iparam][ispec, idofK] += edges[1].fac * jac_outflow[ispec, jparam]
			        end
			        if isoutflownode(edge, 2)
			            dudp[iparam][ispec, idofL] += edges[1].fac * jac_outflow[ispec, jparam]
			        end
			    end

			    assemble_res_jac_reordered(
			        edges[1],
			        system,
			        outflowasm_res,
			        outflowasm_jac,
			        outflowasm_param,
			        new_ind
			    )
			end
		end
    end
    

    bnode = BNode(system, time, λ, params)
    bedge = BEdge(system, time, λ, params)


    boundary_factors::Array{Tv,2} = system.boundary_factors
    boundary_values::Array{Tv,2} = system.boundary_values
    has_legacy_bc = !iszero(boundary_factors) || !iszero(boundary_values)
	#@info has_legacy_bc

    bsrc_evaluator = ResEvaluator(physics, :bsource, UKs[1], bnode, nspecies)
    brea_evaluator = ResJacEvaluator(physics, :breaction, UKs[1], bnode, nspecies)
    bstor_evaluator = ResJacEvaluator(physics, :bstorage, UKs[1], bnode, nspecies)
    oldbstor_evaluator = ResEvaluator(physics, :bstorage, UKs[1], bnode, nspecies)
    bflux_evaluator = ResJacEvaluator(physics, :bflux, UKLs[1], bedge, nspecies)

	

    nballoc = @allocated for item in nodebatch(system.boundary_assembly_data)
        for ibnode in noderange(system.boundary_assembly_data, item)
            _fill!(bnode, system.boundary_assembly_data, ibnode, item)

            if has_legacy_bc
                # Global index of node
                K = bnode.index
                bnode.Dirichlet = Dirichlet / bnode.fac
                # Assemble "standard" boundary conditions: Robin or
                # Dirichlet
                # valid only for interior species, currently not checked
                for ispec = 1:nspecies
                    idof = dof(F, ispec, K)
                    nidof = new_ind[idof]
                    # If species is present, assemble the boundary condition
                    if idof > 0
                        # Get user specified data
                        boundary_factor = boundary_factors[ispec, bnode.region]
                        boundary_value = boundary_values[ispec, bnode.region]

                        if boundary_factor == Dirichlet
                            # Dirichlet is encoded in the boundary condition factor
                            # Penalty method: scale   (u-boundary_value) by penalty=boundary_factor

                            # Add penalty*boundary_value to right hand side
                            F[ispec, K] += boundary_factor * (U[ispec, K] - boundary_value)

                            # Add penalty to matrix main diagonal (without bnode factor, so penalty
                            # is independent of h)
                            _addnz(system.matrix, nidof, nidof, boundary_factor, 1)
                        else
                            # Robin boundary condition
                            F[ispec, K] +=
                                bnode.fac * (boundary_factor * U[ispec, K] - boundary_value)
                            _addnz(system.matrix, nidof, nidof, boundary_factor, bnode.fac)
                        end
                    end
                end
            end # legacy bc

            # Copy unknown values from solution into dense array
            @views UKs[1][1:nspecies] .= U[:, bnode.index]

            evaluate!(bsrc_evaluator)
            bsrc = res(bsrc_evaluator)

            evaluate!(brea_evaluator, UKs[1])
            res_breact = res(brea_evaluator)
            jac_breact = jac(brea_evaluator)

            asm_res1(idof, ispec) =
                _add(F, idof, bnode.fac * (res_breact[ispec] - bsrc[ispec]))

            asm_jac1(idof, jdof, ispec, jspec) =
                _addnz(system.matrix, idof, jdof, jac_breact[ispec, jspec], bnode.fac)

            asm_param1(idof, ispec, iparam) =
                dudp[iparam][ispec, idof] +=
                    jac_breact[ispec, nspecies+iparam] * bnode.fac

            assemble_res_jac_reordered(bnode, system, asm_res1, asm_jac1, asm_param1, new_ind)

            if isnontrivial(bstor_evaluator)
                evaluate!(bstor_evaluator, UKs[1])
                res_bstor = res(bstor_evaluator)
                jac_bstor = jac(bstor_evaluator)

                @views UKOlds[1] .= UOld[:, bnode.index]
                evaluate!(oldbstor_evaluator, UKOlds[1])
                oldbstor = res(oldbstor_evaluator)

                asm_res2(idof, ispec) = _add(
                    F,
                    idof,
                    bnode.fac * (res_bstor[ispec] - oldbstor[ispec]) * tstepinv,
                )

                function asm_jac2(idof, jdof, ispec, jspec)
                    _addnz(
                        system.matrix,
                        idof,
                        jdof,
                        jac_bstor[ispec, jspec],
                        bnode.fac * tstepinv,
                    )
                end

                function asm_param2(idof, ispec, iparam)
                    dudp[iparam][ispec, idof] +=
                        jac_bstor[ispec, nspecies+iparam] * bnode.fac * tstepinv
                end

                assemble_res_jac_reordered(bnode, system, asm_res2, asm_jac2, asm_param2, new_ind)
            end
        end # ibnode=1:nbn
    end
	
	if isnontrivial(bflux_evaluator)
        nballoc += @allocated for item in edgebatch(system.boundary_assembly_data)
            for ibedge in edgerange(system.boundary_assembly_data, item)
                _fill!(bedge, system.boundary_assembly_data, ibedge, item)
                @views UKLs[1][1:nspecies] .= U[:, bedge.node[1]]
                @views UKLs[1][(nspecies+1):(2*nspecies)] .= U[:, bedge.node[2]]

                evaluate!(bflux_evaluator, UKLs[1])
                res_bflux = res(bflux_evaluator)
                jac_bflux = jac(bflux_evaluator)

                function asm_res(idofK, idofL, ispec)
                    _add(F, idofK, bedge.fac * res_bflux[ispec])
                    _add(F, idofL, -bedge.fac * res_bflux[ispec])
                end

                function asm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
                    _addnz(system.matrix, idofK, jdofK, +jac_bflux[ispec, jspec], bedge.fac)
                    _addnz(system.matrix, idofL, jdofK, -jac_bflux[ispec, jspec], bedge.fac)
                    _addnz(
                        system.matrix,
                        idofK,
                        jdofL,
                        +jac_bflux[ispec, jspec+nspecies],
                        bedge.fac,
                    )
                    _addnz(
                        system.matrix,
                        idofL,
                        jdofL,
                        -jac_bflux[ispec, jspec+nspecies],
                        bedge.fac,
                    )
                end

                function asm_param(idofK, idofL, ispec, iparam)
                    jparam = 2 * nspecies + iparam
                    dudp[iparam][ispec, idofK] += bedge.fac * jac_bflux[ispec, jparam]
                    dudp[iparam][ispec, idofL] -= bedge.fac * jac_bflux[ispec, jparam]
                end
                assemble_res_jac_reordered(bedge, system, asm_res, asm_jac, asm_param, new_ind)
            end
        end
    end
	
	#=
    noallocs(m::ExtendableSparseMatrix) = isnothing(m.lnkmatrix)
    noallocs(m::AbstractMatrix) = false
    # if  no new matrix entries have been created, we should see no allocations
    # in the previous two loops
    neval = 1
    if !noallocs(system.matrix)
        ncalloc = 0
        nballoc = 0
        neval = 0
    end
    =#
    _eval_and_assemble_generic_operator(system, U, F)
    _eval_and_assemble_inactive_species(system, U, UOld, F)

    ncalloc, nballoc
    
end
