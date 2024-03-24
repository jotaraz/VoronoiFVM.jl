function node_asm!(
	system::AbstractSystem,
	node::Node, 
	UK::AbstractVector{Tv}, 
	UKOld::AbstractVector{Tv}, 
	src_evaluator::ResEvaluator, 
	rea_evaluator::ResJacEvaluator, 
	stor_evaluator::ResJacEvaluator, 
	oldstor_evaluator::ResEvaluator,
	cells::AbstractVector{Ti},
	U::AbstractMatrix{Tv}, 
	UOld::AbstractMatrix{Tv}, 
	tid::Integer, 
	nspecies::Integer, 
	tstepinv, 
	F::AbstractMatrix{Tv}, 
	dudp, 
	new_ind::AbstractVector{Ti}) where {Tv, Ti<:Integer}
	for item in cells
		for inode in noderange(system.assembly_data, item)
			_fill!(node, system.assembly_data, inode, item)
			@views UK[1:nspecies] .= U[:, node.index]
			@views UKOld[1:nspecies] .= UOld[:, node.index]

			evaluate!(src_evaluator)
			src = res(src_evaluator)

			evaluate!(rea_evaluator, UK)
			res_react = res(rea_evaluator)
			jac_react = jac(rea_evaluator)

			evaluate!(stor_evaluator, UK)
			res_stor = res(stor_evaluator)
			jac_stor = jac(stor_evaluator)

			evaluate!(oldstor_evaluator, UKOld)
			oldstor = res(oldstor_evaluator)

			@inline function asm_res(idof, ispec)
			    _add(
			        F,
			        idof,
			        node.fac * (
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
			        node.fac,
			    )
			end

			@inline function asm_param(idof, ispec, iparam)
			    jparam = nspecies + iparam
			    dudp[iparam][ispec, idof] +=
			        (jac_react[ispec, jparam] + jac_stor[ispec, jparam] * tstepinv) *
			        node.fac
			end

			assemble_res_jac_reordered(node, system, asm_res, asm_jac, asm_param, new_ind)
		end
	end
end

function node_asm_edgewise03!(
	system, node, F, dudp, U, UOld, UK, UKOld, src_evaluator, rea_evaluator, 
	stor_evaluator, oldstor_evaluator, tstepinv, nspecies, nodes, tid)
	
	ni  = system.matrix.new_indices
	rni = system.matrix.rev_new_indices

	for item0 in nodes # for the first block: 1:k
		item = rni[item0] # but this corresponds to the vertices rni[1]:rni[k]
		for inode in noderange(system.assembly_data, item)
            _fill!(node, system.assembly_data, inode, item)
            @views UK[1:nspecies] .= U[:, node.index]
            @views UKOld[1:nspecies] .= UOld[:, node.index]

            evaluate!(src_evaluator)
            src = res(src_evaluator)

            evaluate!(rea_evaluator, UK)
            res_react = res(rea_evaluator)
            jac_react = jac(rea_evaluator)

            evaluate!(stor_evaluator, UK)
            res_stor = res(stor_evaluator)
            jac_stor = jac(stor_evaluator)

            evaluate!(oldstor_evaluator, UKOld)
            oldstor = res(oldstor_evaluator)

            @inline function asm_res(idof, ispec)
                _add(
                    F,
                    idof,
                    node.fac * (
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
                    node.fac,
                )
            end

            @inline function asm_param(idof, ispec, iparam)
                jparam = nspecies + iparam
                dudp[iparam][ispec, idof] +=
                    (jac_react[ispec, jparam] + jac_stor[ispec, jparam] * tstepinv) *
                    node.fac
            end

            assemble_res_jac_reordered(node, system, asm_res, asm_jac, asm_param, ni)
        end
    end
end


function edge_asm!(
	system::AbstractSystem,
	edge::Edge, 
	UKL::AbstractVector{Tv}, 
	flux_evaluator,
	erea_evaluator, 
	outflow_evaluator,
	cells::AbstractVector{Ti},
	U::AbstractMatrix{Tv}, 
	tid::Integer, 
	nspecies::Integer, 
	tstepinv, 
	F::AbstractMatrix{Tv}, 
	dudp, 
	new_ind::AbstractVector{Ti}) where {Tv, Ti<:Integer}
		
	for item in cells
		for iedge in edgerange(system.assembly_data, item)
			_fill!(edge, system.assembly_data, iedge, item)

			@views UKL[1:nspecies] .= U[:, edge.node[1]]
			@views UKL[(nspecies+1):(2*nspecies)] .= U[:, edge.node[2]]

			evaluate!(flux_evaluator, UKL)
			res_flux = res(flux_evaluator)
			jac_flux = jac(flux_evaluator)

			@inline function asm_res(idofK, idofL, ispec)
			    val = edge.fac * res_flux[ispec]
			    _add(F, idofK, val)
			    _add(F, idofL, -val)
			end

			@inline function asm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
			    _addnz(system.matrix, idofK, jdofK, tid, +jac_flux[ispec, jspec], edge.fac)
			    _addnz(system.matrix, idofL, jdofK, tid, -jac_flux[ispec, jspec], edge.fac)
			    _addnz(
			        system.matrix,
			        idofK,
			        jdofL,
			        tid,
			        +jac_flux[ispec, jspec+nspecies],
			        edge.fac,
			    )
			    _addnz(
			        system.matrix,
			        idofL,
			        jdofL,
			        tid,
			        -jac_flux[ispec, jspec+nspecies],
			        edge.fac,
			    )
			end

			@inline function asm_param(idofK, idofL, ispec, iparam)
			    jparam = 2 * nspecies + iparam
			    dudp[iparam][ispec, idofK] += edge.fac * jac_flux[ispec, jparam]
			    dudp[iparam][ispec, idofL] -= edge.fac * jac_flux[ispec, jparam]
			end

			assemble_res_jac_reordered(edge, system, asm_res, asm_jac, asm_param, new_ind)

			##################################################################################
			if isnontrivial(erea_evaluator)
			    evaluate!(erea_evaluator, UKL)
			    res_erea = res(erea_evaluator)
			    jac_erea = jac(erea_evaluator)

			    @inline function ereaasm_res(idofK, idofL, ispec)
			        val = edge.fac * res_erea[ispec]
			        _add(F, idofK, val)
			        _add(F, idofL, val)
			    end

			    @inline function ereaasm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
			        _addnz(system.matrix, idofK, jdofK, tid, +jac_erea[ispec, jspec], edge.fac)
			        _addnz(system.matrix, idofL, jdofK, tid, -jac_erea[ispec, jspec], edge.fac)
			        _addnz(
			            system.matrix,
			            idofK,
			            jdofL,
			            tid,
			            -jac_erea[ispec, jspec+nspecies],
			            edge.fac,
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
			        dudp[iparam][ispec, idofK] += edge.fac * jac_erea[ispec, jparam]
			        dudp[iparam][ispec, idofL] += edge.fac * jac_erea[ispec, jparam]
			    end

			    assemble_res_jac_reordered(edge, system, ereaasm_res, ereaasm_jac, ereaasm_param, new_ind)
			end

			##################################################################################
			if isnontrivial(outflow_evaluator) && hasoutflownode(edge)
			    outflownode!(edge)
			    evaluate!(outflow_evaluator, UKL)
			    res_outflow = res(outflow_evaluator)
			    jac_outflow = jac(outflow_evaluator)
			    
			    @inline function outflowasm_res(idofK, idofL, ispec)
			        val = edge.fac * res_outflow[ispec]

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
			                edge.fac
			            )
			            _addnz(
			                system.matrix,
			                idofK,
			                jdofL,
			                tid,
			                jac_outflow[ispec, jspec+nspecies],
			                edge.fac
			            )
			        end

			        if isoutflownode(edge, 2)
			            _addnz(
			                system.matrix,
			                idofL,
			                jdofK,
			                tid,
			                -jac_outflow[ispec, jspec],
			                edge.fac
			            )
			            _addnz(
			                system.matrix,
			                idofL,
			                jdofL,
			                tid,
			                -jac_outflow[ispec, jspec+nspecies],
			                edge.fac
			            )
			        end
			    end

			    @inline function outflowasm_param(idofK, idofL, ispec, iparam)
			        jparam = 2 * nspecies + iparam
			        if isoutflownode(edge, 1)
			            dudp[iparam][ispec, idofK] += edge.fac * jac_outflow[ispec, jparam]
			        end
			        if isoutflownode(edge, 2)
			            dudp[iparam][ispec, idofL] += edge.fac * jac_outflow[ispec, jparam]
			        end
			    end

			    assemble_res_jac_reordered(
			        edge,
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


function edge_asm_edgewise03!(system, edge, F, dudp, U, UKL, flux_evaluator, erea_evaluator, outflow_evaluator, nspecies, edges, tid)
	ni = system.matrix.new_indices
	#rni = system.matrix.rev_new_indices

	for item in edges #edgebatch(system.assembly_data)
		#item = rni[item0]
        for iedge in edgerange(system.assembly_data, item)
            _fill!(edge, system.assembly_data, iedge, item)

            @views UKL[1:nspecies] .= U[:, edge.node[1]]
            @views UKL[(nspecies+1):(2*nspecies)] .= U[:, edge.node[2]]

            evaluate!(flux_evaluator, UKL)
            res_flux = res(flux_evaluator)
            jac_flux = jac(flux_evaluator)

            @inline function asm_res(idofK, idofL, ispec)
                val = edge.fac * res_flux[ispec]
                _add(F, idofK, val)
                _add(F, idofL, -val)
            end

            @inline function asm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
                _addnz(system.matrix, idofK, jdofK, tid, +jac_flux[ispec, jspec], edge.fac)
                _addnz(system.matrix, idofL, jdofK, tid, -jac_flux[ispec, jspec], edge.fac)
                _addnz(
                    system.matrix,
                    idofK,
                    jdofL,
					tid,
                    +jac_flux[ispec, jspec+nspecies],
                    edge.fac,
                )
                _addnz(
                    system.matrix,
                    idofL,
                    jdofL,
					tid,
                    -jac_flux[ispec, jspec+nspecies],
                    edge.fac,
                )
            end

            @inline function asm_param(idofK, idofL, ispec, iparam)
                jparam = 2 * nspecies + iparam
                dudp[iparam][ispec, idofK] += edge.fac * jac_flux[ispec, jparam]
                dudp[iparam][ispec, idofL] -= edge.fac * jac_flux[ispec, jparam]
            end

            assemble_res_jac_reordered(edge, system, asm_res, asm_jac, asm_param, ni)

            ##################################################################################
            if isnontrivial(erea_evaluator)
                evaluate!(erea_evaluator, UKL)
                res_erea = res(erea_evaluator)
                jac_erea = jac(erea_evaluator)

                @inline function ereaasm_res(idofK, idofL, ispec)
                    val = edge.fac * res_erea[ispec]
                    _add(F, idofK, val)
                    _add(F, idofL, val)
                end

                @inline function ereaasm_jac(idofK, jdofK, idofL, jdofL, ispec, jspec)
                    _addnz(system.matrix, idofK, jdofK, tid, +jac_erea[ispec, jspec], edge.fac)
                    _addnz(system.matrix, idofL, jdofK, tid, -jac_erea[ispec, jspec], edge.fac)
                    _addnz(
                        system.matrix,
                        idofK,
                        jdofL,
						tid,
                        -jac_erea[ispec, jspec+nspecies],
                        edge.fac,
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
                    dudp[iparam][ispec, idofK] += edge.fac * jac_erea[ispec, jparam]
                    dudp[iparam][ispec, idofL] += edge.fac * jac_erea[ispec, jparam]
                end

                assemble_res_jac_reordered(edge, system, ereaasm_res, ereaasm_jac, ereaasm_param, ni)
            end

            ##################################################################################
            if isnontrivial(outflow_evaluator) && hasoutflownode(edge)
                outflownode!(edge)
                evaluate!(outflow_evaluator, UKL)
                res_outflow = res(outflow_evaluator)
                jac_outflow = jac(outflow_evaluator)
                
                @inline function outflowasm_res(idofK, idofL, ispec)
                    val = edge.fac * res_outflow[ispec]

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
                            edge.fac
                        )
                        _addnz(
                            system.matrix,
                            idofK,
                            jdofL,
							tid,
                            jac_outflow[ispec, jspec+nspecies],
                            edge.fac
                        )
                    end

                    if isoutflownode(edge, 2)
                        _addnz(
                            system.matrix,
                            idofL,
                            jdofK,
							tid,
                            -jac_outflow[ispec, jspec],
                            edge.fac
                        )
                        _addnz(
                            system.matrix,
                            idofL,
                            jdofL,
							tid,
                            -jac_outflow[ispec, jspec+nspecies],
                            edge.fac
                        )
                    end
                end

                @inline function outflowasm_param(idofK, idofL, ispec, iparam)
                    jparam = 2 * nspecies + iparam
                    if isoutflownode(edge, 1)
                        dudp[iparam][ispec, idofK] += edge.fac * jac_outflow[ispec, jparam]
                    end
                    if isoutflownode(edge, 2)
                        dudp[iparam][ispec, idofL] += edge.fac * jac_outflow[ispec, jparam]
                    end
                end

                assemble_res_jac_reordered(
                    edge,
                    system,
                    outflowasm_res,
                    outflowasm_jac,
                    outflowasm_param,
					ni
                )
            end


        end
    end
	
end


function bnode_asm!(
	bnode::BNode,
	system::AbstractSystem,
	UK,
	has_legacy_bc::Bool,
	bsrc_evaluator,
	brea_evaluator,
	bstor_evaluator,
	oldbstor_evaluator,
	U,
	UOld,
	boundary_factors,
	boundary_values,
	F::AbstractMatrix{Tv},
	dudp,
	nspecies,
	new_ind
	) where {Tv}
	
	
	for item in nodebatch(system.boundary_assembly_data)
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
            @views UK[1:nspecies] .= U[:, bnode.index]

            evaluate!(bsrc_evaluator)
            bsrc = res(bsrc_evaluator)

            evaluate!(brea_evaluator, UK)
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
                evaluate!(bstor_evaluator, UK)
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
	

end

function bnode_asm_edgewise03!(
	bnode::BNode,
	system::AbstractSystem,
	UK,
	has_legacy_bc::Bool,
	bsrc_evaluator,
	brea_evaluator,
	bstor_evaluator,
	oldbstor_evaluator,
	U,
	UOld,
	boundary_factors,
	boundary_values,
	F::AbstractMatrix{Tv},
	dudp,
	nspecies
	) where {Tv}
	
	ni = system.matrix.new_indices
	rni = system.matrix.rev_new_indices
	
	for item in nodebatch(system.boundary_assembly_data)
		#item = rni[item0]
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
                    nidof = ni[idof] #new_ind[idof]
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
            @views UK[1:nspecies] .= U[:, bnode.index]

            evaluate!(bsrc_evaluator)
            bsrc = res(bsrc_evaluator)

            evaluate!(brea_evaluator, UK)
            res_breact = res(brea_evaluator)
            jac_breact = jac(brea_evaluator)

            asm_res1(idof, ispec) =
                _add(F, idof, bnode.fac * (res_breact[ispec] - bsrc[ispec]))

            asm_jac1(idof, jdof, ispec, jspec) =
                _addnz(system.matrix, idof, jdof, jac_breact[ispec, jspec], bnode.fac)

            asm_param1(idof, ispec, iparam) =
                dudp[iparam][ispec, idof] +=
                    jac_breact[ispec, nspecies+iparam] * bnode.fac

            assemble_res_jac_reordered(bnode, system, asm_res1, asm_jac1, asm_param1, ni)

            if isnontrivial(bstor_evaluator)
                evaluate!(bstor_evaluator, UK)
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
					
                assemble_res_jac_reordered(bnode, system, asm_res2, asm_jac2, asm_param2, ni)
            end
        end # ibnode=1:nbn
    end
	

end


function bedge_asm!(
	bedge::BEdge,
	system::AbstractSystem,
	UKL,
	bflux_evaluator,
	U,
	F::AbstractMatrix{Tv},
	dudp,
	nspecies,
	new_ind
	) where {Tv}
	
	for item in edgebatch(system.boundary_assembly_data)
    	for ibedge in edgerange(system.boundary_assembly_data, item)
            _fill!(bedge, system.boundary_assembly_data, ibedge, item)
            @views UKL[1:nspecies] .= U[:, bedge.node[1]]
            @views UKL[(nspecies+1):(2*nspecies)] .= U[:, bedge.node[2]]

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


function bedge_asm_edgewise03!(
	bedge::BEdge,
	system::AbstractSystem,
	UKL,
	bflux_evaluator,
	U,
	F::AbstractMatrix{Tv},
	dudp,
	nspecies
	) where {Tv}

	ni = system.matrix.new_indices
	rni = system.matrix.rev_new_indices
	
	for item in edgebatch(system.boundary_assembly_data)
		#item = rni[item0]
    	for ibedge in edgerange(system.boundary_assembly_data, item)
            _fill!(bedge, system.boundary_assembly_data, ibedge, item)
            @views UKL[1:nspecies] .= U[:, bedge.node[1]]
            @views UKL[(nspecies+1):(2*nspecies)] .= U[:, bedge.node[2]]

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
            assemble_res_jac_reordered(bedge, system, asm_res, asm_jac, asm_param, ni)
        end
    end

end



