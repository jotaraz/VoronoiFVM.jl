function eval_and_assemble_part_para_ESMP_sepa(
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
    #reset!(system.matrix)
    system.matrix.cscmatrix.nzval .= 0
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
    
    
    

    	
	nodesepaass = 0
	nodesepafct = 0
	nodesepapre = 0
	nodesepaalloc = @allocated begin
		@threads for tid=1:1
		for item in cfp[depth*nt+1]
    		for inode in noderange(system.assembly_data, item)
    			nodesepapre += @allocated begin
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
				end
				nodesepafct += @allocated begin
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
				end

				nodesepaass += @allocated assemble_res_jac_reordered(nodes[1], system, asm_res, asm_jac, asm_param, new_ind)
			end
		end
		end
	end
	
	nnzCSC, nnzLNK = ExtendableSparseParallel.nnz_noflush(system.matrix)
	
	if nnzCSC > 0 && nnzLNK > 0
		VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(system.matrix; do_dense=false)
		#sparse flush
	elseif nnzCSC == 0 && nnzLNK > 0
		VoronoiFVM.ExtendableSparseParallel.ESMP_flush!(system.matrix; do_dense=true)
		#dense flush
	end 
    	
	
	nnzCSC1, nnzLNK1 = ExtendableSparseParallel.nnz_noflush(system.matrix)
	@info "$nodesepaalloc ($nodesepaass | $nodesepapre | $nodesepafct)"
	@info "pre-flush : nnzCSC $nnzCSC, nnzLNK $nnzLNK | post-flush: nnzCSC $nnzCSC1, nnzLNK $nnzLNK1"
	
	#@info system.matrix.cscmatrix.colptr[1:5]'
	#@info system.matrix.cscmatrix.rowval[1:10]'
	
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
	#ncalloc = nodealloc+nodesepaalloc+edgealloc
    0, 0, 1
    
end

