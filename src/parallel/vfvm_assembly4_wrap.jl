function eval_and_assemble_part_para_ESMP_fcts(
    system::System{Tv,Tc,Ti,Tm,TSpecMat,TSolArray},
    U::AbstractMatrix{Tv}, # Actual solution iteration
    UOld::AbstractMatrix{Tv}, # Old timestep solution
    F::AbstractMatrix{Tv},# Right hand side
    time,
    tstep,# time step size. Inf means stationary solution
    λ,
    params::AbstractVector;
    edge_cutoff = 0.0, detail_allocs=false) where {Tv,Tc,Ti,Tm,TSpecMat,TSolArray}
    
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
    
    
    
	nodealloc = zeros(Int64, nt)
	nodealloc2 = zeros(Int64, nt)
	for level=1:depth
		@threads for tid=1:nt
			n = node_asm_count!(
				system, nodes[tid], UKs[tid], UKOlds[tid], src_evaluators[tid], rea_evaluators[tid], stor_evaluators[tid], oldstor_evaluators[tid],
				cfp[(level-1)*nt+tid], U, UOld, tid, nspecies, tstepinv, F, dudp, new_ind
			)
			nodealloc[tid] += n[1]
			nodealloc2[tid] += n[2]
			#nodealloc[tid] += @allocated node_asm!(
			#	system, nodes[tid], UKs[tid], UKOlds[tid], src_evaluators[tid], rea_evaluators[tid], stor_evaluators[tid], oldstor_evaluators[tid],
			#	cfp[(level-1)*nt+tid], U, UOld, tid, nspecies, tstepinv, F, dudp, new_ind
			#)
		end
	end
	
	nodesepaalloc = 0
	@threads for tid=1:1
    nodesepaalloc += @allocated node_asm!(
				system, nodes[1], UKs[1], UKOlds[1], src_evaluators[1], rea_evaluators[1], stor_evaluators[1], oldstor_evaluators[1],
				cfp[depth*nt+1], U, UOld, 1, nspecies, tstepinv, F, dudp, new_ind
			)
	end

		
    if isnontrivial(outflow_evaluators[1])
    end


    edgealloc = zeros(Int64, nt) #@allocated begin
    edgealloc2 = zeros(Int64, nt)
	for level=1:depth
		@threads for tid=1:nt
			n = edge_asm_count!(
				system, edges[tid], UKLs[tid], flux_evaluators[tid], erea_evaluators[tid], outflow_evaluators[tid],
				cfp[(level-1)*nt+tid], U, tid, nspecies, tstepinv, F, dudp, new_ind
			)
			edgealloc[tid] += n[1]
			edgealloc2[tid] += n[2]
			
			#edgealloc[tid] += @allocated edge_asm!(
			#	system, edges[tid], UKLs[tid], flux_evaluators[tid], erea_evaluators[tid], outflow_evaluators[tid],
			#	cfp[(level-1)*nt+tid], U, tid, nspecies, tstepinv, F, dudp, new_ind
			#)
				
    	end
    end
    
    @threads for tid=1:1
    edgealloc[1] += @allocated edge_asm!(
				system, edges[1], UKLs[1], flux_evaluators[1], erea_evaluators[1], outflow_evaluators[1],
				cfp[depth*nt+1], U, 1, nspecies, tstepinv, F, dudp, new_ind
	)
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

	

    nballoc = 0
    nballoc += @allocated bnode_asm!(bnode, system, UKs[1], has_legacy_bc, bsrc_evaluator, brea_evaluator, bstor_evaluator, oldbstor_evaluator, U, UOld, boundary_factors, boundary_values, F, dudp, nspecies, new_ind)
	
	if isnontrivial(bflux_evaluator)
        nballoc += @allocated bedge_asm!(bedge, system, UKLs[1], bflux_evaluator, U, F, dudp, nspecies, new_ind)
	end
	
	
	#=
	nnzCSC, nnzLNK = ExtendableSparse.nnz_noflush(system.matrix)
	if nnzLNK > 0
		@info "[[ LNK rowval : $nnzLNK"
		for tid=1:nt
			@info tid
			tmp = SparseMatrixCSC(system.matrix.lnkmatrices[tid])
			
			if maximum(tmp.nzval) > 0.0
				ids = findall(x->x!=0, tmp.nzval)	
				#@info tmp.colptr
				@info tmp.rowval[ids]
				@info tmp.nzval[ids]
			end
			
			@info tmp.colptr[1:20]
			@info tmp.rowval[1:10]
			@info tmp.nzval[1:10]
			
			#ids = findall(y->y!=0, system.matrix.lnkmatrices[tid].rowval)
			#system.matrix.lnkmatrices[tid].nzval[ids] .+= 0.001
			
		end
			
		#tid = 1
		#ids = findall(x->x!=0, system.matrix.lnkmatrices[tid].rowval)
		
	
		#@info system.matrix.lnkmatrices[1].rowval[1:10]
		#@info system.matrix.lnkmatrices[1].nzval[1:10]
		@info "]]"
	end
	=#
	
	nfa = @allocated begin
		nnzCSC, nnzLNK = ExtendableSparse.nnz_noflush(system.matrix)
		if nnzCSC > 0 && nnzLNK > 0	
			VoronoiFVM.ExtendableSparse.flush!(system.matrix; do_dense=false)
			#sparse flush
		elseif nnzCSC == 0 && nnzLNK > 0
			VoronoiFVM.ExtendableSparse.flush!(system.matrix; do_dense=true)
			#dense flush
		end 
	end	
	
	
	
	#@info "Node allocs:", nodealloc, nodealloc2, nodesepaalloc
	#@info "Edge allocs:", edgealloc, edgealloc2
	
	nodealloc_sum = sum(nodealloc)+sum(nodealloc2)
	edgealloc_sum = sum(edgealloc)+sum(edgealloc2)
	
	
	#nnzCSC1, nnzLNK1 = ExtendableSparse.nnz_noflush(system.matrix)
	if detail_allocs
		@info "pre-flush : nnzCSC $nnzCSC, nnzLNK $nnzLNK"
		#@info "$nodealloc_sum $nodesepaalloc $edgealloc_sum"
		#@info "$nodealloc_sum $nodesepaalloc ($nodesepaass | $nodesepapre | $nodesepafct) $edgealloc_sum"
	end
	#@info "pre-flush : nnzCSC $nnzCSC, nnzLNK $nnzLNK | post-flush: nnzCSC $nnzCSC1, nnzLNK $nnzLNK1"
	
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
	ncalloc = nodealloc_sum+nodesepaalloc+edgealloc_sum
    ncalloc, nballoc, 1, nfa
    
end



function eval_and_assemble_part_para_ESMP_fcts2(
    system::System{Tv,Tc,Ti,Tm,TSpecMat,TSolArray},
    U::AbstractMatrix{Tv}, # Actual solution iteration
    UOld::AbstractMatrix{Tv}, # Old timestep solution
    F::AbstractMatrix{Tv},# Right hand side
    time,
    tstep,# time step size. Inf means stationary solution
    λ,
    params::AbstractVector;
    edge_cutoff = 0.0, detail_allocs=false) where {Tv,Tc,Ti,Tm,TSpecMat,TSolArray}
    
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
    
    
    
	nodealloc = zeros(Int64, nt)
	nodealloc2 = zeros(Int64, nt)
	for level=1:depth
		@threads for tid=1:nt
			n = node_asm_count!(
				system, nodes[tid], UKs[tid], UKOlds[tid], src_evaluators[tid], rea_evaluators[tid], stor_evaluators[tid], oldstor_evaluators[tid],
				cfp[(level-1)*nt+tid], U, UOld, tid, nspecies, tstepinv, F, dudp, new_ind
			)
			nodealloc[tid] += n[1]
			nodealloc2[tid] += n[2]
			#nodealloc[tid] += @allocated node_asm!(
			#	system, nodes[tid], UKs[tid], UKOlds[tid], src_evaluators[tid], rea_evaluators[tid], stor_evaluators[tid], oldstor_evaluators[tid],
			#	cfp[(level-1)*nt+tid], U, UOld, tid, nspecies, tstepinv, F, dudp, new_ind
			#)
		end
	end
	
	nodesepaalloc = 0
	nodesepaalloc += @allocated node_asm!(
				system, nodes[1], UKs[1], UKOlds[1], src_evaluators[1], rea_evaluators[1], stor_evaluators[1], oldstor_evaluators[1],
				cfp[depth*nt+1], U, UOld, 1, nspecies, tstepinv, F, dudp, new_ind
			)
	
		
    if isnontrivial(outflow_evaluators[1])
    end


    edgealloc = zeros(Int64, nt) #@allocated begin
    edgealloc2 = zeros(Int64, nt)
	for level=1:depth
		@threads for tid=1:nt
			n = edge_asm_count!(
				system, edges[tid], UKLs[tid], flux_evaluators[tid], erea_evaluators[tid], outflow_evaluators[tid],
				cfp[(level-1)*nt+tid], U, tid, nspecies, tstepinv, F, dudp, new_ind
			)
			edgealloc[tid] += n[1]
			edgealloc2[tid] += n[2]
			
			#edgealloc[tid] += @allocated edge_asm!(
			#	system, edges[tid], UKLs[tid], flux_evaluators[tid], erea_evaluators[tid], outflow_evaluators[tid],
			#	cfp[(level-1)*nt+tid], U, tid, nspecies, tstepinv, F, dudp, new_ind
			#)
				
    	end
    end
    
    edgealloc[1] += @allocated edge_asm!(
				system, edges[1], UKLs[1], flux_evaluators[1], erea_evaluators[1], outflow_evaluators[1],
				cfp[depth*nt+1], U, 1, nspecies, tstepinv, F, dudp, new_ind
	)
    
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

	

    nballoc = 0
    nballoc += @allocated bnode_asm!(bnode, system, UKs[1], has_legacy_bc, bsrc_evaluator, brea_evaluator, bstor_evaluator, oldbstor_evaluator, U, UOld, boundary_factors, boundary_values, F, dudp, nspecies, new_ind)
	
	if isnontrivial(bflux_evaluator)
		nballoc += @allocated bedge_asm!(bedge, system, UKLs[1], bflux_evaluator, U, F, dudp, nspecies, new_ind)
	end
	
	nfa = @allocated begin
		nnzCSC, nnzLNK = ExtendableSparse.nnz_noflush(system.matrix)
		if nnzCSC > 0 && nnzLNK > 0
			VoronoiFVM.ExtendableSparse.ESMP_flush!(system.matrix; do_dense=false)
			#sparse flush
		elseif nnzCSC == 0 && nnzLNK > 0
			VoronoiFVM.ExtendableSparse.ESMP_flush!(system.matrix; do_dense=true)
			#dense flush
		end 
	end	
	
	
	
	#@info "Node allocs:", nodealloc, nodealloc2, nodesepaalloc
	#@info "Edge allocs:", edgealloc, edgealloc2
	
	nodealloc_sum = sum(nodealloc)+sum(nodealloc2)
	edgealloc_sum = sum(edgealloc)+sum(edgealloc2)
	
	
	#nnzCSC1, nnzLNK1 = ExtendableSparse.nnz_noflush(system.matrix)
	#if detail_allocs
	#	@info "pre-flush : nnzCSC $nnzCSC, nnzLNK $nnzLNK"
	#	#@info "$nodealloc_sum $nodesepaalloc $edgealloc_sum"
	#	#@info "$nodealloc_sum $nodesepaalloc ($nodesepaass | $nodesepapre | $nodesepafct) $edgealloc_sum"
	#end
	#@info "pre-flush : nnzCSC $nnzCSC, nnzLNK $nnzLNK | post-flush: nnzCSC $nnzCSC1, nnzLNK $nnzLNK1"
	
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
	ncalloc = nodealloc_sum+nodesepaalloc+edgealloc_sum
    ncalloc, nballoc, 1, nfa
    
end
