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

	if system.assembly_type == :cellwise
		eval_and_assemble_part_para_ESMP_fcts_cellwise(system, U, UOld, F, time, tstep, λ, params; edge_cutoff, detail_allocs)
	else
		eval_and_assemble_part_para_ESMP_fcts_edgewise(system, U, UOld, F, time, tstep, λ, params; edge_cutoff, detail_allocs)
	end
end

function eval_and_assemble_part_para_ESMP_fcts_cellwise(
    system::System{Tv,Tc,Ti,Tm,TSpecMat,TSolArray},
    U::AbstractMatrix{Tv}, # Actual solution iteration
    UOld::AbstractMatrix{Tv}, # Old timestep solution
    F::AbstractMatrix{Tv},# Right hand side
    time,
    tstep,# time step size. Inf means stationary solution
    λ,
    params::AbstractVector;
    edge_cutoff = 0.0, detail_allocs=false) where {Tv,Tc,Ti,Tm,TSpecMat,TSolArray}
    

    #@info ":cellwise assembly"

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
    
    
    leveltimes = zeros(depth+2)
	nodeallocs = zeros(Int64, nt)
	#nodealloc2 = zeros(Int64, nt)
	for level=1:depth
		leveltimes[level] = @elapsed @threads for tid=1:nt
			nodeallocs[tid] += @allocated node_asm!(
				system, nodes[tid], UKs[tid], UKOlds[tid], src_evaluators[tid], rea_evaluators[tid], stor_evaluators[tid], oldstor_evaluators[tid],
				cfp[(level-1)*nt+tid], U, UOld, tid, nspecies, tstepinv, F, dudp, new_ind
			)
		end
	end
	
	leveltimes[depth+1] = @elapsed @threads for tid=1:1
    nodeallocs[1] += @allocated node_asm!(
				system, nodes[1], UKs[1], UKOlds[1], src_evaluators[1], rea_evaluators[1], stor_evaluators[1], oldstor_evaluators[1],
				cfp[depth*nt+1], U, UOld, 1, nspecies, tstepinv, F, dudp, new_ind
			)
	end

		
    if isnontrivial(outflow_evaluators[1])
    end


    edgeallocs = zeros(Int64, nt)
	for level=1:depth
		leveltimes[level] += @elapsed @threads for tid=1:nt
			edgeallocs[tid] += @allocated edge_asm!(
				system, edges[tid], UKLs[tid], flux_evaluators[tid], erea_evaluators[tid], outflow_evaluators[tid],
				cfp[(level-1)*nt+tid], U, tid, nspecies, tstepinv, F, dudp, new_ind
			)
    	end
    end
    
    leveltimes[depth+1] += @elapsed @threads for tid=1:1
    edgeallocs[1] += @allocated edge_asm!(
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
	leveltimes[depth+2] = @elapsed begin
		nballoc += @allocated bnode_asm!(bnode, system, UKs[1], has_legacy_bc, bsrc_evaluator, brea_evaluator, bstor_evaluator, oldbstor_evaluator, U, UOld, boundary_factors, boundary_values, F, dudp, nspecies, new_ind)
		
		if isnontrivial(bflux_evaluator)
			nballoc += @allocated bedge_asm!(bedge, system, UKLs[1], bflux_evaluator, U, F, dudp, nspecies, new_ind)
		end
	end
	
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
	
	#nodealloc_sum = sum(nodealloc)+sum(nodealloc2)
	#edgealloc_sum = sum(edgealloc)+sum(edgealloc2)
	
	
	#nnzCSC1, nnzLNK1 = ExtendableSparse.nnz_noflush(system.matrix)
	if detail_allocs
		print_level_times(leveltimes)
		@info "pre-flush : nnzCSC $nnzCSC, nnzLNK $nnzLNK"
		#@info "$nodealloc_sum $nodesepaalloc $edgealloc_sum"
		#@info "$nodealloc_sum $nodesepaalloc ($nodesepaass | $nodesepapre | $nodesepafct) $edgealloc_sum"
	end
	_eval_and_assemble_generic_operator(system, U, F)
    _eval_and_assemble_inactive_species(system, U, UOld, F)
	ncalloc = sum(nodeallocs)+sum(edgeallocs)  #nodealloc_sum+nodesepaalloc+edgealloc_sum
    ncalloc, nballoc, 1, nfa
    
end

function eval_and_assemble_part_para_ESMP_fcts_edgewise(
    system::System{Tv,Tc,Ti,Tm,TSpecMat,TSolArray},
    U::AbstractMatrix{Tv}, # Actual solution iteration
    UOld::AbstractMatrix{Tv}, # Old timestep solution
    F::AbstractMatrix{Tv},# Right hand side
    time,
    tstep,# time step size. Inf means stationary solution
    λ,
    params::AbstractVector;
    edge_cutoff = 0.0, detail_allocs=false) where {Tv,Tc,Ti,Tm,TSpecMat,TSolArray}
    
    #@info ":edgewise assembly 03"
    _complete!(system) # needed here as well for test function system which does not use newton


	nn = num_nodes(system.grid)

	# Reset matrix + rhs
    #reset!(system.matrix)
	
    system.matrix.cscmatrix.nzval .= 0
    #new_ind = system.matrix.new_indices
    #rni     = system.matrix.rev_new_indices
    cfp     = system.matrix.cellsforpart
    nt      = system.matrix.nt
    depth   = system.matrix.depth
    start   = system.matrix.start


    grid = system.grid
    physics = system.physics
    nodes = [Node(system, time, λ, params) for tid=1:nt]
    edges = [Edge(system, time, λ, params) for tid=1:nt]
    nspecies::Int = num_species(system)

	nodetime = zeros(nt*depth+1)
    edgetime = zeros(nt*depth+1)
    bndtime  = 0.0 #zeros(2*nt*depth+2+1)
    
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
    
	nodeallocs = zeros(nt)
	edgeallocs = zeros(nt)

	for level=1:depth
		@threads for tid=1:nt
			nodeallocs[tid] += @allocated begin
				tmp = start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				nodetime[(level-1)*nt+tid] += @elapsed node_asm_edgewise03!(
					system, nodes[tid], F, dudp, U, UOld, UKs[tid], UKOlds[tid], src_evaluators[tid], rea_evaluators[tid], 
					stor_evaluators[tid], oldstor_evaluators[tid], tstepinv, nspecies, tmp, tid
				)
			end
		end
	end

	level = depth+1
	nodeallocs[1] += @allocated begin
		tmp0 = start[nt*depth+1]:start[nt*depth+2]-1
		nodetime[nt*depth+1] += @elapsed node_asm_edgewise03!(
			system, nodes[1], F, dudp, U, UOld, UKs[1], UKOlds[1], src_evaluators[1], rea_evaluators[1], 
			stor_evaluators[1], oldstor_evaluators[1], tstepinv, nspecies, tmp0, 1
		)
	end

	if isnontrivial(outflow_evaluators[1])
    end
    
	
    for level=1:depth
		@threads for tid=1:nt
			edgeallocs[tid] += @allocated begin
				tmp = cfp[(level-1)*nt+tid]
				edgetime[(level-1)*nt+tid] += @elapsed edge_asm_edgewise03!(
					system, edges[tid], F, dudp, U, UKLs[tid], flux_evaluators[tid], 
					erea_evaluators[tid], outflow_evaluators[tid], nspecies, tmp, tid
				)
			end
		end
	end

	level = depth+1
	edgeallocs[1] += @allocated begin
		tmp0 = cfp[depth*nt+1]
		edgetime[nt*depth+1] += @elapsed edge_asm_edgewise03!(
			system, edges[1], F, dudp, U, UKLs[1], flux_evaluators[1], 
			erea_evaluators[1], outflow_evaluators[1], nspecies, tmp0, 1
		)
	end
    
    bnode = BNode(system, time, λ, params)
    bedge = BEdge(system, time, λ, params)


    boundary_factors::Array{Tv,2} = system.boundary_factors
    boundary_values::Array{Tv,2} = system.boundary_values
    has_legacy_bc = !iszero(boundary_factors) || !iszero(boundary_values)
    bsrc_evaluator = ResEvaluator(physics, :bsource, UKs[1], bnode, nspecies)
    brea_evaluator = ResJacEvaluator(physics, :breaction, UKs[1], bnode, nspecies)
    bstor_evaluator = ResJacEvaluator(physics, :bstorage, UKs[1], bnode, nspecies)
    oldbstor_evaluator = ResEvaluator(physics, :bstorage, UKs[1], bnode, nspecies)
    bflux_evaluator = ResJacEvaluator(physics, :bflux, UKLs[1], bedge, nspecies)
	

    nballoc = 0
	bndtime = @elapsed begin
		nballoc += @allocated bnode_asm_edgewise03!(bnode, system, UKs[1], has_legacy_bc, bsrc_evaluator, brea_evaluator, bstor_evaluator, oldbstor_evaluator, U, UOld, boundary_factors, boundary_values, F, dudp, nspecies)
		
		if isnontrivial(bflux_evaluator)
			nballoc += @allocated bedge_asm_edgewise03!(bedge, system, UKLs[1], bflux_evaluator, U, F, dudp, nspecies)
		end
	end
	
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
	
	
	if detail_allocs
		#print_level_times(leveltimes)
		@info nodetime
		@info edgetime
		@info bndtime
		@info "pre-flush : nnzCSC $nnzCSC, nnzLNK $nnzLNK", length.(cfp)
	end
	
    _eval_and_assemble_generic_operator(system, U, F)
    _eval_and_assemble_inactive_species(system, U, UOld, F)
	ncalloc = sum(nodeallocs)+sum(edgeallocs)
	
	ncalloc, nballoc, 1, nfa
    #0, 0, 1, 0
end




function print_level_times(x)
	depth = length(x)-2

	s = ""
	for i=1:depth
		s = s*"L$i: $(round(x[i],digits=3)), "
	end
	s = s*"Sepa: $(round(x[depth+1],digits=3)), "
	s = s*"Bnd.: $(round(x[depth+2],digits=3)), "

	@info s
end
