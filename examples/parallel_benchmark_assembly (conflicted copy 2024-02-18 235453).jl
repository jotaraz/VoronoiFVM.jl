#include("/home/johannes/Nextcloud/Documents/Uni/VIII/WIAS/juliaCode(anfang)/para/ExtendableSparseParallel/src/ESMP/ExtendableSparseParallel.jl")

#using .ExtendableSparseParallel


#https://github.com/j-fu/VoronoiFVM.jl/blob/7039d768ba26e80dbb1aa4fabcce5993486561a4/examples/Example207_NonlinearPoisson2D.jl

#using PyPlot, GridVisualize, ExtendableGrids

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

"""
`function example3(nm; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)`
              
This function uses an ExtendableSparseMatrix as the system matrix.
Sequentially assemble the system matrix and the right hand side of some physics on an nm grid.
For a 2d 100 x 78 grid: nm = (100,78). For a 3d 50x30x21 grid: nm=(50,30,21).

Physics taken from https://github.com/j-fu/VoronoiFVM.jl/blob/7039d768ba26e80dbb1aa4fabcce5993486561a4/examples/Example207_NonlinearPoisson2D.jl.
"""
function example3(nm; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)
    
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
    if do_init
	    VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble(sys,
		                               solution,
		                               oldsol,
		                               residual,
		                               0.0,
		                               0.01,
		                               0.0,
		                               zeros(0);
		                               edge_cutoff = 1e-5)) #control.edge_cutoff,)
    
    
    t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble(sys,
		                               solution,
		                               oldsol,
		                               residual,
		                               0.0,
		                               0.01,
		                               0.0,
		                               zeros(0);
		                               edge_cutoff = 1e-5))
    
    sys.matrix, residual, (t1, t2), (na1, na2)
end


"""
`function example3_ESMP_part(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)`
              
This function uses an ExtendableSparseMatrixParallel as the system matrix.
Sequentially assembles the system matrix and the right hand side of some physics on an nm grid.
For a 2d 100 x 78 grid: nm = (100,78). For a 3d 50x30x21 grid: nm=(50,30,21).
This function partions the grid but goes over all partitions sequentially, it should only be used for testing and benchmarking.

Physics taken from https://github.com/j-fu/VoronoiFVM.jl/blob/7039d768ba26e80dbb1aa4fabcce5993486561a4/examples/Example207_NonlinearPoisson2D.jl.
"""
function example3_ESMP_part(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
    
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth;  unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    inival = unknowns(sys)
    inival .= 0.5
	

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble_part_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0);edge_cutoff = 1e-5))
	
	t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble_part_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0);edge_cutoff = 1e-5))
	
	
	sys.matrix, residual, (t1, t2), (na1, na2)
end

"""
`function example3_ESMP_part_para(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)`
              
This function uses an ExtendableSparseMatrixParallel as the system matrix.
Parallely assembles the system matrix and the right hand side of some physics on an nm grid.
For a 2d 100 x 78 grid: nm = (100,78). For a 3d 50x30x21 grid: nm=(50,30,21).

Physics taken from https://github.com/j-fu/VoronoiFVM.jl/blob/7039d768ba26e80dbb1aa4fabcce5993486561a4/examples/Example207_NonlinearPoisson2D.jl.
"""
function example3_ESMP_part_para(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true)
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
								   
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    inival = unknowns(sys)
    inival .= 0.5

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble_part_para_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5)) #control.edge_cutoff)
    
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t1, na1
	
	t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble_part_para_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5))
	
	
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t2, na2
	
	sys.matrix, residual, (t1, t2), (na1, na2)
end


function example3_ESMP_part_para_wrappedfcts(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, detail_allocs=true)
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
								   
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

    inival = unknowns(sys)
    inival .= 0.5

    solution = unknowns(sys)
    oldsol = inival
    
    VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    solution .= oldsol
    
    residual = sys.residual
    update = sys.update
    
    if do_init
    	VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
    end
    
    t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble_part_para_ESMP_fcts(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5, detail_allocs)) #control.edge_cutoff)
    
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t1, na1
	
	t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble_part_para_ESMP_fcts(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5, detail_allocs))
	
	
    @info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t2, na2
	
	sys.matrix, residual, (t1, t2), (na1, na2)
end

function reorder(x, ni)
	y = Vector{typeof(x[1])}(undef, length(x)) #copy(x)
	for i=1:length(x)
		y[ni[i]] = x[i]
	end
	y

end



#----------------------------------------------------------------------








function bm_example3_ESMP_part_para(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, num=5, detail_allocs=true)
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
								   
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)
	VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
	
	T1 = 100
	T2 = 100	
    na1 = 0
    na2 = 0
	
	for i=1:num
		inival = unknowns(sys)
		inival .= 0.5

		solution = unknowns(sys)
		oldsol = inival
		
		
		solution .= oldsol
		
		residual = sys.residual
		update = sys.update
		
		if do_init
			VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
		end
		
		t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble_part_para_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5)) #control.edge_cutoff)
		
		if detail_allocs
			@info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t1, na1
		end
		
		t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble_part_para_ESMP(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5))
		
		T1 = min(T1, t1)
		T2 = min(T2, t2)
		
		if detail_allocs
			@info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t2, na2
		end
    end
	
	
	
	sys.matrix, (T1, T2), (na1, na2)
end


function bm_example3_ESMP_part_para_wrappedfcts(nm, nt, depth; verbose = false, unknown_storage = :sparse,
              method_linear = nothing, assembly = :cellwise,
              precon_linear = A -> VoronoiFVM.Identity(), do_init = true, detail_allocs=true, num=5)
    
    physics = VoronoiFVM.Physics(; reaction = reaction,
								   flux = flux,
								   source = source, 
								   storage = storage)
								   
    sys = VoronoiFVM.ParallelSystem(Float64, Float64, Int32, Int64, nm, nt, depth; unknown_storage, species = [1])
    
	physics!(sys, physics)
    # enable_species!(sys, 1, [1])

    boundary_dirichlet!(sys, 1, 2, 0.1)
    boundary_dirichlet!(sys, 1, 4, 0.1)

	VoronoiFVM._complete_nomatrix!(sys; create_newtonvectors = true)
    
    T1 = 100.0 #zeros(num)
    T2 = 100.0 #zeros(num)
    na1 = 0
    na2 = 0
    
	for i=1:num
		inival = unknowns(sys)
		inival .= 0.5

		solution = unknowns(sys)
		oldsol = inival
		
		
		solution .= oldsol
		
		residual = sys.residual
		update = sys.update
		
		if do_init
			VoronoiFVM._initialize!(solution, sys; time=0.0, λ = 0.0, params=zeros(0))
		end
		
		t1 = @elapsed (na1 = VoronoiFVM.eval_and_assemble_part_para_ESMP_fcts(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5, detail_allocs)) #control.edge_cutoff)
		
		if detail_allocs
			@info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t1, na1
		end
		
		t2 = @elapsed (na2 = VoronoiFVM.eval_and_assemble_part_para_ESMP_fcts(sys, solution, oldsol, residual, 0.0, 0.01, 0.0, zeros(0); edge_cutoff = 1e-5, detail_allocs))
		
		T1 = min(T1, t1)
		T2 = min(T2, t2)
		
		if detail_allocs
			@info VoronoiFVM.ExtendableSparseParallel.nnz_noflush(sys.matrix), t2, na2
		end
	end
	
	sys.matrix, (T1, T2), (na1, na2)
end



#---------------------------------------------------------------------



"""
`function validate_ESMP(nm, nt, depth, do_init; test_vec=true)`

Computes the system matrix and the residual normally, partitioned (but sequentially) and partitioned parallel.
Then compares the results (first outputs, iff test_vec=true).
Then prints times and allocations.
"""
function validate_ESMP_2(nm, nt, depth, do_init; test_vec=true, detail_allocs=true)
	@info "Sequential"
	ESM,r1,t1,n1     = example3(nm; do_init=do_init)
	CSC1             = SparseArrays.SparseMatrixCSC(ESM)
	
	@info "Wrapped fcts"
	ESMP2,r4,t4,n4   = example3_ESMP_part_para_wrappedfcts(nm, nt, depth; do_init=do_init, detail_allocs)
	CSC4     = ESMP2.cscmatrix
	
	@info "Old Part Para"
	ESMP3,r5,t5,n5   = example3_ESMP_part_para(nm, nt, depth; do_init=do_init)
	CSC5             = ESMP3.cscmatrix
	
	r1 = VoronoiFVM.values(r1)
	r4 = VoronoiFVM.values(r4)
	r5 = VoronoiFVM.values(r5)
	
	
	if test_vec	
		nn = num_nodes(ESMP2.grid)
		b = rand(nn)
		v = CSC1*b
		
		br = reorder(b, ESMP2.new_indices)
		vr4 = reorder(CSC4*br, ESMP2.rev_new_indices)
		vr5 = reorder(CSC5*br, ESMP3.rev_new_indices)
		
		rr4 = reorder(r4, ESMP2.rev_new_indices)
		rr5 = reorder(r5, ESMP3.rev_new_indices)
		
		@warn "Differences in RHS"
		@info "max diff 4: ", maximum(abs.(r1-rr4))
		@info "max diff 5: ", maximum(abs.(r1-rr5))
		
		@warn "Differences in Matrix"
		@info "max diff 4: ", maximum(abs.(v-vr4))
		@info "max diff 5: ", maximum(abs.(v-vr5))
	end
	
	@warn "Times:"
	@info "Old Seq: ", t1
	@info "New Seq: ", t4
	@info "New Par: ", t5
	@warn "Allocations:"
	@info "Old Seq: ", n1
	@info "New Seq: ", n4
	@info "New Par: ", n5
	

end


"""
`function validate_ESMP(nm, nt, depth, do_init; test_vec=true)`

Computes the system matrix and the residual normally, partitioned (but sequentially) and partitioned parallel.
Then compares the results (first outputs, iff test_vec=true).
Then prints times and allocations.
"""
function benchmark_ESMP_2(nm, nt, depth, do_init; test_vec=true, detail_allocs=true, num=5)
	@info "Sequential"
	ESM,r1,t1,n1     = example3(nm; do_init=do_init)
	CSC1             = SparseArrays.SparseMatrixCSC(ESM)
	
	@info "Wrapped fcts"
	ESMP2,t4,n4   = bm_example3_ESMP_part_para_wrappedfcts(nm, nt, depth; do_init=do_init, detail_allocs, num)
	CSC4     = ESMP2.cscmatrix
	
	@info "Old Part Para"
	ESMP3,t5,n5   = bm_example3_ESMP_part_para(nm, nt, depth; do_init=do_init, detail_allocs, num)
	CSC5             = ESMP3.cscmatrix
	
	
	if test_vec	
		nn = num_nodes(ESMP2.grid)
		b = rand(nn)
		v = CSC1*b
		
		br = reorder(b, ESMP2.new_indices)
		vr4 = reorder(CSC4*br, ESMP2.rev_new_indices)
		vr5 = reorder(CSC5*br, ESMP3.rev_new_indices)
		
		@warn "Differences in Matrix"
		@info "max diff 4: ", maximum(abs.(v-vr4))
		@info "max diff 5: ", maximum(abs.(v-vr5))
	end
	
	@warn "Times:"
	@info "Old Seq: ", t1
	@info "New Seq: ", t4
	@info "New Par: ", t5
	@warn "Allocations:"
	@info "Old Seq: ", n1
	@info "New Seq: ", n4
	@info "New Par: ", n5
	

end




"""
`function validate_ESMP(nm, nt, depth, do_init; test_vec=true)`

Computes the system matrix and the residual normally, partitioned (but sequentially) and partitioned parallel.
Then compares the results (first outputs, iff test_vec=true).
Then prints times and allocations.
"""
function validate_ESMP(nm, nt, depth, do_init; test_vec=true)
	ESM,r1,t1,n1     = example3(nm; do_init=do_init)
	CSC1             = SparseArrays.SparseMatrixCSC(ESM)
	
	ESMP2,r4,t4,n4   = example3_ESMP_part(nm, nt, depth; do_init=do_init)
	CSC4     = ESMP2.cscmatrix
	
	ESMP3,r5,t5,n5   = example3_ESMP_part_para(nm, nt, depth; do_init=do_init)
	CSC5             = ESMP3.cscmatrix
	
	r1 = VoronoiFVM.values(r1)
	r4 = VoronoiFVM.values(r4)
	r5 = VoronoiFVM.values(r5)
	
	
	if test_vec	
		nn = num_nodes(ESMP2.grid)
		b = rand(nn)
		v = CSC1*b
		
		br = reorder(b, ESMP2.new_indices)
		vr4 = reorder(CSC4*br, ESMP2.rev_new_indices)
		vr5 = reorder(CSC5*br, ESMP3.rev_new_indices)
		
		rr4 = reorder(r4, ESMP2.rev_new_indices)
		rr5 = reorder(r5, ESMP3.rev_new_indices)
		
		@warn "Differences in RHS"
		@info "max diff 4: ", maximum(abs.(r1-rr4))
		@info "max diff 5: ", maximum(abs.(r1-rr5))
		
		@warn "Differences in Matrix"
		@info "max diff 4: ", maximum(abs.(v-vr4))
		@info "max diff 5: ", maximum(abs.(v-vr5))
	end
	
	@warn "Times:"
	@info "Old Seq: ", t1
	@info "New Seq: ", t4
	@info "New Par: ", t5
	@warn "Allocations:"
	@info "Old Seq: ", n1
	@info "New Seq: ", n4
	@info "New Par: ", n5
	

end


