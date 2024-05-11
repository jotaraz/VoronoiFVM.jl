"""
$(TYPEDEF)

Structure holding data for finite volume system.
"""
mutable struct System{Tv, Tc, Ti, Tm, TSpecMat <: AbstractMatrix, TSolArray <: AbstractMatrix} <: AbstractSystem{Tv, Tc, Ti, Tm}
    """
    Grid
    """
    grid::ExtendableGrid{Tc, Ti}

    """
    Physics data
    """
    physics::Physics

    """
    Array of boundary condition values 
    """
    boundary_values::Array{Tv, 2}

    """
    Array of boundary condition factors 
    """
    boundary_factors::Array{Tv, 2}
    6
    """
    Matrix containing species numbers for inner regions
    """
    region_species::TSpecMat

    """
    Matrix containing species numbers for boundary regions
    """
    bregion_species::TSpecMat

    """
    Matrix containing degree of freedom numbers for each node
    """
    node_dof::TSpecMat

    """
    - :multidiagonal  (currently disabled)
    - :sparse
    - :banded
    - :tridiagonal
    """
    matrixtype::Symbol

    """
    Jacobi matrix for nonlinear problem
    """
    matrix::Union{ExtendableSparse.ExtendableSparseMatrixParallel{Tv, Tm}, 
    			  ExtendableSparseMatrix{Tv, Tm},
                  Tridiagonal{Tv, Vector{Tv}},
                  #                  MultidiagonalMatrix,
                  BandedMatrix{Tv}}

    """
    Linear solver cache
    """
    linear_cache::Union{Nothing, LinearSolve.LinearCache}

    """
    Flag which says if the number of unknowns per node is constant
    """
    species_homogeneous::Bool

    """
    Number of quantities defined on system
    """
    num_quantities::Ti

    """
    Number of parameter the system depends on.
    """
    num_parameters::Ti

    """
    Parameter derivative (vector of solution arrays)
    """
    dudp::Vector{TSolArray}

    """
    Solution vector holding Newton update
    """
    update::TSolArray

    """
    Solution vector holding Newton residual
    """
    residual::TSolArray

    """
    Precomputed form factors for bulk  assembly
    """
    assembly_data::Union{Nothing, AbstractAssemblyData{Tc, Ti}}

    """
    Precomputed form factors for boundary assembly
    """
    boundary_assembly_data::AbstractAssemblyData{Tc, Ti}

    """
    :edgewise or :cellwise
    """
    assembly_type::Symbol

    """
    Is the system linear ?
    """
    is_linear::Bool

    """
    Outflow nodes with their region numbers.
    """
    outflownoderegions::Union{SparseMatrixCSC{Bool, Int}, Nothing}

    """
    Sparse matrix for generic operator handling
    """
    generic_matrix::SparseMatrixCSC

    """
    Sparse matrix colors for generic operator handling
    """
    generic_matrix_colors::Vector

    """
    Hash value of latest unknowns vector the assembly was called with
    """
    uhash::UInt64

    """
    History record for last solution process
    """
    history::Any

    System{Tv, Tc, Ti, Tm, TSpecMat, TSolArray}() where {Tv, Tc, Ti, Tm, TSpecMat, TSolArray} = new()
end

function Base.getproperty(sys::System, sym::Symbol)
    if sym == :bfacenodefactors
        @warn "sys.bfacenodefactors is deprecated and will be removed in one of the next minor releases. Use  bfacenodefactors(sys) instead"
        return sys.boundary_assembly_data.nodefactors
    else # fallback to getfield
        return getfield(sys, sym)
    end
end

"""
    const DenseSystem

Type alias for system with dense matrix based species management

"""
const DenseSystem = System{Tv, Tc, Ti, Tm, Matrix{Ti}, Matrix{Tv}} where {Tv, Tc, Ti, Tm}

isdensesystem(s::System{Tv, Tc, Ti, Tm, TSpecMat, TSolArray}) where {Tv, Tc, Ti, Tm, TSpecMat, TSolArray} = TSolArray <: Matrix

"""
    const SparseSystem

Type alias for system with sparse matrix based species management

"""
const SparseSystem = System{Tv, Tc, Ti, Tm, SparseMatrixCSC{Ti, Ti}, SparseSolutionArray{Tv, Ti}} where {Tv, Tc, Ti, Tm}

##################################################################
"""
````
System(grid; kwargs...)
````
    
Create structure of type [`VoronoiFVM.System{Tv,Ti, Tm, TSpecMat<:AbstractMatrix, TSolArray<:AbstractMatrix}`](@ref)  holding data for finite volume system solution. 

Parameters: 

- `grid::ExtendableGrid`: 1, 2 or 3D computational grid

Keyword arguments:
- `species`: vector of integer species indices. Added to all grid regions, avoiding the need to call [`enable_species!`](@ref) for this default case.
              If it is kept empty, species have be added to the system after creation via  [`enable_species!`](@ref).
- `unknown_storage`: string or symbol.  
    Information  on  species  distribution  is kept  in  sparse  or  dense
    matrices matrices and, correspondingly, the  solution array is of type
    SparseSolutionArray  or matrix,  respectively. In  the case  of sparse
    unknown storage,  the system matrix  handles exactly those  degrees of
    freedom which correspond to unknowns.  However, handling of the sparse
    matrix  structures  for  the   bookkeeping  of  the  unknowns  creates
    overhead.
     - `:dense` :  solution vector is an  `nspecies` x `nnodes`  dense matrix
     - `:sparse` :  solution vector is an `nspecies` x `nnodes`  sparse matrix
- `matrixindextype`: Integer type. Index type for sparse matrices created in the system.
- `is_linear`: whether the system is linear or not. If it is linear, only one Newton step is used to solve it.
- `assembly`: either `:cellwise` (default) or `:edgewise`. Determine, how the assembly loop is organized.
   `:cellwise` means that the outer loop goes over grid cells (triangles, tetrahedra), and contributions to
   edge fluxes and node reactions are calculated for each cell. As a consequence, e.g. im 2D for all interior
   edges, flux functions are callled twice, once for each adjacent cell. Especially in 3D, this becomes a significant
   overhead. With `:edgewise`, geometry factors of these edges are pre-assembled, and the outer assembly loops
   go over all grid edges resp. nodes, still with separate calls if neigboring cells belong to different regions.
!!! note
    It is planned to make `:edgewise` the default in a later version.

Physics keyword arguments:
- `flux`: Function.     Flux between neighboring control volumes: `flux(f,u,edge)` or `flux(f,u,edge,data)`
    should return in `f[i]` the flux of species i along the edge joining circumcenters
    of neighboring control volumes.  For species i,`u[i,1]` and `u[i,2]` contain the unknown values at the corresponding ends of the edge.
- `storage`: Function.  Storage term (term under time derivative): `storage(f,u,node)` or `storage(f,u,node,data)` 
    It should return in `f[i]` the storage term for the i-th equation. `u[i]` contains the value of
    the i-th unknown.
- `reaction`:  Function. Reaction term:  `reaction(f,u,node)` or `reaction(f,u,node,data)` 
    It should return in `f[i]` the reaction term for the i-th equation. `u[i]` contains the value of
    the i-th unknown.
- `edgereaction`:  Function. Edge reeaction term:  `edgereaction(f,u,edge)` or `edgereaction(f,u,edge,data)` 
    It should return in `f[i]` the reaction term for the i-th equation.  For species i,`u[i,1]` and `u[i,2]` contain the unknown values at the corresponding ends of the edge.
- `source`:  Function. Source term: `source(f,node)` or `source(f,node,data)`.
    It should return the in `f[i]` the value of the source term for the i-th equation.
- `bflux`:  Function. Flux between neighboring control volumes on the boundary
- `breaction` Function.  Boundary reaction term:  `breaction(f,u,node)` or `breaction(f,u,node,data)` 
    Similar to reaction, but restricted to the inner or outer boundaries.
- `bcondition` Function. Alias for `breaction`.
- `bsource`: Function. Boundary source term: `bsource(f,node)` or `bsource(f,node,data)`.
    It should return in `f[i]` the value of the source term for the i-th equation.
- `bstorage`: Function.  Boundary storage term: `bstorage(f,u,node)` or `bstorage(f,u,node,data)` 
    Similar to storage, but restricted to the inner or outer boundaries.
- `generic_operator`: Function.  Generic operator  `generic_operator(f,u,sys)`. 
    This operator acts on the full solution `u` of a system. Sparsity
    is detected automatically  unless `generic_operator_sparsity` is given.
-  `generic_operator_sparsity`:  Function defining the sparsity structure of the generic operator.
    This should return the sparsity pattern of the `generic_operator`.
-  `nparams`: number of parameters the system is depending on, and with respect to which the derivatives
    need to be obtained
-  `data`:  User data (parameters).
    This allows to pass various parameters to the callback functions. If `data` is given, all callback functions
    should accept a last `data` argument. Otherwise, no data are passed explicitly, and constitutive callbacks can
    take parameters from the closure where the function is defined.

-  `matrixtype`: :default, :sparse, :tridiagonal, :banded

"""
function System(grid::ExtendableGrid;
                valuetype = coord_type(grid),
                indextype = index_type(grid),
                species = Int[],
                assembly = :cellwise,
                unknown_storage = :dense,
                matrixindextype = Int64,
                matrixtype = :sparse,
                is_linear = false,
                nparams = 0,
                kwargs...)
    Tv = valuetype
    Tc = coord_type(grid)
    Ti = indextype
    Tm = matrixindextype

    if Symbol(unknown_storage) == :dense
        system = System{Tv, Tc, Ti, Tm, Matrix{Ti}, Matrix{Tv}}()
    elseif Symbol(unknown_storage) == :sparse
        system = System{Tv, Tc, Ti, Tm, SparseMatrixCSC{Ti, Ti}, SparseSolutionArray{Tv, Ti}}()
    else
        throw("specify either unknown_storage=:dense  or unknown_storage=:sparse")
    end

    maxspec = 0
    system.grid = grid
    system.region_species = spzeros(Ti, Int16, maxspec, num_cellregions(grid))
    system.bregion_species = spzeros(Ti, Int16, maxspec, num_bfaceregions(grid))
    system.node_dof = spzeros(Ti, Tm, maxspec, num_nodes(grid))
    system.boundary_values = zeros(Tv, maxspec, num_bfaceregions(grid))
    system.boundary_factors = zeros(Tv, maxspec, num_bfaceregions(grid))
    system.species_homogeneous = false
    system.assembly_type = assembly
    system.num_quantities = 0
    system.uhash = 0x0
    system.matrixtype = matrixtype
    system.outflownoderegions = nothing
    system.linear_cache = nothing
    system.assembly_data = nothing
    system.history = nothing
    system.num_parameters = nparams
    system.is_linear = is_linear
    physics!(system; kwargs...)
    enable_species!(system; species)
    return system
end



function ParallelSystem(Tv, Tc, Ti, Tm, nm, nt, depth;
                #valuetype = coord_type(grid),
                #indextype = index_type(grid),
                species = Int[],
                assembly = :cellwise,
                unknown_storage = :dense,
                matrixindextype = Int64,
                matrixtype = :sparse,
                is_linear = false,
                nparams = 0,
                print_prep=false,
                check_partition=false,
                kwargs...)
    #Tv = valuetype
    #Tc = coord_type(grid)
    #Ti = indextype
    #Tm = matrixindextype

	
	if Symbol(unknown_storage) == :dense
        system = System{Tv, Tc, Ti, Tm, Matrix{Ti}, Matrix{Tv}}()
    elseif Symbol(unknown_storage) == :sparse
        system = System{Tv, Tc, Ti, Tm, SparseMatrixCSC{Ti, Ti}, SparseSolutionArray{Tv, Ti}}()
    else
        throw("specify either unknown_storage=:dense  or unknown_storage=:sparse")
    end
    
    grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts, cellparts, depth = preparatory_multi_ps_less_reverse(nm, nt, depth, Tm; do_print=print_prep, check_partition, assembly)
    
    #system.matrix = ExtendableSparseParallel.ExtendableSparseMatrixParallel{Tv, Tm}(nm, nt, depth)

    maxspec = 0
    system.grid = grid
    system.region_species = spzeros(Ti, Int16, maxspec, num_cellregions(system.grid))
    system.bregion_species = spzeros(Ti, Int16, maxspec, num_bfaceregions(system.grid))
    system.node_dof = spzeros(Ti, Tm, maxspec, num_nodes(system.grid))
    system.boundary_values = zeros(Tv, maxspec, num_bfaceregions(system.grid))
    system.boundary_factors = zeros(Tv, maxspec, num_bfaceregions(system.grid))
    system.species_homogeneous = false
    system.assembly_type = assembly
    system.num_quantities = 0
    system.uhash = 0x0
    system.matrixtype = matrixtype
    system.outflownoderegions = nothing
    system.linear_cache = nothing
    system.assembly_data = nothing
    system.history = nothing
    system.num_parameters = nparams
    system.is_linear = is_linear
    physics!(system; kwargs...)
    enable_species!(system; species)
    
    csc = spzeros(Tv, Tm, num_nodes(grid), num_nodes(grid))
	lnk = [SuperSparseMatrixLNK{Tv, Tm}(num_nodes(grid), nnts[tid]) for tid=1:nt]
	system.matrix = ExtendableSparseMatrixParallel{Tv, Tm}(csc, lnk, nnts, s, onr, cfp, gi, ni, rni, starts, cellparts, nt, depth, 0, csc.n, csc.m)
    
    return system
end

"""
````
System(X; kwargs...)
````
Create an [1D grid from vector X](https://j-fu.github.io/ExtendableGrids.jl/stable/gridconstructors/) and call  [`VoronoiFVM.System(grid::ExtendableGrid; kwargs...)`](@ref).
"""
System(X::AbstractVector; kwargs...) = System(simplexgrid(X); kwargs...)

"""
````
System(X,Y; kwargs...)
````
Create a [2D grid from vectors X,Y ](https://j-fu.github.io/ExtendableGrids.jl/stable/gridconstructors/) and call  [`VoronoiFVM.System(grid::ExtendableGrid; kwargs...)`](@ref).
"""
System(X::AbstractVector, Y::AbstractVector; kwargs...) = System(simplexgrid(X, Y); kwargs...)

"""
````
System(X,Y, Z; kwargs...)
````
Create a [3D grid from vectors X,Y,Z ](https://j-fu.github.io/ExtendableGrids.jl/stable/gridconstructors/) and call  [`VoronoiFVM.System(grid::ExtendableGrid; kwargs...)`](@ref).
"""
System(X::AbstractVector, Y::AbstractVector, Z::AbstractVector; kwargs...) = System(simplexgrid(X, Y, Z); kwargs...)

"""
    physics!(system,physics)

Replace System's physics data
"""
function physics!(system, physics)
    system.physics = physics
    system
end

"""
   physics!(system; kwargs...)

Replace System's physics data.
"""
function physics!(system; kwargs...)
    kwdict = Dict(kwargs)

    if haskey(kwdict, :bcondition)
        if haskey(kwdict, :breaction)
            error("specify either bcondition or breaction")
        end
        kwdict[:breaction] = kwdict[:bcondition]
        delete!(kwdict, :bcondition)
    end

    if haskey(kwdict, :source) && isa(kwdict[:source], AbstractArray)
        src = kwdict[:source]
        if isa(src, AbstractVector)
            kwdict[:source] = (y, node, args...) -> @views y[1] = src[node.index]
        else
            kwdict[:source] = (y, node, args...) -> @views y .= src[:, node.index]
        end
    end

    physics!(system, Physics(; kwdict...))
end

##################################################################

# Constant to be used as boundary condition factor 
# to mark Dirichlet boundary conditions.    
const Dirichlet = 1.0e30

#################################################################

"""
    addzrows(matrix,maxrow)

Return matrix with number of rows increased to maxrow, and set
the new elements to zero.
"""
function addzrows(matrix::Matrix, maxrow)
    nrow, ncol = size(matrix)
    if maxrow <= nrow
        return matrix
    end
    newmatrix = zeros(eltype(matrix), maxrow, ncol)
    for icol = 1:ncol
        for irow = 1:nrow
            newmatrix[irow, icol] = matrix[irow, icol]
        end
    end
    newmatrix
end

function addzrows(matrix::SparseMatrixCSC, maxrow)
    nrow, ncol = size(matrix)
    if maxrow <= nrow
        return matrix
    end
    SparseMatrixCSC(maxrow, matrix.n, matrix.colptr, matrix.rowval, matrix.nzval)
end

"""
     increase_num_species!(system,maxspec)

Increase number of species in system to maxspec by adding new rows to all  relevant
matrices.
"""
function increase_num_species!(system, maxspec)
    if maxspec <= num_species(system)
        return
    end

    if isdefined(system, :matrix)
        error("Unable to increase number of species to $(maxspec).\nPlease add species before first solver run.")
    end

    system.region_species = addzrows(system.region_species, maxspec)
    system.bregion_species = addzrows(system.bregion_species, maxspec)
    system.node_dof = addzrows(system.node_dof, maxspec)
    system.boundary_values = addzrows(system.boundary_values, maxspec)
    system.boundary_factors = addzrows(system.boundary_factors, maxspec)
end

##################################################################
"""
````
is_boundary_species(AbstractSystem, ispec) -> Bool
````

Check if species number corresponds to a boundary species.
"""
function is_boundary_species(system::AbstractSystem, ispec)
    isbspec = false
    for ibreg = 1:num_bfaceregions(system.grid)
        if system.bregion_species[ispec, ibreg] > 0
            isbspec = true
        end
    end
    return isbspec
end

##################################################################
"""
````
is_bulk_species(AbstractSystem, ispec) -> Bool
````

Check if species number corresponds to a bulk species.
"""
function is_bulk_species(system::AbstractSystem, ispec)
    isrspec = false
    for ixreg = 1:num_cellregions(system.grid)
        if system.region_species[ispec, ixreg] > 0
            isrspec = true
        end
    end
    return isrspec
end

##################################################################
"""
````
enable_species!(system,ispec,regions)
````

Add species `ispec` to a list of bulk regions. Species numbers for
bulk and boundary species have to be distinct. 
Once a species has been added, it cannot be removed.
"""
function enable_species!(system::AbstractSystem, ispec::Integer, regions::AbstractVector)
    increase_num_species!(system, ispec)

    if is_boundary_species(system, ispec)
        throw(DomainError(ispec, "Species is already boundary species"))
    end
    _cellregions = cellregions(system.grid)
    _cellnodes = cellnodes(system.grid)
    for i in eachindex(regions)
        ireg = regions[i]
        system.region_species[ispec, ireg] = ispec
        for icell = 1:num_cells(system.grid)
            if _cellregions[icell] == ireg
                for iloc = 1:num_targets(_cellnodes, icell)
                    iglob = _cellnodes[iloc, icell]
                    system.node_dof[ispec, iglob] = ispec
                end
            end
        end
    end
end

"""
````
enable_species!(system; kwargs...)
````

Keyword arguments:
- `species`: Integer or vector of integers. Species to be added to the system.
- `regions`: Vector of integers. Regions, where these species shall be added.If `nothing`, they are added to all species.
Once a species has been added, it cannot be removed.
"""
function enable_species!(sys::AbstractSystem; species = nothing, regions = nothing)
    if regions == nothing
        regions = collect(1:num_cellregions(sys.grid))
    end

    if isa(species, Number)
        species = [species]
    end

    for ispec ∈ species
        enable_species!(sys, ispec, regions)
    end
    sys
end

##################################################################
"""
````
enable_boundary_species!(system,ispec,regions)
````

Add species `ispec` to a list of boundary regions. Species numbers for
bulk and boundary species have to be distinct.
Once a species has been added, it cannot be removed.
"""
function enable_boundary_species!(system::AbstractSystem, ispec::Integer, bregions::AbstractVector)
    increase_num_species!(system, ispec)

    if is_bulk_species(system, ispec)
        throw(DomainError(ispec, "Species is already bulk species"))
    end
    _bfaceregions = bfaceregions(system.grid)
    _bfacenodes = bfacenodes(system.grid)

    for i in eachindex(bregions)
        ireg = bregions[i]
        system.bregion_species[ispec, ireg] = ispec
        for ibface = 1:num_bfaces(system.grid)
            if _bfaceregions[ibface] == ireg
                for iloc = 1:size(_bfacenodes, 1)
                    iglob = _bfacenodes[iloc, ibface]
                    system.node_dof[ispec, iglob] = ispec
                end
            end
        end
    end
end

# Create matrix in system and figure out if species
# distribution is homgeneous
function _complete!(system::AbstractSystem{Tv, Tc, Ti, Tm}; create_newtonvectors = true) where {Tv, Tc, Ti, Tm}
    if isdefined(system, :matrix)
        return
    end

    system.species_homogeneous = true
    species_added = false
    for inode = 1:size(system.node_dof, 2)
        for ispec = 1:size(system.node_dof, 1)
            if system.node_dof[ispec, inode] == ispec
                species_added = true
            else
                system.species_homogeneous = false
            end
        end
    end

    if (!species_added)
        error("No species enabled.\n Call enable_species(system,species_number, list_of_regions) at least once.")
    end

    nspec = size(system.node_dof, 1)
    n = num_dof(system)

    matrixtype = system.matrixtype
    #    matrixtype=:sparse
    # Sparse even in 1D is not bad, 

    if matrixtype == :default
        if !isdensesystem(system)
            matrixtype = :sparse
        else
            if nspec == 1
                matrixtype = :tridiagonal
            else
                matrixtype = :banded
            end
        end
    end

    if matrixtype == :tridiagonal
        system.matrix = Tridiagonal(zeros(Tv, n - 1), zeros(Tv, n), zeros(Tv, n - 1))
    elseif matrixtype == :banded
        system.matrix = BandedMatrix{Tv}(Zeros(n, n), (2 * nspec - 1, 2 * nspec - 1))
        # elseif matrixtype==:multidiagonal
        #     system.matrix=mdzeros(Tv,n,n,[-1,0,1]; blocksize=nspec)
    else # :sparse
        system.matrix = ExtendableSparseMatrix{Tv, Tm}(n, n)
    end

    if create_newtonvectors
        system.residual = unknowns(system)
        system.update = unknowns(system)
    end
    system.dudp = [unknowns(system) for i = 1:(system.num_parameters)]

    update_grid!(system)
    if has_generic_operator(system)
        if has_generic_operator_sparsity(system)
            system.generic_matrix = system.physics.generic_operator_sparsity(system)
        else
            generic_operator(f, u) = system.physics.generic_operator(f, u, system)
            input = rand(num_dof(system))
            output = similar(input)
            tdetect = @elapsed begin
                sparsity_pattern = Symbolics.jacobian_sparsity(generic_operator, output, input)
                system.generic_matrix = Float64.(sparse(sparsity_pattern))
            end
            println("sparsity detection for generic operator: $(tdetect) s")
            if nnz(system.generic_matrix) == 0
                error("Sparsity detection failed: no pattern found")
            end
        end
        tdetect = @elapsed begin
            system.generic_matrix_colors = matrix_colors(system.generic_matrix)
        end
        println("matrix coloring for generic operator: $(tdetect) s")
    end
end


function _complete_nomatrix!(system::AbstractSystem{Tv, Tc, Ti, Tm}; create_newtonvectors = true) where {Tv, Tc, Ti, Tm}
    #if isdefined(system, :matrix)
    #    return
    #end

    system.species_homogeneous = true
    species_added = false
    for inode = 1:size(system.node_dof, 2)
        for ispec = 1:size(system.node_dof, 1)
            if system.node_dof[ispec, inode] == ispec
                species_added = true
            else
                system.species_homogeneous = false
            end
        end
    end

    if (!species_added)
        error("No species enabled.\n Call enable_species(system,species_number, list_of_regions) at least once.")
    end

    nspec = size(system.node_dof, 1)
    n = num_dof(system)
	#=
    matrixtype = system.matrixtype
    #    matrixtype=:sparse
    # Sparse even in 1D is not bad, 

    if matrixtype == :default
        if !isdensesystem(system)
            matrixtype = :sparse
        else
            if nspec == 1
                matrixtype = :tridiagonal
            else
                matrixtype = :banded
            end
        end
    end

    if matrixtype == :tridiagonal
        system.matrix = Tridiagonal(zeros(Tv, n - 1), zeros(Tv, n), zeros(Tv, n - 1))
    elseif matrixtype == :banded
        system.matrix = BandedMatrix{Tv}(Zeros(n, n), (2 * nspec - 1, 2 * nspec - 1))
        # elseif matrixtype==:multidiagonal
        #     system.matrix=mdzeros(Tv,n,n,[-1,0,1]; blocksize=nspec)
    else # :sparse
        system.matrix = ExtendableSparseMatrix{Tv, Tm}(n, n)
    end
    =#

    if create_newtonvectors
        system.residual = unknowns(system)
        system.update = unknowns(system)
    end
    system.dudp = [unknowns(system) for i = 1:(system.num_parameters)]

    update_grid!(system)
    if has_generic_operator(system)
        if has_generic_operator_sparsity(system)
            system.generic_matrix = system.physics.generic_operator_sparsity(system)
        else
            generic_operator(f, u) = system.physics.generic_operator(f, u, system)
            input = rand(num_dof(system))
            output = similar(input)
            tdetect = @elapsed begin
                sparsity_pattern = Symbolics.jacobian_sparsity(generic_operator, output, input)
                system.generic_matrix = Float64.(sparse(sparsity_pattern))
            end
            println("sparsity detection for generic operator: $(tdetect) s")
            if nnz(system.generic_matrix) == 0
                error("Sparsity detection failed: no pattern found")
            end
        end
        tdetect = @elapsed begin
            system.generic_matrix_colors = matrix_colors(system.generic_matrix)
        end
        println("matrix coloring for generic operator: $(tdetect) s")
    end
end

"""
$(SIGNATURES)
Set generic operator sparsity, in the case where a generic operator has been
defined in physics.
"""
function generic_operator_sparsity!(system::AbstractSystem, sparsematrix::SparseMatrixCSC)
    system.generic_matrix = sparsematrix
end

"""
````
update_grid!(system; grid=system.grid)
````

Update grid (e.g. after rescaling of coordinates).
"""
function update_grid!(system::AbstractSystem; grid = system.grid)
    system.assembly_type == :cellwise ? update_grid_cellwise!(system, grid) : update_grid_edgewise!(system, grid)

    if length(system.physics.outflowboundaries) > 0
        bfacenodes = system.grid[BFaceNodes]
        bfaceregions = system.grid[BFaceRegions]
        outflownoderegions = ExtendableSparseMatrix{Bool, Int}(num_bfaceregions(system.grid),
                                                               num_nodes(system.grid))
        for ibface = 1:num_bfaces(system.grid)
            for ibn = 1:dim_space(system.grid)
                if bfaceregions[ibface] ∈ system.physics.outflowboundaries
                    outflownoderegions[bfaceregions[ibface], bfacenodes[ibn, ibface]] = true
                end
            end
        end
        system.outflownoderegions = SparseMatrixCSC(outflownoderegions)
    end
end

function update_grid_cellwise!(system::AbstractSystem{Tv, Tc, Ti, Tm}, grid) where {Tv, Tc, Ti, Tm}
    geom = grid[CellGeometries][1]
    csys = grid[CoordinateSystem]
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    bgeom = grid[BFaceGeometries][1]
    bfacenodes = grid[BFaceNodes]
    nbfaces = num_bfaces(grid)
    ncells = num_cells(grid)

    cellnodefactors = zeros(Tv, num_nodes(geom), ncells)
    celledgefactors = zeros(Tv, num_edges(geom), ncells)
    bfacenodefactors = zeros(Tv, num_nodes(bgeom), nbfaces)
    bfaceedgefactors = zeros(Tv, num_edges(bgeom), nbfaces)

    function cellwise_factors!(csys::Type{T}) where {T}
        nalloc = @allocated for icell = 1:ncells
            @views cellfactors!(geom, csys, coord, cellnodes, icell,
                                cellnodefactors[:, icell], celledgefactors[:, icell])
        end
        nalloc > 0 && @warn "$nalloc allocations in cell factor calculation"

        nalloc = @allocated for ibface = 1:nbfaces
            @views bfacefactors!(bgeom, csys, coord, bfacenodes, ibface,
                                 bfacenodefactors[:, ibface], bfaceedgefactors[:, ibface])
        end
        nalloc > 0 && @warn "$nalloc allocations in bface factor calculation"
    end

    cellwise_factors!(csys)

    system.assembly_data = CellwiseAssemblyData{Tc, Ti}(cellnodefactors, celledgefactors)
    system.boundary_assembly_data = CellwiseAssemblyData{Tc, Ti}(bfacenodefactors, bfaceedgefactors)
end

function update_grid_edgewise!(system::AbstractSystem{Tv, Tc, Ti, Tm}, grid) where {Tv, Tc, Ti, Tm}
	#if typeof(system.matrix) == ExtendableSparseMatrixParallel
	#	update_grid_edgewise_part!(system::AbstractSystem{Tv, Tc, Ti, Tm}, grid)
	#else
	update_grid_edgewise_seq!(system::AbstractSystem{Tv, Tc, Ti, Tm}, grid)
	#end
end

function update_grid_edgewise_seq!(system::AbstractSystem{Tv, Tc, Ti, Tm}, grid) where {Tv, Tc, Ti, Tm}
    geom = grid[CellGeometries][1]
    csys = grid[CoordinateSystem]
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    cellregions = grid[CellRegions]
    bgeom = grid[BFaceGeometries][1]
    bfacenodes = grid[BFaceNodes]
    bfaceedges = grid[BFaceEdges]
    nbfaces = num_bfaces(grid)
    ncells = num_cells(grid)

    celledges = grid[CellEdges]
    grid[EdgeNodes] # !!!workaround for bug in extendablegrids: sets num_edges right.
    bfacenodefactors = zeros(Tv, num_nodes(bgeom), nbfaces)
    bfaceedgefactors = zeros(Tv, num_edges(bgeom), nbfaces)
    cnf = ExtendableSparseMatrix{Tv, Ti}(num_cellregions(grid), num_nodes(grid))
    cef = ExtendableSparseMatrix{Tv, Ti}(num_cellregions(grid), num_edges(grid))

    nn::Int = num_nodes(geom)
    ne::Int = num_edges(geom)

    function edgewise_factors!(csys::Type{T}) where {T}
        cellnodefactors = zeros(Tv, nn)
        celledgefactors = zeros(Tv, ne)

        for icell = 1:ncells
            @views cellfactors!(geom, csys, coord, cellnodes, icell, cellnodefactors, celledgefactors)
            ireg = cellregions[icell]
            for inode = 1:nn
                cnf[ireg, cellnodes[inode, icell]] += cellnodefactors[inode]
            end

            for iedge = 1:ne
                cef[ireg, celledges[iedge, icell]] += celledgefactors[iedge]
            end
        end

        #        nalloc > 0 && @warn "$nalloc allocations in cell factor calculation"

        nalloc = @allocated for ibface = 1:nbfaces
            @views bfacefactors!(bgeom, csys, coord, bfacenodes, ibface,
                                 bfacenodefactors[:, ibface], bfaceedgefactors[:, ibface])
        end
        nalloc > 0 && @warn "$nalloc allocations in bface factor calculation"
    end

    edgewise_factors!(csys)

    system.assembly_data = EdgewiseAssemblyData{Tc, Ti}(SparseMatrixCSC(cnf), SparseMatrixCSC(cef))
    system.boundary_assembly_data = CellwiseAssemblyData{Tc, Ti}(bfacenodefactors, bfaceedgefactors)
end

"""
Much of this could and should be done in preparatory or smth.
"""
#=
function update_grid_edgewise_part!(system::AbstractSystem{Tv, Tc, Ti, Tm}, grid) where {Tv, Tc, Ti, Tm}
    geom = grid[CellGeometries][1]
    csys = grid[CoordinateSystem]
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    cellregions = grid[CellRegions]
    bgeom = grid[BFaceGeometries][1]
    bfacenodes = grid[BFaceNodes]
    bfaceedges = grid[BFaceEdges]
    nbfaces = num_bfaces(grid)
    ncells = num_cells(grid)

    celledges = grid[CellEdges]
    grid[EdgeNodes] # !!!workaround for bug in extendablegrids: sets num_edges right.
    bfacenodefactors = zeros(Tv, num_nodes(bgeom), nbfaces)
    bfaceedgefactors = zeros(Tv, num_edges(bgeom), nbfaces)
    
    nt = system.matrix.nt
    depth = system.matrix.depth
    cfp   = system.matrix.cellsforpart
    #cellparts = system.matrix.cellparts
    
    #cnf = [ExtendableSparseMatrix{Tv, Ti}(num_cellregions(grid), num_nodes(grid)) for i=nt*depth+1]
    #cef = [ExtendableSparseMatrix{Tv, Ti}(num_cellregions(grid), num_edges(grid)) for i=nt*depth+1]

    nn::Int = num_nodes(geom)
    ne::Int = num_edges(geom)

    function edgewise_factors!(csys::Type{T}) where {T}
    	nnp = zeros(Int, nt*depth+1) #number of nodes per partition
    	nep = zeros(Int, nt*depth+1) #number of edges per partition
    	
    	# count how many nodes & edges per partition
    	for part=1:nt*depth+1
    		tmp_n = []
    		tmp_e = []
    		for icell in cfp[part]
    			for inode in cellnodes[:,icell]
    				if !(inode in tmp_n)
    					push!(tmp_n, inode)
    				end
    			end
    		
    			for iedge in celledges[:,icell]
    				if !(iedge in tmp_e)
    					push!(tmp_e, iedge)
    				end
    			end
    		end
    		
    		nnp[part] = length(tmp_n)
    		nep[part] = length(tmp_e)
    	end
    	
    	# create local countings for edges and nodes
    	# should be in prep
    	
    	cnf = [ExtendableSparseMatrix{Tv, Ti}(num_cellregions(grid), nnp[i]) for i=nt*depth+1]
    	cef = [ExtendableSparseMatrix{Tv, Ti}(num_cellregions(grid), nep[i]) for i=nt*depth+1]
	
    		
    
        cellnodefactors = zeros(Tv, nn)
        celledgefactors = zeros(Tv, ne)

        for icell = 1:ncells
        	
        
            @views cellfactors!(geom, csys, coord, cellnodes, icell, cellnodefactors, celledgefactors)
            ireg = cellregions[icell]
            for inode = 1:nn
                cnf[cellparts[icell]][ireg, cellnodes[inode, icell]] += cellnodefactors[inode]
            end

            for iedge = 1:ne
                cef[cellparts[icell]][ireg, celledges[iedge, icell]] += celledgefactors[iedge]
            end
        end

        #        nalloc > 0 && @warn "$nalloc allocations in cell factor calculation"

        nalloc = @allocated for ibface = 1:nbfaces
            @views bfacefactors!(bgeom, csys, coord, bfacenodes, ibface,
                                 bfacenodefactors[:, ibface], bfaceedgefactors[:, ibface])
        end
        nalloc > 0 && @warn "$nalloc allocations in bface factor calculation"
    end

    edgewise_factors!(csys)

    system.assembly_data = EdgewiseAssemblyData{Tc, Ti}(SparseMatrixCSC.(cnf), SparseMatrixCSC.(cef))
    system.boundary_assembly_data = CellwiseAssemblyData{Tc, Ti}(bfacenodefactors, bfaceedgefactors)
end
=#

################################################################################################
# Degree of freedom handling

"""
$(SIGNATURES)
    
Check if species is defined in node.
"""
isnodespecies(system::AbstractSystem, ispec, inode) = system.node_dof[ispec, inode] == ispec

# This would works only
# for those calculated earlier in the right way, so depends on context
# => speaks for dispatching the whole assembly methods on the system type
# => speaks for enabling homogeneous systems as well
# isdof(system::SparseSystem,ispec,inode) = true

"""
    $(SIGNATURES)

Check if species is defined in region.
"""
isregionspecies(system::AbstractSystem, ispec, ireg) = system.region_species[ispec, ireg] > 0

"""
    getnodedof(system,ispec,inode)

Get active or dummy degree of freedom associated with node and species
"""
function getnodedof end

"""
    firstnodedof(system, inode)

Get first degree of freedom associated with node.
"""
function firstnodedof end

"""
    lastnodedof(system, inode)

Get last  degree of freedom associated with node.
"""
function lastnodedof end

"""
    getspecies(system,idof)

Get species associated to degree of freedom
"""
function getspecies end

firstnodedof(sys::DenseSystem{Tv}, K::Integer) where {Tv} = (K - 1) * num_species(sys) + 1
lastnodedof(sys::DenseSystem{Tv}, K::Integer) where {Tv} = K * num_species(sys)
getspecies(sys::DenseSystem{Tv}, idof) where {Tv} = mod1(idof, num_species(sys))
getnodedof(sys::DenseSystem{Tv}, ispec::Integer, inode::Integer) where {Tv} = (inode - 1) * num_species(sys) + ispec

firstnodedof(sys::SparseSystem, K::Integer) = sys.node_dof.colptr[K]
lastnodedof(sys::SparseSystem, K::Integer) = sys.node_dof.colptr[K + 1] - 1
getspecies(sys::SparseSystem, idof) = sys.node_dof.rowval[idof]
@inline function getnodedof(sys::SparseSystem{Tv, Tc, Ti}, ispec, inode) where {Tv, Tc, Ti}
    A = sys.node_dof
    coljfirstk = A.colptr[inode]
    coljlastk = A.colptr[inode + 1] - one(Ti)
    searchk = searchsortedfirst(A.rowval, ispec, coljfirstk, coljlastk, Base.Order.Forward)
    if searchk <= coljlastk && A.rowval[searchk] == ispec
        return searchk
    end
    return 0
end

##################################################################
"""
$(SIGNATURES)

Number of species in system
"""
num_species(system::AbstractSystem) = size(system.node_dof, 1)

##################################################################
"""
$(SIGNATURES)

Retrieve user data record.
"""
data(system::AbstractSystem) = system.physics.data

##################################################################
"""
$(SIGNATURES)

Set Dirichlet boundary condition for species ispec at boundary ibc:
    
``u_{ispec}=v`` on ``\\Gamma_{ibc}``

!!! info  
    Starting with version 0.14, it is preferable to define boundary conditions within the `bcondition` physics callback
"""
function boundary_dirichlet!(system::AbstractSystem, ispec, ibc, v)
    increase_num_species!(system, ispec)
    system.boundary_factors[ispec, ibc] = Dirichlet
    system.boundary_values[ispec, ibc] = v
end

"""
      boundary_dirichlet!(system; kwargs...)
Keyword argument version:
- `species`: species number
- `region`: region number
- `value`: value

!!! info  
    Starting with version 0.14, it is preferable to define boundary conditions within the `bcondition` physics callback
"""
function boundary_dirichlet!(system::AbstractSystem; species = 1, region = 1, value = 0)
    boundary_dirichlet!(system, species, region, value)
end

"""
     boundary_dirichlet!(y,u,bnode,ispec,ireg,val)

Set Dirichlet boundary condition for species ispec at boundary ibc.
"""
function boundary_dirichlet!(y, u, bnode, ispec, ireg, val; penalty = bnode.Dirichlet)
    if bnode.region == ireg
        y[ispec] += penalty * (u[ispec] - val)
        # just for call during initialization, so we can convert from dual number
        bnode.dirichlet_value[ispec] = value(val)
    end
    nothing
end

"""
     boundary_dirichlet!(y,u,bnode, args...; kwargs...)
Keyword argument version:
- `species`: species number. Default: 1
- `region`: boundary region number. By default, all boundary regions.
- `value`: value
"""
function boundary_dirichlet!(y, u, bnode, args...; species = 1, region = bnode.region, value = 0, penalty = bnode.Dirichlet)
    boundary_dirichlet!(y, u, bnode, species, region, value; penalty)
end

"""
       ramp(t; kwargs...)
Ramp function for specifying time dependent boundary conditions

Keyword arguments:
- `dt`: Tuple: start and end time of ramp. Default: `(0,0.1)`
- `du`: Tuple: values at start and end time. Default: `(0,0)`
"""
function ramp(t; dt = (0, 0.1), du = (0, 0))
    (t, ubegin, uend, tbegin, tend) = promote(Float64(t), du[1], du[2], dt[1], dt[2])
    if t < tbegin
        return ubegin
    elseif t < tend
        return ubegin + (uend - ubegin) * (t - tbegin) / (tend - tbegin)
    else
        return uend
    end
end

##################################################################
"""
$(SIGNATURES)

Set Neumann boundary condition for species ispec at boundary ibc:

``\\mathrm{flux}_{ispec}\\cdot \\vec n=v`` on ``\\Gamma_{ibc}``
!!! info  
    Starting with version 0.14, it is preferable to define boundary conditions within the `bcondition` physics callback
"""
function boundary_neumann!(system::AbstractSystem, ispec, ibc, v)
    increase_num_species!(system, ispec)
    system.boundary_factors[ispec, ibc] = 0.0
    system.boundary_values[ispec, ibc] = v
end

"""
      boundary_neumann!(system; kwargs...)
Keyword argument version:
- `species`: species number
- `region`: region number
- `value`: value
!!! info  
    Starting with version 0.14, it is preferable to define boundary conditions within the `bcondition` physics callback
"""
boundary_neumann!(system::AbstractSystem; species = 0, region = 0, value = 0) = boundary_neumann!(system, species, region, value)

"""
     boundary_neumann!(y,u,bnode,ispec,ireg,val)

Set Neumann boundary condition for species ispec at boundary ibc.
"""
boundary_neumann!(y, u, bnode, ispec, ireg, val) = bnode.region == ireg ? y[ispec] -= val : nothing

"""
     boundary_neumann!(y,u,bnode, args...; kwargs...)
Keyword argument version:
- `species`: species number. Default: 1
- `region`: boundary region number. By default, all boundary regions.
- `value`: value
"""
function boundary_neumann!(y, u, bnode, args...; species = 1, region = bnode.region, value = 0)
    boundary_neumann!(y, u, bnode, species, region, value)
end

##################################################################
"""
$(SIGNATURES)

Set Robin boundary condition for species ispec at boundary ibc:

``\\mathrm{flux}_{ispec}\\cdot \\vec n + \\alpha u_{ispec}=v`` on ``\\Gamma_{ibc}``

!!! info  
    Starting with version 0.14, it is preferable to define boundary conditions within the `bcondition` physics callback
"""
function boundary_robin!(system::AbstractSystem, ispec, ibc, α, v)
    increase_num_species!(system, ispec)
    system.boundary_factors[ispec, ibc] = α
    system.boundary_values[ispec, ibc] = v
end

"""
      boundary_robin!(system; kwargs...)
Keyword argument version:
- `species`: species number
- `region`: region number
- `factor`: factor
- `value`: value
!!! info  
    Starting with version 0.14, it is preferable to define boundary conditions within the `bcondition` physics callback
"""
function boundary_robin!(system::AbstractSystem; species = 0, region = 0, factor = 0, value = 0)
    boundary_robin!(system, species, region, factor, value)
end

"""
     boundary_robin!(y,u,bnode,ispec,ireg,fac,val)

Set Robin boundary condition for species ispec at boundary ibc.
"""
boundary_robin!(y, u, bnode, ispec, ireg, fac, val) = bnode.region == ireg ? y[ispec] += fac * u[ispec] - val : nothing

"""
     boundary_robin!(y,u,bnode, args...; kwargs...)
Keyword argument version:
- `species`: species number. Default: 1
- `region`: boundary region number. By default, all boundary regions.
- `factor`: factor
- `value`: value
"""
function boundary_robin!(y, u, bnode, args...; species = 1, region = bnode.region, factor = 0, value = 0)
    boundary_robin!(y, u, bnode, species, region, factor, value)
end

##################################################################
"""
$(SIGNATURES)

Number of species (size of first dimension) of solution array.
"""
num_species(a::AbstractArray) = size(a, 1)

#
# Initialize Dirichlet BC
#
function _initialize_dirichlet!(U::AbstractMatrix, system::AbstractSystem{Tv, Tc, Ti, Tm}; time = 0.0, λ = 0.0,
                                params::Vector{Tp} = Float64[]) where {Tv, Tp, Tc, Ti, Tm}
    _complete!(system; create_newtonvectors = true)
    nspecies = num_species(system)

    # set up bnode
    bnode = BNode(system, time, λ, params)
    bnodeparams = (bnode,)
    data = system.physics.data
    if isdata(data)
        bnodeparams = (bnode, data)
    end

    # setup unknowns to be passed
    UK = zeros(Tv, num_species(system) + length(params))
    for iparm = 1:length(params)
        UK[nspecies + iparm] = params[iparm]
    end
    u = unknowns(bnode, UK)

    # right hand side to be passed
    y = rhs(bnode, zeros(Tv, num_species(system)))

    # loop over all boundary faces
    for item in nodebatch(system.boundary_assembly_data)
        for ibnode in noderange(system.boundary_assembly_data, item)
            _fill!(bnode, system.boundary_assembly_data, ibnode, item)

            bnode.dirichlet_value .= Inf
            # set up solution vector, call boundary reaction
            @views UK[1:nspecies] .= U[:, bnode.index]
            system.physics.breaction(y, u, bnodeparams...)

            # Check for Dirichlet bc
            for ispec = 1:nspecies
                # Dirichlet bc given in breaction
                if !isinf(bnode.dirichlet_value[ispec])
                    U[ispec, bnode.index] = bnode.dirichlet_value[ispec]
                end

                # Dirichlet bc given after system creation (old API)
                if system.boundary_factors[ispec, bnode.region] ≈ Dirichlet
                    U[ispec, bnode.index] = system.boundary_values[ispec, bnode.region]
                end
            end
        end
    end
end

function _initialize!(U::AbstractMatrix, system::AbstractSystem; time = 0.0, λ = 0.0, params = Number[])
    _initialize_dirichlet!(U, system; time, λ, params)
    _initialize_inactive_dof!(U, system)
end

function _eval_and_assemble_inactive_species(system::AbstractSystem, U, Uold, F) end

function _eval_and_assemble_inactive_species(system::DenseSystem, U, Uold, F)
    if system.species_homogeneous
        return
    end
    for inode = 1:size(system.node_dof, 2)
        for ispec = 1:size(system.node_dof, 1)
            if !isnodespecies(system, ispec, inode)
                F[ispec, inode] += U[ispec, inode] - Uold[ispec, inode]
                idof = dof(F, ispec, inode)
                system.matrix[idof, idof] += 1.0
            end
        end
    end
end

function _initialize_inactive_dof!(U::AbstractMatrix, system::AbstractSystem) end

function _initialize_inactive_dof!(U::DenseSolutionArray, system::DenseSystem)
    if system.species_homogeneous
        return
    end
    for inode = 1:size(system.node_dof, 2)
        for ispec = 1:size(system.node_dof, 1)
            if !isnodespecies(system, ispec, inode)
                U[ispec, inode] = 0
            end
        end
    end
end

function Base.show(io::IO, sys::AbstractSystem)
    str = @sprintf("%s(num_species=%d)", typeof(sys), num_species(sys))
    println(io, str)
end

#####################################################
has_generic_operator(sys::AbstractSystem) = sys.physics.generic_operator != nofunc_generic
has_generic_operator_sparsity(sys::AbstractSystem) = sys.physics.generic_operator_sparsity != nofunc_generic_sparsity

##################################################################
"""
$(SIGNATURES)

Number of degrees of freedom for system.
"""
function num_dof(system::AbstractSystem) end

num_dof(system::SparseSystem) = nnz(system.node_dof)

num_dof(system::DenseSystem) = length(system.node_dof)

num_dof(a::DenseSolutionArray)=length(a)

num_dof(a::SparseSolutionArray)=nnz(a.node_dof)

"""
$(SIGNATURES)

Detect if array fits to the system.
"""
isunknownsof(u::Any, sys::AbstractSystem) = false
isunknownsof(u::DenseSolutionArray, sys::DenseSystem) = size(u) == size(sys.node_dof)
isunknownsof(u::SparseSolutionArray, sys::SparseSystem) = size(u) == size(sys.node_dof)

"""
$(SIGNATURES)

Create a solution vector for system.
If inival is not specified, the entries of the returned vector are undefined.
"""
unknowns(system::AbstractSystem{Tv}; inival = undef, inifunc = nothing) where {Tv} = unknowns(Tv, system; inival, inifunc)

"""
$(SIGNATURES)

Create a solution vector for system with elements of type `Tu`.
If inival is not specified, the entries of the returned vector are undefined.
"""
function unknowns(Tu::Type, system::AbstractSystem; inival = undef, inifunc = nothing) end

function unknowns(Tu::Type, system::SparseSystem; inival = undef, inifunc = nothing)
    a0 = Array{Tu}(undef, num_dof(system))
    if inival != undef
        fill!(a0, inival)
    end
    u = SparseSolutionArray(SparseMatrixCSC(system.node_dof.m,
                                            system.node_dof.n,
                                            system.node_dof.colptr,
                                            system.node_dof.rowval,
                                            a0))
    isa(inifunc, Function) && map!(inifunc, u, system)
    u
end

"""
    $(TYPEDEF)

Equationwise partitioning mode.
"""
struct Equationwise end

"""
    $(SIGNATURES)

Calculate partitioning of system unknowns.
"""
function partitioning(system::DenseSystem,::Equationwise)
    len = length(system.node_dof)
    nspec = size(system.node_dof, 1)
    [i:nspec:len for i = 1:nspec]
end

function partitioning(system::SparseSystem,::Equationwise)
    node_dof=system.node_dof
    nspec = size(node_dof, 1)
    nnodes = size(node_dof, 2)
    count=zeros(Int,nspec)
    colptr=node_dof.colptr
    rowval=node_dof.rowval
    nzval=node_dof.nzval

    # how many unknowns are there for each species ?
    # to make things fast we want to avoid push!
    for i=1:nnodes
        for j=colptr[i]:colptr[i+1]-1
            ispec=rowval[j]
            @assert ispec==nzval[j]
            if ispec==nzval[j]
                count[ispec]+=1
            end
        end
    end

    #
    # We no know how many parts are there
    # and can allocate
    #
    parts=[zeros(Int,c) for c in count]
    partcount=ones(Int,nspec)

    #
    # Sort unknown numbers into partitions
    #
    for i=1:nnodes
        for j=colptr[i]:colptr[i+1]-1
            ispec=rowval[j]
            @assert ispec==nzval[j]
            if ispec==nzval[j]
                parts[ispec][partcount[ispec]]=j
                partcount[ispec]+=1
            end
        end
    end
    parts
end



function unknowns(Tu::Type, system::DenseSystem; inival = undef, inifunc = nothing)
    a = Array{Tu}(undef, size(system.node_dof)...)
    if inival != undef
        fill!(a, inival)
    end
    isa(inifunc, Function) && map!(inifunc, a, system)
    return a
end

"""
$(SIGNATURES)

Create a solution vector for system using the callback `inifunc` which has the same
signature as a source term.
"""
function Base.map(inifunc::TF,
                  sys::System{Tv, Tc, Ti, Tm, TSpecMat, TSolArray}) where {Tv, Tc, Ti, Tm, TSpecMat, TSolArray, TF <: Function}
    unknowns(sys; inifunc)
end

"""
$(SIGNATURES)

Create a solution vector for system using a constant initial value
"""
function Base.map(inival::TI,
                  sys::System{Tv, Tc, Ti, Tm, TSpecMat, TSolArray}) where {Tv, Tc, Ti, Tm, TSpecMat, TSolArray, TI <: Number}
    unknowns(sys; inival)
end

"""
$(SIGNATURES)

Map `inifunc` onto solution array `U`
"""
function Base.map!(inifunc::TF,
                   U::AbstractMatrix{Tu},
                   system::System{Tv, Tc, Ti, Tm, TSpecMat, TSolArray}) where {Tu, Tv, Tc, Ti, Tm, TSpecMat, TSolArray, TF}
    isunknownsof(U, system) || error("U is not unknowns of system")
    _complete!(system)
    grid = system.grid
    node = Node(system, 0, 0, Tv[])
    nspecies::Int = num_species(system)
    UK = Array{Tu, 1}(undef, nspecies)

    for item in nodebatch(system.assembly_data)
        for inode in noderange(system.assembly_data, item)
            _fill!(node, system.assembly_data, inode, item)
            @views UK .= U[:, node.index]
            inifunc(unknowns(node, UK), node)
            K = node.index
            ireg = node.region
            for idof = firstnodedof(system, K):lastnodedof(system, K)
                ispec = getspecies(system, idof)
                if isregionspecies(system, ispec, ireg)
                    _set(U, idof, UK[ispec])
                end
            end
        end
    end
    U
end

"""
$(SIGNATURES)

Reshape vector to fit as solution to system.
"""
function Base.reshape(v::AbstractVector, system::AbstractSystem) end

Base.reshape(v::DenseSolutionArray, system::DenseSystem) = v

Base.reshape(v::SparseSolutionArray, sys::SparseSystem) = v

function Base.reshape(v::AbstractVector, sys::DenseSystem)
    @assert length(v) == num_dof(sys)
    nspec = num_species(sys)
    reshape(v, Int64(nspec), Int64(length(v) / nspec))
end

function Base.reshape(v::AbstractVector, system::SparseSystem)
    @assert length(v) == num_dof(system)
    SparseSolutionArray(SparseMatrixCSC(system.node_dof.m,
                                        system.node_dof.n,
                                        system.node_dof.colptr,
                                        system.node_dof.rowval,
                                        Vector(v)))
end

######################################
# History
"""
    history(sys)

Return solver history from last `solve` call, if `log` was set to true.
See  see [`NewtonSolverHistory`](@ref), [`TransientSolverHistory`](@ref).
"""
history(sys::AbstractSystem) = sys.history

"""
    history_details(sys)

Return details of solver history from last `solve` call, if `log` was set to true.
See [`details`](@ref).
"""
history_details(sys::AbstractSystem) = details(sys.history)

"""
    history_summary(sys)

Return summary of solver history from last `solve` call, if `log` was set to true.
"""
history_summary(sys::AbstractSystem) = summary(sys.history)

####################################################################
# LEGACY

"""
````
System(grid,physics; kwargs...)
````
Create system with physics record.
!!! info  
    Starting with version 0.14, all physics data can be passed directly to the system constructor
"""
function System(grid::ExtendableGrid, physics::Physics;
                valuetype = coord_type(grid),
                indextype = index_type(grid),
                unknown_storage = :dense,
                matrixindextype = Int64,
                kwargs...)
    system = System(grid; valuetype, indextype, unknown_storage, matrixindextype, kwargs...)
    physics!(system, physics)
end

"""
$(SIGNATURES)

Constructor for DenseSystem.
!!! compat  
    Will be removed in future versions
"""
function DenseSystem(grid, physics::Physics; matrixindextype = Int64)
    System(grid, physics; matrixindextype = matrixindextype, unknown_storage = :dense)
end

"""
$(SIGNATURES)

Constructor for SparseSystem.
!!! compat  
    Will be removed in future versions
"""
function SparseSystem(grid, physics::Physics; matrixindextype = Int64)
    System(grid, physics; matrixindextype = matrixindextype, unknown_storage = :sparse)
end
