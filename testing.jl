dim = parse(Int, ARGS[1]) # dimension of grid
nna = parse(Int, ARGS[2]) # number of nodes per axis
p   = parse(Int, ARGS[3]) # number of threads

#if dim == 3
#    nm = (nna,nna,nna)
#else
#    nm = (nna, nna)
#end

#command = `julia BLAS_dot.jl $(nna^dim) $p`
#println(command)
#run(command)

# normal, non reordered matrix with ILUZero.jl 
command = `julia normal_matrix_test.jl $dim $nna $p`
run(command)

for parmatvecmul in [true, false]
    @info "$parmatvecmul"
    tmp = `julia -t $p reordered_matrix_test.jl $dim $nna $p $parmatvecmul`
    run(tmp)
end


#------------------------
#=


parmatvecmul = parse(Bool, ARGS[1])

if parmatvecmul
    include("test_solve_matvec_par.jl")
else
    include("test_solve_matvec_seq.jl")
end

=#

















