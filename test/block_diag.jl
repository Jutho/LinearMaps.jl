#=
test/block_diag.jl
2019-09-29 Jeff Fessler, University of Michigan
=#

using LinearMaps: LinearMap
using SparseArrays: sparse, blockdiag
using Random: seed!
using Test: @test, @testset

@testset "block_diag" begin
    m = 5; n = 6
    M1 = 10*(1:m) .+ (1:(n+1))'; L1 = LinearMap(M1)
    M2 = randn(m,n+2); L2 = LinearMap(M2)
    M3 = randn(m,n+3); L3 = LinearMap(M3)

    # Md = diag(M1, M2, M3, M2, M1) # unsupported so use sparse:
    Md = Matrix(blockdiag(sparse.((M1, M2, M3, M2, M1))...))
    x = randn(size(Md,2))
#   Bd = @inferred block_diag(L1, L2, L3, L2, L1) # todo: type instability
    Bd = block_diag(L1, L2, L3, L2, L1)
    @test Bd isa LinearMaps.LinearCombination
    @test @inferred Bd * x ≈ Md * x
    @test @inferred Matrix(Bd) == Md
    @test @inferred Matrix(Bd') == Matrix(Bd)'
end
