using Test, LinearMaps, LinearAlgebra, BenchmarkTools

@testset "identity/scaling map" begin
    @test_throws ArgumentError LinearMaps.UniformScalingMap(true, -1)
    A = 2 * rand(ComplexF64, (10, 10)) .- 1
    B = rand(size(A)...)
    M = @inferred 1 * LinearMap(A)
    N = @inferred LinearMap(B)
    LC = @inferred M + N
    v = rand(ComplexF64, 10)
    w = similar(v)
    Id = @inferred LinearMap(I, 10)
    @test occursin("10×10 LinearMaps.UniformScalingMap{Bool}", sprint((t, s) -> show(t, "text/plain", s), Id))
    # @test_throws ErrorException LinearMaps.UniformScalingMap(1, 10, 20)
    # @test_throws ErrorException LinearMaps.UniformScalingMap(1, (10, 20))
    @test size(Id) == (10, 10)
    @test @inferred isreal(Id)
    @test @inferred issymmetric(Id)
    @test @inferred ishermitian(Id)
    @test @inferred isposdef(Id)
    @test Id * v == v
    @test (2 * M' + 3 * I) * v == 2 * A'v + 3v
    @test (3 * I + 2 * M') * v == 2 * A'v + 3v
    @test (2 * M' - 3 * I) * v == 2 * A'v - 3v
    @test (3 * I - 2 * M') * v == -2 * A'v + 3v
    @test (3 * I - 2 * M') * v == -2 * A'v + 3v
    @test transpose(LinearMap(2 * M' + 3 * I)) * v ≈ transpose(2 * A' + 3 * I) * v
    @test LinearMap(2 * M' + 0I)' * v ≈ (2 * A')' * v
    for λ in (0, 1, rand()), α in (0, 1, rand()), β in (0, 1, rand()), sz in (10, (10,5))
        Λ = @inferred LinearMap(λ*I, 10)
        x = rand(Float64, sz)
        y = rand(Float64, sz)
        if testallocs
            b = @benchmarkable mul!($y, $Λ, $x, $α, $β)
            @test run(b, samples=3).allocs == 0
        end
        y = deepcopy(x)
        @inferred mul!(y, Λ, x, α, β)
        @test y ≈ λ * x * α + β * x
    end
    for elty in (Float64, ComplexF64), transform in (identity, transpose, adjoint)
        λ = rand(elty)
        x = rand(10)
        J = @inferred LinearMap(LinearMaps.UniformScalingMap(λ, 10))
        @test transform(J) * x == transform(λ) * x
        J = @inferred LinearMap(λ*I, 10)
        @test (λ * J) * x == (J * λ) * x == (λ * λ) * x
    end
    X = rand(10, 10); Y = similar(X)
    @test mul!(Y, Id, X) == X
    @test Id*X isa LinearMap
    @test X*Id isa LinearMap
    @test Matrix(Id*X) == X
    @test Matrix(X*Id) == X
end
