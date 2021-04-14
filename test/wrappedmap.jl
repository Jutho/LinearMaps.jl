using Test, LinearMaps, LinearAlgebra

@testset "wrapped maps" begin
    A = rand(10, 20)
    B = rand(ComplexF64, 10, 20)
    SA = A'A + I
    SB = B'B + I
    L = @inferred LinearMap{Float64}(A)
    @test summary(L) == "10×20 LinearMaps.WrappedMap{Float64}"
    @test occursin("10×20 LinearMaps.WrappedMap{Float64}", sprint((t, s) -> show(t, "text/plain", s), L))
    MA = @inferred LinearMap(SA)
    MB = @inferred LinearMap(SB)
    @test eltype(Matrix{Complex{Float32}}(LinearMap(A))) == ComplexF32
    @test size(L) == size(A)
    @test @inferred !issymmetric(L)
    @test @inferred issymmetric(MA)
    @test @inferred !issymmetric(MB)
    @test @inferred isposdef(MA)
    @test @inferred isposdef(MB)

    A = 2 * rand(ComplexF64, (20, 10)) .- 1
    v = rand(ComplexF64, 10)
    w = rand(ComplexF64, 20)
    u = rand(ComplexF64, 20, 1)
    V = rand(ComplexF64, 10, 3)
    W = rand(ComplexF64, 20, 3)
    Av = A * v
    AV = A * V
    M = @inferred LinearMap(A)
    N = @inferred LinearMap(M)

    @test @inferred M' * w == A' * w
    @test @inferred mul!(copy(V), adjoint(M), W) ≈ A' * W
    @test @inferred transpose(M) * w == transpose(A) * w
    @test @inferred transpose(LinearMap(transpose(M))) * v == A * v
    @test @inferred LinearMap(M')' * v == A * v
    @test @inferred(transpose(transpose(M))) === M
    @test @inferred(adjoint(adjoint(M))) === M
    Mherm = @inferred LinearMap(A'A)
    @test @inferred ishermitian(Mherm)
    @test @inferred !issymmetric(Mherm)
    @test @inferred !issymmetric(transpose(Mherm))
    @test @inferred ishermitian(transpose(Mherm))
    @test @inferred ishermitian(Mherm')
    @test @inferred isposdef(Mherm)
    @test @inferred isposdef(transpose(Mherm))
    @test @inferred isposdef(adjoint(Mherm))
    @test @inferred !(transpose(M) == adjoint(M))
    @test @inferred !(adjoint(M) == transpose(M))
    @test @inferred transpose(M') * v ≈ transpose(A') * v
    @test @inferred transpose(LinearMap(M')) * v ≈ transpose(A') * v
    @test @inferred LinearMap(transpose(M))' * v ≈ transpose(A)' * v
    @test @inferred transpose(LinearMap(transpose(M))) * v ≈ Av
    @test @inferred adjoint(LinearMap(adjoint(M))) * v ≈ Av

    for w in (vec(u), u)
        @test @inferred mul!(copy(w), transpose(LinearMap(M')), v) ≈ transpose(A') * v
        @test @inferred mul!(copy(w), LinearMap(transpose(M))', v) ≈ transpose(A)' * v
        @test @inferred mul!(copy(w), transpose(LinearMap(transpose(M))), v) ≈ Av
        @test @inferred mul!(copy(w), adjoint(LinearMap(adjoint(M))), v) ≈ Av
    end
    @test @inferred mul!(copy(W), transpose(LinearMap(M')), V) ≈ transpose(A') * V
    @test @inferred mul!(copy(W), LinearMap(transpose(M))', V) ≈ transpose(A)' * V
    @test @inferred mul!(copy(W), transpose(LinearMap(transpose(M))), V) ≈ AV
    @test @inferred mul!(copy(W), adjoint(LinearMap(adjoint(M))), V) ≈ AV
    @test @inferred mul!(copy(V), transpose(M), W) ≈ transpose(A) * W
    @test @inferred mul!(copy(V), adjoint(M), W) ≈ A' * W

    B = @inferred LinearMap(Symmetric(rand(10, 10)))
    @test transpose(B) == B
    @test B == transpose(B)

    B = @inferred LinearMap(Hermitian(rand(ComplexF64, 10, 10)))
    @test adjoint(B) == B
    @test B == B'

    N = 10
    Id = LinearMap(identity, N; issymmetric=true)
    A = rand(N, N)
    B = similar(A)
    @test mul!(copy(B), Id, A) == A
    @test mul!(B, A, Id) == B == A
    @test mul!(copy(B), A, LinearMap(Matrix(Id))) == A
    @test mul!(B, Id, A, true, true) ≈ 2A
    @test mul!(B, A, Id, true, true) == B == 3A
    @test mul!(copy(B), A, LinearMap(Matrix(Id)), true, true) == 4A
end
