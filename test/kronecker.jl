using Test, LinearMaps, LinearAlgebra

@testset "kronecker product" begin
    A = rand(ComplexF64, 3, 3)
    B = rand(ComplexF64, 2, 2)
    K = kron(A, B)
    LA = LinearMap(A)
    LB = LinearMap(B)
    LK = @inferred kron(LA, LB)
    @test @inferred size(LK) == size(K)
    for i in (1, 2)
        @test @inferred size(LK, i) == size(K, i)
    end
    @test LK isa LinearMaps.KroneckerMap{ComplexF64}
    for transform in (identity, transpose, adjoint)
        @test Matrix(transform(LK)) ≈ transform(Matrix(LK)) ≈ transform(kron(A, B))
        @test Matrix(kron(transform(LA), transform(LB))) ≈ transform(kron(A, B))
        @test Matrix(transform(LinearMap(LK))) ≈ transform(Matrix(LK)) ≈ transform(kron(A, B))
    end
    @test kron(A, A, A) ≈ Matrix(@inferred kron(LA, LA, LA))
    K = @inferred kron(A, A, A, LA)
    @test K isa LinearMaps.KroneckerMap
    @test Matrix(K) ≈ kron(A, A, A, A)
    K4 = kron(A, B, B, LB)
    @test K4.maps[1].maps[1].maps[1].lmap === A
    @test @inferred kron(LA, LB)' == @inferred kron(LA', LB')
    @test kron(LA, B) == kron(LA, LB) == kron(A, LB)
    @test ishermitian(kron(LA'LA, LB'LB))
    A = rand(3, 3); B = rand(2, 2); LA = LinearMap(A); LB = LinearMap(B)
    @test issymmetric(kron(LA'LA, LB'LB))
    @test ishermitian(kron(LA'LA, LB'LB))
end