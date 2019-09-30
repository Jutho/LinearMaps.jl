#=
test/indexmap.jl
2019-09-29 Jeff Fessler, University of Michigan
=#

using LinearAlgebra: mul!
using LinearMaps: LinearMap
using Random: seed!
using Test: @test, @testset

@testset "indexmap" begin
	m = 5; n = 6
	M1 = 10*(1:m) .+ (1:(n+1))'; L1 = LinearMap(M1)
	M2 = randn(m,n+2); L2 = LinearMap(M2)
	M3 = randn(m,n+3); L3 = LinearMap(M3)
	Mh = [M1 M2 M3]
	Lh = [L1 L2 L3]
	offset = (3,4)
	BM1 = [zeros(offset...) zeros(offset[1],size(M1,2));
		zeros(size(M1,1),offset[2]) M1]
	BL1 = @inferred LinearMap(L1, size(BM1) ; offset=offset)
	(s1,s2) = size(BM1)

	seed!(0)
	x = randn(s2)
	y = BM1 * x

	@test @inferred BL1 * x ≈ BM1 * x
	@test @inferred BL1' * y ≈ BM1' * y

	# todo: add @test_throws

	# test all flavors of 5-arg mul! for IndexMap
	for α in (0, 0.3, 1)
		for β in (0, 0.4, 1)
		#	@show α, β
			yl1 = randn(s1)
			ym1 = copy(yl1)
		#	mul!(ym1, BM1, x, α, β) # fails in 1.2
			ym1 = α * (BM1 * x) .+ β * ym1
			@test @inferred mul!(yl1, BL1, x, α, β) ≈ ym1
		end
	end

	@test @inferred Matrix(BL1) == BM1
	@test @inferred Matrix(BL1') == Matrix(BL1)'

#end


#@testset "hcat_new" begin

	x = randn(size(Mh,2))
	Bh = hcat_new(L1, L2, L3)
#	Bh = @inferred hcat_new(L1, L2, L3) # todo: type instability
	@test Mh == Matrix(Lh)
	@test Mh == Matrix(Bh)
	@test Matrix(Bh') == Matrix(Bh)'
	@test Mh * x ≈ Bh * x

end

#=
todo
@testset "blockdiag" begin

	using SparseArrays: blockdiag, sparse
	Md = Matrix(blockdiag(sparse.((M1, M2, M3, M2, M1))...))
	# Md = diag(M1, M2, M3)
	x = randn(size(Md,2))
	Bd = blockdiag(L1, L2, L3, L2, L1)
	@test @inferred Bd * x ≈ Md * x
	@test @inferred Matrix(Bd) == Md
	@test @inferred Matrix(Bd') == Matrix(Bd)'

end
=#