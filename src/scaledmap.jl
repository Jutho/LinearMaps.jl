"""
Lazy representation of a scaled map `λ * A = A * λ` with real or complex map
`A <: LinearMap{RealOrComplex}` and real or complex scaling factor
`λ <: RealOrComplex`.
"""
struct ScaledMap{T, S<:RealOrComplex, L<:LinearMap} <: LinearMap{T}
    λ::S
    lmap::L
    function ScaledMap{T}(λ::S, A::L) where {T, S <: RealOrComplex, L <: LinearMap{<:RealOrComplex}}
        @assert Base.promote_op(*, S, eltype(A)) == T "target type $T cannot hold products of $S and $(eltype(A)) objects"
        new{T,S,L}(λ, A)
    end
end

# constructor
ScaledMap(λ::RealOrComplex, lmap::LinearMap{<:RealOrComplex}) =
    ScaledMap{Base.promote_op(*, typeof(λ), eltype(lmap))}(λ, lmap)

# basic methods
Base.size(A::ScaledMap) = size(A.lmap)
Base.isreal(A::ScaledMap) = isreal(A.λ) && isreal(A.lmap)
LinearAlgebra.issymmetric(A::ScaledMap) = issymmetric(A.lmap)
LinearAlgebra.ishermitian(A::ScaledMap) = isreal(A.λ) && ishermitian(A.lmap)
LinearAlgebra.isposdef(A::ScaledMap) = isposdef(A.λ) && isposdef(A.lmap)

Base.transpose(A::ScaledMap) = A.λ * transpose(A.lmap)
Base.adjoint(A::ScaledMap) = conj(A.λ) * adjoint(A.lmap)

# comparison (sufficient, not necessary)
Base.:(==)(A::ScaledMap, B::ScaledMap) =
    eltype(A) == eltype(B) && A.lmap == B.lmap && A.λ == B.λ

# scalar multiplication and division
Base.:(*)(α::RealOrComplex, A::LinearMap{<:RealOrComplex}) = ScaledMap(α, A)
Base.:(*)(A::LinearMap{<:RealOrComplex}, α::RealOrComplex) = ScaledMap(α, A)

Base.:(*)(α::Number, A::ScaledMap) = (α * A.λ) * A.lmap
Base.:(*)(A::ScaledMap, α::Number) = A.lmap * (A.λ * α)
# needed for disambiguation
Base.:(*)(α::RealOrComplex, A::ScaledMap) = (α * A.λ) * A.lmap
Base.:(*)(A::ScaledMap, α::RealOrComplex) = (A.λ * α) * A.lmap
Base.:(-)(A::LinearMap) = -1 * A
Base.:(*)(J::UniformScalingMap, B::ScaledMap) =
    size(B, 1) == J.M ? J.λ * B : throw(DimensionMismatch("*"))
Base.:(*)(A::ScaledMap, J::UniformScalingMap) =
    size(A, 2) == J.M ? A * J.λ : throw(DimensionMismatch("*"))

# composition (not essential, but might save multiple scaling operations)
Base.:(*)(A::ScaledMap, B::ScaledMap) = (A.λ * B.λ) * (A.lmap * B.lmap)
Base.:(*)(A::ScaledMap, B::LinearMap) = A.λ * (A.lmap * B)
Base.:(*)(A::LinearMap, B::ScaledMap) = (A * B.lmap) * B.λ

# multiplication with vectors/matrices
for (In, Out) in ((AbstractVector, AbstractVecOrMat),
                  (AbstractMatrix, AbstractMatrix))
    @eval begin
        # commutative case
        function _unsafe_mul!(y::$Out, A::ScaledMap, x::$In{<:RealOrComplex})
            return _unsafe_mul!(y, A.lmap, x, A.λ, false)
        end
        function _unsafe_mul!(y::$Out, A::ScaledMap, x::$In{<:RealOrComplex}, α::Number, β::Number)
            return _unsafe_mul!(y, A.lmap, x, A.λ * α, β)
        end
        # non-commutative case
        function _unsafe_mul!(y::$Out, A::ScaledMap, x::$In)
            return lmul!(A.λ, _unsafe_mul!(y, A.lmap, x))
        end
    end
end
