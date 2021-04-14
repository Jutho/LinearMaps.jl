struct UniformScalingMap{T} <: LinearMap{T}
    λ::T
    M::Int
    function UniformScalingMap(λ::Number, M::Int)
        M < 0 && throw(ArgumentError("size of UniformScalingMap must be non-negative, got $M"))
        return new{typeof(λ)}(λ, M)
    end
end
@deprecate UniformScalingMap(λ::Number, M::Int, N::Int) UniformScalingMap(λ, M) false
@deprecate UniformScalingMap(λ::Number, sz::Dims{2}) UniformScalingMap(λ, sz[1]) false
# the following methods are misleading: they introduce the opportunity to pass 2 dims,
# but throw if the dims are unequal!
# UniformScalingMap(λ::Number, M::Int, N::Int) =
#     (M == N ?
#         UniformScalingMap(λ, M) : error("UniformScalingMap needs to be square"))
# UniformScalingMap(λ::T, sz::Dims{2}) where {T} =
#     (sz[1] == sz[2] ?
#         UniformScalingMap(λ, sz[1]) : error("UniformScalingMap needs to be square"))

MulStyle(::UniformScalingMap) = FiveArg()

# properties
Base.size(A::UniformScalingMap) = (A.M, A.M)
Base.isreal(A::UniformScalingMap) = isreal(A.λ)
LinearAlgebra.issymmetric(::UniformScalingMap) = true
LinearAlgebra.ishermitian(A::UniformScalingMap) = isreal(A)
LinearAlgebra.isposdef(A::UniformScalingMap) = isposdef(A.λ)

# comparison of UniformScalingMap objects
Base.:(==)(A::UniformScalingMap, B::UniformScalingMap) = (A.λ == B.λ && A.M == B.M)

# special transposition behavior
LinearAlgebra.transpose(A::UniformScalingMap) = A
LinearAlgebra.adjoint(A::UniformScalingMap)   = UniformScalingMap(conj(A.λ), A.M)

# multiplication with scalar
Base.:(*)(A::UniformScaling, B::LinearMap) = A.λ * B
Base.:(*)(A::LinearMap, B::UniformScaling) = A * B.λ
Base.:(*)(α::Number, J::UniformScalingMap) = UniformScalingMap(α * J.λ, J.M)
Base.:(*)(J::UniformScalingMap, α::Number) = UniformScalingMap(J.λ * α, J.M)
# needed for disambiguation
Base.:(*)(α::RealOrComplex, J::UniformScalingMap) = UniformScalingMap(α * J.λ, J.M)
Base.:(*)(J::UniformScalingMap, α::RealOrComplex) = UniformScalingMap(J.λ * α, J.M)

# multiplication with vector
Base.:(*)(J::UniformScalingMap, x::AbstractVector) =
    length(x) == J.M ? J.λ * x : throw(DimensionMismatch("*"))
# multiplication with map/matrix
Base.:(*)(J::UniformScalingMap, B::LinearMap) =
    size(B, 1) == J.M ? J.λ * B : throw(DimensionMismatch("*"))
Base.:(*)(J::UniformScalingMap, B::AbstractMatrix) =
    size(B, 1) == J.M ? J.λ * convert_to_lmap_(B) : throw(DimensionMismatch("*"))
Base.:(*)(A::LinearMap, J::UniformScalingMap) =
    size(A, 2) == J.M ? A * J.λ : throw(DimensionMismatch("*"))
Base.:(*)(A::VecOrMat, J::UniformScalingMap) =
    size(A, 2) == J.M ? convert_to_lmap_(A) * J.λ : throw(DimensionMismatch("*"))
Base.:(*)(U::UniformScalingMap, J::UniformScalingMap) =
    U.M == J.M ? UniformScalingMap(U.λ*J.λ, J.M) : throw(DimensionMismatch("*"))

# disambiguation
Base.:(*)(xc::LinearAlgebra.AdjointAbsVec, J::UniformScalingMap) = xc * J.λ
Base.:(*)(xt::LinearAlgebra.TransposeAbsVec, J::UniformScalingMap) = xt * J.λ

# multiplication with vector/matrix
for (In, Out) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix))
    @eval begin
        function _unsafe_mul!(y::$Out, J::UniformScalingMap, x::$In)
            _scaling!(y, J.λ, x, true, false)
            return y
        end
        function _unsafe_mul!(y::$Out, J::UniformScalingMap{<:RealOrComplex}, x::$In{<:RealOrComplex},
                    α::RealOrComplex, β::Number)
            _scaling!(y, J.λ * α, x, true, β)
            return y
        end
        function _unsafe_mul!(y::$Out, J::UniformScalingMap, x::$In,
                    α::Number, β::Number)
            _scaling!(y, J.λ, x, α, β)
            return y
        end
    end
end

function _scaling!(y, λ, x, α, β)
    if (iszero(α) || iszero(λ))
        iszero(β) && return fill!(y, zero(eltype(y)))
        isone(β) && return y
        return rmul!(y, β)
    elseif isone(α) && isone(λ)
        iszero(β) && return copyto!(y, x)
        isone(β) && return y .+= x
        return y .= y .* β .+ x
    elseif isone(α)
        iszero(β) && return y .= λ .* x
        isone(β) && return y .+= λ .* x
        return y .= y .* β .+ λ .* x
    elseif isone(λ)
        iszero(β) && return y .= x .* α
        isone(β) && return y .+= x .* α
        return y .= y .* β .+ x .* α
    else
        iszero(β) && return y .= λ .* x .* α
        isone(β) && return y .+= λ .* x .* α
        return y .= y .* β .+ λ .* x .* α
    end
end

# combine LinearMap and UniformScaling objects in linear combinations
Base.:(+)(A₁::LinearMap, A₂::UniformScaling) = A₁ + UniformScalingMap(A₂.λ, size(A₁, 1))
Base.:(+)(A₁::UniformScaling, A₂::LinearMap) = UniformScalingMap(A₁.λ, size(A₂, 1)) + A₂
Base.:(-)(A₁::LinearMap, A₂::UniformScaling) = A₁ - UniformScalingMap(A₂.λ, size(A₁, 1))
Base.:(-)(A₁::UniformScaling, A₂::LinearMap) = UniformScalingMap(A₁.λ, size(A₂, 1)) - A₂
