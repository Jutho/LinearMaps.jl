struct KroneckerMap{T, As<:LinearMapTuple} <: LinearMap{T}
    maps::As
    function KroneckerMap{T}(maps::LinearMapTuple) where {T}
        for TA in Base.Generator(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in KroneckerMap constructor")
        end
        return new{T,typeof(maps)}(maps)
    end
end

"""
    kron(A::LinearMap, B::LinearMap)::KroneckerMap
    kron(A, B, Cs...)::KroneckerMap

Construct a (lazy) representation of the Kronecker product `A⊗B`. One of the two factors
can be an `AbstractMatrix`, which is then promoted to a `LinearMap` automatically.

To avoid fallback to the generic [`Base.kron`](@ref) in the multi-map case,
there must be a `LinearMap` object among the first 8 arguments in usage like
`kron(A, B, Cs...)`.

For convenience, one can also use `A ⊗ B` or `⊗(A, B, Cs...)` (typed as `\\otimes+TAB`) to construct the
`KroneckerMap`, even when all arguments are of `AbstractMatrix` type.

If `A`, `B`, `C` and `D` are linear maps of such size that one can form the matrix
products `A*C` and `B*D`, then the mixed-product property `(A⊗B)*(C⊗D) = (A*C)⊗(B*D)`
holds. Upon vector multiplication, this rule is checked for applicability.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
2×2 LinearMaps.UniformScalingMap{Bool} with scaling factor: true

julia> E = spdiagm(-1 => trues(1)); D = E + E' - 2I;

julia> Δ = kron(D, J) + kron(J, D); # discrete 2D-Laplace operator

julia> Matrix(Δ)
4×4 Array{Int64,2}:
 -4   1   1   0
  1  -4   0   1
  1   0  -4   1
  0   1   1  -4
```
"""
Base.kron(A::LinearMap, B::LinearMap) =
    KroneckerMap{promote_type(eltype(A), eltype(B))}((A, B))
Base.kron(A::LinearMap, B::KroneckerMap) =
    KroneckerMap{promote_type(eltype(A), eltype(B))}(tuple(A, B.maps...))
Base.kron(A::KroneckerMap, B::LinearMap) =
    KroneckerMap{promote_type(eltype(A), eltype(B))}(tuple(A.maps..., B))
Base.kron(A::KroneckerMap, B::KroneckerMap) =
    KroneckerMap{promote_type(eltype(A), eltype(B))}(tuple(A.maps..., B.maps...))
# hoist out scalings
Base.kron(A::ScaledMap, B::LinearMap) = A.λ * kron(A.lmap, B)
Base.kron(A::LinearMap, B::ScaledMap) = kron(A, B.lmap) * B.λ
Base.kron(A::ScaledMap, B::ScaledMap) = (A.λ * B.λ) * kron(A.lmap, B.lmap)
# reduce UniformScalingMaps
Base.kron(A::UniformScalingMap, B::UniformScalingMap) = UniformScalingMap(A.λ * B.λ, A.M * B.M)
# disambiguation
Base.kron(A::ScaledMap, B::KroneckerMap) = A.λ * kron(A.lmap, B)
Base.kron(A::KroneckerMap, B::ScaledMap) = kron(A, B.lmap) * B.λ
# generic definitions
Base.kron(A::LinearMap, B::LinearMap, C::LinearMap, Ds::LinearMap...) =
    kron(kron(A, B), C, Ds...)
Base.kron(A::AbstractVecOrMat, B::LinearMap) = kron(LinearMap(A), B)
Base.kron(A::LinearMap, B::AbstractVecOrMat) = kron(A, LinearMap(B))
# promote AbstractMatrix arguments to LinearMaps, then take LinearMap-Kronecker product
for k in 3:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A, n))::AbstractVecOrMat), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A, k))::LinearMap)
    # yields :Ak::LinearMap
    mapargs = ntuple(n -> :(LinearMap($(Symbol(:A, n)))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., :LinearMap(A(k-1)))

    @eval Base.kron($(Is...), $L, As::MapOrVecOrMat...) =
        kron($(mapargs...), $(Symbol(:A, k)), convert_to_lmaps(As...)...)
end

struct KronPower{p}
    function KronPower(p::Integer)
        p > 1 || throw(ArgumentError("the Kronecker power is only defined for exponents larger than 1, got $k"))
        return new{p}()
    end
end

"""
    ⊗(k::Integer)

Construct a lazy representation of the `k`-th Kronecker power
`A^⊗(k) = A ⊗ A ⊗ ... ⊗ A`, where `A` can be an `AbstractMatrix` or a `LinearMap`.
"""
⊗(k::Integer) = KronPower(k)

⊗(A, B, Cs...) = kron(convert_to_lmaps(A, B, Cs...)...)

Base.:(^)(A::MapOrMatrix, ::KronPower{p}) where {p} =
    kron(ntuple(n -> convert_to_lmap_(A), Val(p))...)

Base.size(A::KroneckerMap) = map(*, size.(A.maps)...)

LinearAlgebra.issymmetric(A::KroneckerMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap{<:Real}) = issymmetric(A)
LinearAlgebra.ishermitian(A::KroneckerMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerMap) = KroneckerMap{eltype(A)}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerMap) = KroneckerMap{eltype(A)}(map(transpose, A.maps))

Base.:(==)(A::KroneckerMap, B::KroneckerMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

#################
# multiplication helper functions
#################

@inline function _kronmul!(y, B, x, A, T)
    ma, na = size(A)
    mb, nb = size(B)
    X = reshape(x, (nb, na))
    Y = reshape(y, (mb, ma))
    if B isa UniformScalingMap
        _unsafe_mul!(Y, X, transpose(A))
        lmul!(B.λ, y)
    else
        temp = similar(Y, (ma, nb))
        _unsafe_mul!(temp, A, copy(transpose(X)))
        _unsafe_mul!(Y, B, transpose(temp))
    end
    return y
end
@inline function _kronmul!(y, B, x, A::UniformScalingMap, _)
    ma, na = size(A)
    mb, nb = size(B)
    iszero(A.λ) && return fill!(y, zero(eltype(y)))
    X = reshape(x, (nb, na))
    Y = reshape(y, (mb, ma))
    _unsafe_mul!(Y, B, X)
    !isone(A.λ) && rmul!(y, A.λ)
    return y
end
@inline function _kronmul!(y, B, x, A::VecOrMatMap, _)
    ma, na = size(A)
    mb, nb = size(B)
    X = reshape(x, (nb, na))
    Y = reshape(y, (mb, ma))
    At = transpose(A.lmap)
    if B isa UniformScalingMap
        # the following is (perhaps due to the reshape?) faster than
        # _unsafe_mul!(Y, B * X, At)
        _unsafe_mul!(Y, X, At)
        lmul!(B.λ, y)
    elseif nb*ma <= mb*na
        _unsafe_mul!(Y, B, X * At)
    else
        _unsafe_mul!(Y, Matrix(B * X), At)
    end
    return y
end
const VectorMap{T} = WrappedMap{T,<:AbstractVector}
const AdjOrTransVectorMap{T} = WrappedMap{T,<:LinearAlgebra.AdjOrTransAbsVec}
@inline _kronmul!(y, B::AdjOrTransVectorMap, x, a::VectorMap, _) = mul!(y, a.lmap, B.lmap * x)

#################
# multiplication with vectors
#################

const KroneckerMap2{T} = KroneckerMap{T, <:Tuple{LinearMap, LinearMap}}

function _unsafe_mul!(y::AbstractVecOrMat, L::KroneckerMap2, x::AbstractVector)
    require_one_based_indexing(y)
    A, B = L.maps
    _kronmul!(y, B, x, A, eltype(L))
    return y
end
function _unsafe_mul!(y::AbstractVecOrMat, L::KroneckerMap, x::AbstractVector)
    require_one_based_indexing(y)
    A = first(L.maps)
    B = kron(Base.tail(L.maps)...)
    _kronmul!(y, B, x, A, eltype(L))
    return y
end
# mixed-product rule, prefer the right if possible
# (A₁ ⊗ A₂ ⊗ ... ⊗ Aᵣ) * (B₁ ⊗ B₂ ⊗ ... ⊗ Bᵣ) = (A₁B₁) ⊗ (A₂B₂) ⊗ ... ⊗ (AᵣBᵣ)
function _unsafe_mul!(y::AbstractVecOrMat,
                        L::CompositeMap{<:Any,<:Tuple{KroneckerMap,KroneckerMap}},
                        x::AbstractVector)
    require_one_based_indexing(y)
    B, A = L.maps
    if length(A.maps) == length(B.maps) && all(_iscompatible, zip(A.maps, B.maps))
        _unsafe_mul!(y, kron(map(*, A.maps, B.maps)...), x)
    else
        _unsafe_mul!(y, LinearMap(A)*B, x)
    end
    return y
end
# mixed-product rule, prefer the right if possible
# (A₁⊗B₁) * (A₂⊗B₂) * ... * (Aᵣ⊗Bᵣ) = (A₁*A₂*...*Aᵣ) ⊗ (B₁*B₂*...*Bᵣ)
function _unsafe_mul!(y::AbstractVecOrMat,
                        L::CompositeMap{T, <:Tuple{Vararg{KroneckerMap2}}},
                        x::AbstractVector) where {T}
    require_one_based_indexing(y)
    As = map(AB -> AB.maps[1], L.maps)
    Bs = map(AB -> AB.maps[2], L.maps)
    As1, As2 = Base.front(As), Base.tail(As)
    Bs1, Bs2 = Base.front(Bs), Base.tail(Bs)
    apply = all(_iscompatible, zip(As1, As2)) && all(_iscompatible, zip(Bs1, Bs2))
    if apply
        _unsafe_mul!(y, kron(prod(As), prod(Bs)), x)
    else
        _unsafe_mul!(y, CompositeMap{T}(map(LinearMap, L.maps)), x)
    end
    return y
end

###############
# KroneckerSumMap
###############
struct KroneckerSumMap{T, As<:Tuple{LinearMap, LinearMap}} <: LinearMap{T}
    maps::As
    function KroneckerSumMap{T}(maps::Tuple{LinearMap,LinearMap}) where {T}
        A1, A2 = maps
        (size(A1, 1) == size(A1, 2) && size(A2, 1) == size(A2, 2)) ||
            throw(ArgumentError("operators need to be square in Kronecker sums"))
        for TA in Base.Generator(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in KroneckerSumMap constructor")
        end
        return new{T, typeof(maps)}(maps)
    end
end

"""
    kronsum(A, B)::KroneckerSumMap
    kronsum(A, B, Cs...)::KroneckerSumMap

Construct a (lazy) representation of the Kronecker sum `A⊕B = A ⊗ Ib + Ia ⊗ B`
of two square linear maps of type `LinearMap` or `AbstractMatrix`. Here, `Ia`
and `Ib` are identity operators of the size of `A` and `B`, respectively.
Arguments of type `AbstractMatrix` are automatically promoted to `LinearMap`.

For convenience, one can also use `A ⊕ B` or `⊕(A, B, Cs...)` (typed as
`\\oplus+TAB`) to construct the `KroneckerSumMap`.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
2×2 LinearMaps.UniformScalingMap{Bool} with scaling factor: true

julia> E = spdiagm(-1 => trues(1)); D = LinearMap(E + E' - 2I);

julia> Δ₁ = kron(D, J) + kron(J, D); # discrete 2D-Laplace operator, Kronecker sum

julia> Δ₂ = kronsum(D, D);

julia> Δ₃ = D^⊕(2);

julia> Matrix(Δ₁) == Matrix(Δ₂) == Matrix(Δ₃)
true
```
"""
kronsum(A::MapOrMatrix, B::MapOrMatrix) =
    KroneckerSumMap{promote_type(eltype(A), eltype(B))}(convert_to_lmaps(A, B))
kronsum(A::MapOrMatrix, B::MapOrMatrix, C::MapOrMatrix, Ds::MapOrMatrix...) =
    kronsum(A, kronsum(B, C, Ds...))

struct KronSumPower{p}
    function KronSumPower(p::Integer)
        p > 1 || throw(ArgumentError("the Kronecker sum power is only defined for exponents larger than 1, got $k"))
        return new{p}()
    end
end

"""
    ⊕(k::Integer)

Construct a lazy representation of the `k`-th Kronecker sum power `A^⊕(k) = A ⊕ A ⊕ ... ⊕ A`,
where `A` can be a square `AbstractMatrix` or a `LinearMap`.
"""
⊕(k::Integer) = KronSumPower(k)

⊕(a, b, c...) = kronsum(a, b, c...)

Base.:(^)(A::MapOrMatrix, ::KronSumPower{p}) where {p} =
    kronsum(ntuple(n->convert_to_lmap_(A), Val(p))...)

Base.size(A::KroneckerSumMap, i) = prod(size.(A.maps, i))
Base.size(A::KroneckerSumMap) = (size(A, 1), size(A, 2))

LinearAlgebra.issymmetric(A::KroneckerSumMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerSumMap{<:Real}) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerSumMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerSumMap) =
    KroneckerSumMap{eltype(A)}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerSumMap) =
    KroneckerSumMap{eltype(A)}(map(transpose, A.maps))

Base.:(==)(A::KroneckerSumMap, B::KroneckerSumMap) =
    (eltype(A) == eltype(B) && A.maps == B.maps)

function _unsafe_mul!(y::AbstractVecOrMat, L::KroneckerSumMap, x::AbstractVector)
    A, B = L.maps
    ma, na = size(A)
    mb, nb = size(B)
    X = reshape(x, (nb, na))
    Y = reshape(y, (nb, na))
    _unsafe_mul!(Y, X, transpose(A))
    _unsafe_mul!(Y, B, X, true, true)
    return y
end
