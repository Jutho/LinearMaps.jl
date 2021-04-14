module LinearMaps

export LinearMap
export ⊗, kronsum, ⊕
export FillMap

using LinearAlgebra
import LinearAlgebra: mul!
using SparseArrays

if VERSION < v"1.2-"
    import Base: has_offset_axes
    require_one_based_indexing(A...) = !has_offset_axes(A...) || throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
else
    import Base: require_one_based_indexing
end

abstract type LinearMap{T} end

const MapOrVecOrMat{T} = Union{LinearMap{T}, AbstractVecOrMat{T}}
const MapOrMatrix{T} = Union{LinearMap{T}, AbstractMatrix{T}}
const RealOrComplex = Union{Real, Complex}

Base.eltype(::LinearMap{T}) where {T} = T

abstract type MulStyle end

struct FiveArg <: MulStyle end
struct ThreeArg <: MulStyle end

MulStyle(::FiveArg, ::FiveArg) = FiveArg()
MulStyle(::ThreeArg, ::FiveArg) = ThreeArg()
MulStyle(::FiveArg, ::ThreeArg) = ThreeArg()
MulStyle(::ThreeArg, ::ThreeArg) = ThreeArg()
MulStyle(::LinearMap) = ThreeArg() # default
@static if VERSION ≥ v"1.3.0-alpha.115"
    MulStyle(::AbstractVecOrMat) = FiveArg()
else
    MulStyle(::AbstractVecOrMat) = ThreeArg()
end
MulStyle(A::LinearMap, As::LinearMap...) = MulStyle(MulStyle(A), MulStyle(As...))

Base.isreal(A::LinearMap) = eltype(A) <: Real
LinearAlgebra.issymmetric(::LinearMap) = false # default assumptions
LinearAlgebra.ishermitian(A::LinearMap{<:Real}) = issymmetric(A)
LinearAlgebra.ishermitian(::LinearMap) = false # default assumptions
LinearAlgebra.isposdef(::LinearMap) = false # default assumptions

Base.ndims(::LinearMap) = 2
Base.size(A::LinearMap, n) =
    (n==1 || n==2 ? size(A)[n] : error("LinearMap objects have only 2 dimensions"))
Base.length(A::LinearMap) = size(A)[1] * size(A)[2]

# check dimension consistency for multiplication A*B
_iscompatible((A, B)) = size(A, 2) == size(B, 1)
function check_dim_mul(A, B)
    _iscompatible((A, B)) ||
        throw(DimensionMismatch("second dimension of left factor, $(size(A, 2)), " *
            "does not match first dimension of right factor, $(size(B, 1))"))
    return nothing
end
# check dimension consistency for multiplication C = A*B
function check_dim_mul(C, A, B)
    mA, nA = size(A) # A always has two dimensions
    mB, nB = size(B, 1), size(B, 2)
    (mB == nA && size(C, 1) == mA && size(C, 2) == nB) ||
        throw(DimensionMismatch("A has size ($mA,$nA), B has size ($mB,$nB), C has size $(size(C))"))
    return nothing
end

# conversion of AbstractVecOrMat to LinearMap
convert_to_lmap_(A::AbstractVecOrMat) = LinearMap(A)
convert_to_lmap_(A::LinearMap) = A
convert_to_lmaps() = ()
convert_to_lmaps(A) = (convert_to_lmap_(A),)
@inline convert_to_lmaps(A, B, Cs...) =
    (convert_to_lmap_(A), convert_to_lmap_(B), convert_to_lmaps(Cs...)...)

# The (internal) multiplication logic is as follows:
#  - `*(A, x)` calls `mul!(y, A, x)` for appropriately-sized y
#  - `mul!` checks consistency of the sizes, and calls `_unsafe_mul!`,
#    which does not check sizes, but potentially one-based indexing if necessary
#  - by default, `_unsafe_mul!` is redirected back to `mul!`
#  - custom map types only need to implement 3-arg (vector) `mul!`, and
#    everything else (5-arg multiplication, application to matrices,
#    conversion to matrices) will just work

"""
    *(A::LinearMap, x::AbstractVector)::AbstractVector

Compute the action of the linear map `A` on the vector `x`.

!!! compat "Julia 1.3"
    In Julia versions v1.3 and above, objects `L` of any subtype of `LinearMap`
    are callable in the sense that `L(x) = L*x` for `x::AbstractVector`.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); x=[1.0, 1.0];

julia> A*x
2-element Array{Float64,1}:
 3.0
 7.0

julia> A(x)
2-element Array{Float64,1}:
 3.0
 7.0
```
"""
function Base.:(*)(A::LinearMap, x::AbstractVector)
    check_dim_mul(A, x)
    return mul!(similar(x, promote_type(eltype(A), eltype(x)), size(A, 1)), A, x)
end
if VERSION ≥ v"1.3"
    (L::LinearMap)(x::AbstractVector) = L*x
end

"""
    mul!(Y::AbstractVecOrMat, A::LinearMap, B::AbstractVector) -> Y
    mul!(Y::AbstractMatrix, A::LinearMap, B::AbstractMatrix) -> Y

Calculates the action of the linear map `A` on the vector or matrix `B` and stores the
result in `Y`, overwriting the existing value of `Y`. Note that `Y` must not be aliased
with either `A` or `B`.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); B=[1.0, 1.0]; Y = similar(B); mul!(Y, A, B);

julia> Y
2-element Array{Float64,1}:
 3.0
 7.0

julia> A=LinearMap([1.0 2.0; 3.0 4.0]); B=[1.0 1.0; 1.0 1.0]; Y = similar(B); mul!(Y, A, B);

julia> Y
2×2 Array{Float64,2}:
 3.0  3.0
 7.0  7.0
```
"""
function mul!(y::AbstractVecOrMat, A::LinearMap, x::AbstractVector)
    check_dim_mul(y, A, x)
    return _unsafe_mul!(y, A, x)
end

"""
    mul!(C::AbstractVecOrMat, A::LinearMap, B::AbstractVector, α, β) -> C
    mul!(C::AbstractMatrix, A::LinearMap, B::AbstractMatrix, α, β) -> C

Combined inplace multiply-add ``A B α + C β``. The result is stored in `C` by overwriting
it. Note that `C` must not be aliased with either `A` or `B`.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); B=[1.0, 1.0]; C=[1.0, 3.0];

julia> mul!(C, A, B, 100.0, 10.0) === C
true

julia> C
2-element Array{Float64,1}:
 310.0
 730.0

julia> A=LinearMap([1.0 2.0; 3.0 4.0]); B=[1.0 1.0; 1.0 1.0]; C=[1.0 2.0; 3.0 4.0];

julia> mul!(C, A, B, 100.0, 10.0) === C
true

julia> C
2×2 Array{Float64,2}:
 310.0  320.0
 730.0  740.0
```
"""
function mul!(y::AbstractVecOrMat, A::LinearMap, x::AbstractVector, α::Number, β::Number)
    check_dim_mul(y, A, x)
    return _unsafe_mul!(y, A, x, α, β)
end

function _generic_mapvec_mul!(y, A, x, α, β)
    # this function needs to call mul! for, e.g.,  AdjointMap{...,<:CustomMap}
    if isone(α)
        iszero(β) && return mul!(y, A, x)
        z = A * x
        if isone(β)
            y .+= z
        else
            y .= y.*β .+ z
        end
        return y
    elseif iszero(α)
        iszero(β) && return fill!(y, zero(eltype(y)))
        isone(β) && return y
        # β != 0, 1
        return rmul!(y, β)
    else # α != 0, 1
        iszero(β) && return rmul!(mul!(y, A, x), α)
        z = A * x
        if isone(β)
            y .+= z .* α
        else
            y .= y .* β .+ z .* α
        end
        return y
    end
end

# the following is of interest in, e.g., subspace-iteration methods
function mul!(Y::AbstractMatrix, A::LinearMap, X::AbstractMatrix)
    check_dim_mul(Y, A, X)
    return _unsafe_mul!(Y, A, X)
end
function mul!(Y::AbstractMatrix, A::LinearMap, X::AbstractMatrix, α::Number, β::Number)
    check_dim_mul(Y, A, X)
    return _unsafe_mul!(Y, A, X, α, β)
end

function _generic_mapmat_mul!(Y, A, X, α=true, β=false)
    @views for i in 1:size(X, 2)
        _unsafe_mul!(Y[:, i], A, X[:, i], α, β)
    end
    # starting from Julia v1.1, we could use the `eachcol` iterator
    # for (Xi, Yi) in zip(eachcol(X), eachcol(Y))
    #     mul!(Yi, A, Xi, α, β)
    # end
    return Y
end

_unsafe_mul!(y, A::MapOrVecOrMat, x) = mul!(y, A, x)
_unsafe_mul!(y, A::AbstractVecOrMat, x, α, β) = mul!(y, A, x, α, β)
function _unsafe_mul!(y::AbstractVecOrMat, A::LinearMap, x::AbstractVector, α, β)
    return _generic_mapvec_mul!(y, A, x, α, β)
end
function _unsafe_mul!(y::AbstractMatrix, A::LinearMap, x::AbstractMatrix)
    return _generic_mapmat_mul!(y, A, x)
end
function _unsafe_mul!(y::AbstractMatrix, A::LinearMap, x::AbstractMatrix, α, β)
    return _generic_mapmat_mul!(y, A, x, α, β)
end

const LinearMapTuple = Tuple{Vararg{LinearMap}}

include("left.jl") # left multiplication by a transpose or adjoint vector
include("transpose.jl") # transposing linear maps
include("wrappedmap.jl") # wrap a matrix of linear map in a new type, thereby allowing to alter its properties
include("uniformscalingmap.jl") # the uniform scaling map, to be able to make linear combinations of LinearMap objects and multiples of I
include("linearcombination.jl") # defining linear combinations of linear maps
include("scaledmap.jl") # multiply by a (real or complex) scalar
include("composition.jl") # composition of linear maps
include("functionmap.jl") # using a function as linear map
include("blockmap.jl") # block linear maps
include("kronecker.jl") # Kronecker product of linear maps
include("fillmap.jl") # linear maps representing constantly filled matrices
include("indexablemap.jl") # indexable linear maps
include("conversion.jl") # conversion of linear maps to matrices
include("show.jl") # show methods for LinearMap objects

"""
    LinearMap(A::LinearMap; kwargs...)::WrappedMap
    LinearMap(A::AbstractVecOrMat; kwargs...)::WrappedMap
    LinearMap(J::UniformScaling, M::Int)::UniformScalingMap
    LinearMap{T=Float64}(f, [fc,], M::Int, N::Int = M; kwargs...)::FunctionMap

Construct a linear map object, either from an existing `LinearMap` or `AbstractVecOrMat` `A`,
with the purpose of redefining its properties via the keyword arguments `kwargs`;
a `UniformScaling` object `J` with specified (square) dimension `M`; from a `Number`
object to lazily represent filled matrices; or
from a function or callable object `f`. In the latter case, one also needs to specify
the size of the equivalent matrix representation `(M, N)`, i.e., for functions `f` acting
on length `N` vectors and producing length `M` vectors (with default value `N=M`).
Preferably, also the `eltype` `T` of the corresponding matrix representation needs to be
specified, i.e. whether the action of `f` on a vector will be similar to, e.g., multiplying
by numbers of type `T`. If not specified, the devault value `T=Float64` will be assumed.
Optionally, a corresponding function `fc` can be specified that implements the adjoint
(=transpose in the real case) of `f`.

The keyword arguments and their default values for the function-based constructor are:
*   `issymmetric::Bool = false` : whether `A` or `f` acts as a symmetric matrix
*   `ishermitian::Bool = issymmetric & T<:Real` : whether `A` or `f` acts as a Hermitian
    matrix
*   `isposdef::Bool = false` : whether `A` or `f` acts as a positive definite matrix.
For existing linear maps or matrices `A`, the default values will be taken by calling
`issymmetric`, `ishermitian` and `isposdef` on the existing object `A`.

For the function-based constructor, there is one more keyword argument:
*   `ismutating::Bool` : flags whether the function acts as a mutating matrix multiplication
    `f(y,x)` where the result vector `y` is the first argument (in case of `true`),
    or as a normal matrix multiplication that is called as `y=f(x)` (in case of `false`).
    The default value is guessed by looking at the number of arguments of the first
    occurrence of `f` in the method table.
"""
LinearMap(A::MapOrVecOrMat; getind=nothing, kwargs...) = _LinearMap(getind, A; kwargs...)
_LinearMap(::Nothing, A; kwargs...) = WrappedMap(A; kwargs...)
_LinearMap(getind, A; kwargs...) = IndexableMap(WrappedMap(A; kwargs...), getind)
LinearMap(J::UniformScaling, M::Int) = UniformScalingMap(J.λ, M)
LinearMap(f, M::Int; kwargs...) = LinearMap{Float64}(f, M; kwargs...)
LinearMap(f, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, M, N; kwargs...)
LinearMap(f, fc, M::Int; kwargs...) = LinearMap{Float64}(f, fc, M; kwargs...)
LinearMap(f, fc, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, fc, M, N; kwargs...)

LinearMap{T}(A::MapOrMatrix; kwargs...) where {T} = WrappedMap{T}(A; kwargs...)
LinearMap{T}(f, args...; kwargs...) where {T} = FunctionMap{T}(f, args...; kwargs...)

end # module
