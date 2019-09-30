module LinearMaps

export LinearMap

using LinearAlgebra
using SparseArrays

if VERSION < v"1.2-"
    import Base: has_offset_axes
    require_one_based_indexing(A...) = !has_offset_axes(A...) || throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
else
    import Base: require_one_based_indexing
end

abstract type LinearMap{T} end

Base.eltype(::LinearMap{T}) where {T} = T

Base.isreal(A::LinearMap) = eltype(A) <: Real
LinearAlgebra.issymmetric(::LinearMap) = false # default assumptions
LinearAlgebra.ishermitian(A::LinearMap{<:Real}) = issymmetric(A)
LinearAlgebra.ishermitian(::LinearMap) = false # default assumptions
LinearAlgebra.isposdef(::LinearMap) = false # default assumptions

Base.ndims(::LinearMap) = 2
Base.size(A::LinearMap, n) = (n==1 || n==2 ? size(A)[n] : error("LinearMap objects have only 2 dimensions"))
Base.length(A::LinearMap) = size(A)[1] * size(A)[2]

Base.:(*)(A::LinearMap, x::AbstractVector) = mul!(similar(x, promote_type(eltype(A), eltype(x)), size(A, 1)), A, x)
function LinearAlgebra.mul!(y::AbstractVector, A::LinearMap, x::AbstractVector, α::Number=true, β::Number=false)
    length(y) == size(A, 1) || throw(DimensionMismatch("mul!"))
    if isone(α)
        iszero(β) && (A_mul_B!(y, A, x); return y)
        isone(β) && (y .+= A * x; return y)
        # β != 0, 1
        rmul!(y, β)
        y .+= A * x
        return y
    elseif iszero(α)
        iszero(β) && (fill!(y, zero(eltype(y))); return y)
        isone(β) && return y
        # β != 0, 1
        rmul!(y, β)
        return y
    else # α != 0, 1
        iszero(β) && (A_mul_B!(y, A, x); rmul!(y, α); return y)
        isone(β) && (y .+= rmul!(A * x, α); return y)
        # β != 0, 1
        rmul!(y, β)
        y .+= rmul!(A * x, α)
        return y
    end
end
# the following is of interest in, e.g., subspace-iteration methods
function LinearAlgebra.mul!(Y::AbstractMatrix, A::LinearMap, X::AbstractMatrix, α::Number=true, β::Number=false)
    (size(Y, 1) == size(A, 1) && size(X, 1) == size(A, 2) && size(Y, 2) == size(X, 2)) || throw(DimensionMismatch("mul!"))
    @inbounds @views for i = 1:size(X, 2)
        mul!(Y[:, i], A, X[:, i], α, β)
    end
    # starting from Julia v1.1, we could use the `eachcol` iterator
    # for (Xi, Yi) in zip(eachcol(X), eachcol(Y))
    #     mul!(Yi, A, Xi, α, β)
    # end
    return Y
end

A_mul_B!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector)  = mul!(y, A, x)
At_mul_B!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, transpose(A), x)
Ac_mul_B!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, adjoint(A), x)

# Matrix: create matrix representation of LinearMap
function Base.Matrix(A::LinearMap)
    M, N = size(A)
    T = eltype(A)
    mat = Matrix{T}(undef, (M, N))
    v = fill(zero(T), N)
    @inbounds for i = 1:N
        v[i] = one(T)
        mul!(view(mat, :, i), A, v)
        v[i] = zero(T)
    end
    return mat
end

Base.Array(A::LinearMap) = Matrix(A)
Base.convert(::Type{Matrix}, A::LinearMap) = Matrix(A)
Base.convert(::Type{Array}, A::LinearMap) = Matrix(A)
Base.convert(::Type{SparseMatrixCSC}, A::LinearMap) = sparse(A)

# sparse: create sparse matrix representation of LinearMap
function SparseArrays.sparse(A::LinearMap{T}) where {T}
    M, N = size(A)
    rowind = Int[]
    nzval = T[]
    colptr = Vector{Int}(undef, N+1)
    v = fill(zero(T), N)
    Av = Vector{T}(undef, M)

    for i = 1:N
        v[i] = one(T)
        mul!(Av, A, v)
        js = findall(!iszero, Av)
        colptr[i] = length(nzval) + 1
        if length(js) > 0
            append!(rowind, js)
            append!(nzval, Av[js])
        end
        v[i] = zero(T)
    end
    colptr[N+1] = length(nzval) + 1

    return SparseMatrixCSC(M, N, colptr, rowind, nzval)
end

include("wrappedmap.jl") # wrap a matrix of linear map in a new type, thereby allowing to alter its properties
include("uniformscalingmap.jl") # the uniform scaling map, to be able to make linear combinations of LinearMap objects and multiples of I
include("transpose.jl") # transposing linear maps
include("linearcombination.jl") # defining linear combinations of linear maps
include("composition.jl") # composition of linear maps
include("functionmap.jl") # using a function as linear map
include("blockmap.jl") # block linear maps
include("indexmap.jl") # single block map defined by row/col index

"""
    LinearMap(A; kwargs...)
    LinearMap{T=Float64}(f, [fc,], M::Int, N::Int = M; kwargs...)

Construct a linear map object, either from an existing `LinearMap` or `AbstractMatrix` `A`,
with the purpose of redefining its properties via the keyword arguments `kwargs`, or
from a function or callable object `f`. In the latter case, one also needs to specify
the size of the equivalent matrix representation `(M, N)`, i.e. for functions `f` acting
on length `N` vectors and producing length `M` vectors (with default value `N=M`). Preferably,
also the `eltype` `T` of the corresponding matrix representation needs to be specified, i.e.
whether the action of `f` on a vector will be similar to e.g. multiplying by numbers of type `T`.
If not specified, the devault value `T=Float64` will be assumed. Optionally, a corresponding
function `fc` can be specified that implements the transpose/adjoint of `f`.

The keyword arguments and their default values for functions `f` are
*   issymmetric::Bool = false : whether `A` or `f` acts as a symmetric matrix
*   ishermitian::Bool = issymmetric & T<:Real : whether `A` or `f` acts as a Hermitian matrix
*   isposdef::Bool = false : whether `A` or `f` acts as a positive definite matrix.
For existing linear maps or matrices `A`, the default values will be taken by calling
`issymmetric`, `ishermitian` and `isposdef` on the existing object `A`.

For functions `f`, there is one more keyword arguments
*   ismutating::Bool : flags whether the function acts as a mutating matrix multiplication
    `f(y,x)` where the result vector `y` is the first argument (in case of `true`),
    or as a normal matrix multiplication that is called as `y=f(x)` (in case of `false`).
    The default value is guessed by looking at the number of arguments of the first occurence
    of `f` in the method table.
"""
LinearMap(A::Union{AbstractMatrix, LinearMap}; kwargs...) = WrappedMap(A; kwargs...)
LinearMap(f, M::Int; kwargs...) = LinearMap{Float64}(f, M; kwargs...)
LinearMap(f, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, M, N; kwargs...)
LinearMap(f, fc, M::Int; kwargs...) = LinearMap{Float64}(f, fc, M; kwargs...)
LinearMap(f, fc, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, fc, M, N; kwargs...)

LinearMap{T}(A::Union{AbstractMatrix, LinearMap}; kwargs...) where {T} = WrappedMap{T}(A; kwargs...)
LinearMap{T}(f, args...; kwargs...) where {T} = FunctionMap{T}(f, args...; kwargs...)

end # module
