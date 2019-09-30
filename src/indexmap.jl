#=
indexmap.jl
2019-08-29 Jeff Fessler, University of Michigan
=#

export hcat_new
export block_diag # use _ to avoid odd conflict with SparseArrays.blockdiag


"""
`check_index(row::AbstractVector{Int}, dims::Dims{2})`

Do not allow repeated index values because that would complicate the adjoint.
Insist on monotone increasing too because it is the easiest way to
ensure no repeats and is probably better for cache too.
"""
function check_index(index::AbstractVector{Int}, dimA::Int, dimB::Int)
#@show index, dimA, dimB
    length(index) != dimA && throw("invalid index length")
    minimum(index) < 0 && throw("negative index")
    maximum(index) > dimB &&
        throw("index $(maximum(index)) > dimB=$dimB")
    !(index isa UnitRange{Int64} ||
        ((index isa StepRange{Int64,Int64}) && (index.step > 0)) ||
        all(diff(index) .> 0)) && throw("non-monotone index")
    nothing
end


"""
`IndexMap`

The original motivation was to support construction of linear operators
of the form `B = [0 0 0; 0 A 0; 0 0 0]`
from a given linear operator `A`
where `size(B) == dims`

This version is more general than the form above
because it allows the rows and colums of `A`
to be interspersed among the rows and columns of `B` arbitrarily.
Essentially like the above form but with some permutation-like matrices
before and after, implicitly.
"""
struct IndexMap{T, As <: LinearMap, Rs <: AbstractVector{Int},
        Cs <: AbstractVector{Int}} <: LinearMap{T}
    map::As
    dims::Dims{2}
    rows::Rs # typically i1:i2 with 1 <= i1 <= i2 <= size(map,1)
    cols::Cs # typically j1:j2 with 1 <= j1 <= j2 <= size(map,2)

    function IndexMap{T,As,Rs,Cs}(map::As, dims::Dims{2},
            rows::Rs, cols::Cs) where
            {T, As <: LinearMap,
            Rs <: AbstractVector{Int}, Cs <: AbstractVector{Int}}

        check_index(rows, size(map,1), dims[1])
        check_index(cols, size(map,2), dims[2])

        return new{T,As,Rs,Cs}(map, dims, rows, cols)
    end
end

IndexMap{T}(map::As, dims::Dims{2} ; offset::Dims{2}) where
        {T, As <: LinearMap} =
    IndexMap{T, As, UnitRange{Int64}, UnitRange{Int64}}(map, dims,
        offset[1] .+ (1:size(map,1)),
        offset[2] .+ (1:size(map,2)))

IndexMap(map::LinearMap, dims::Dims{2} ; offset::Dims{2}) =
    IndexMap{eltype(map)}(map, dims, offset=offset)

IndexMap(map::LinearMap, dims::Dims{2},
        rows::AbstractVector{Int}, cols::AbstractVector{Int}) =
    IndexMap{eltype(map), typeof(map), typeof(rows), typeof(cols)}(
        map, dims, rows, cols)

Base.size(A::IndexMap) = A.dims


# Provide constructors via LinearMap

LinearMap(A::LinearMap, dims::Dims{2},
    index::Tuple{AbstractVector{Int}, AbstractVector{Int}}) =
        IndexMap(A, dims, index[1], index[2]) # given (rows,cols) tuple

LinearMap(A::LinearMap, dims::Dims{2} ; offset::Dims{2}) =
    IndexMap(A, dims ; offset=offset) # given offset

#LinearMap{T}(A::LinearMap, dims::Dims{2} ; offset::Dims{2}) where {T} =
#   IndexMap{T}(A, dims ; offset=offset) # given offset

#=
LinearMap(A::LinearMap ; offset::Dims{2}, dims::Dims{2}=size(A) .+ offset) =
    IndexMap(A, dims, offset) # possible alternative offset syntax
=#


#=
basic methods
=#

# the following symmetry rules are sufficient but perhaps not necessary
LinearAlgebra.issymmetric(A::IndexMap) = issymmetric(A.map) &&
    (A.dims[1] == A.dims[2]) && (A.rows == A.cols)

LinearAlgebra.ishermitian(A::IndexMap) = ishermitian(A.map) &&
    (A.dims[1] == A.dims[2]) && (A.rows == A.cols)

LinearAlgebra.ishermitian(A::IndexMap{<:Real}) = issymmetric(A)


# compare IndexMap objects
Base.:(==)(A::IndexMap, B::IndexMap) = (eltype(A) == eltype(B)) &&
    (A.map == B.map) && (A.dims == B.dims) &&
    (A.rows == B.rows) && (A.cols == B.cols)

# transpose/adjoint by simple swap
LinearAlgebra.transpose(A::IndexMap) =
    IndexMap(transpose(A.map), reverse(A.dims), A.cols, A.rows)
LinearAlgebra.adjoint(A::IndexMap) =
    IndexMap(adjoint(A.map), reverse(A.dims), A.cols, A.rows)


#=
combinations of IndexMap objects
=#

"""
(possibly) simplified version of hcat - initial draft
Requires 5-arg mul! to be efficient.
"""
function hcat_new(As::LinearMap...)
    rows = collect(map(A -> size(A,1), As))
    any(rows .!= rows[1]) && throw("row mismatch")
    cols = collect(map(A -> size(A,2), As))
    nmap = length(As)
    col_offsets = [0; cumsum(cols)[1:nmap-1]]
    dims = (rows[1], sum(cols))
    T = promote_type(map(eltype, As)...)
#   return +([LinearMap{T}(As[ii], dims ; offset=(0, col_offsets[ii]))
#       for ii in 1:nmap]...)
     return +([IndexMap{T}(As[ii], dims ; offset=(0, col_offsets[ii]))
         for ii in 1:nmap]...)
end


"""
`B = block_diag(As::LinearMap...)`

Block diagonal `LinearMap`, akin to `SparseArrays.blockdiag()`
Requires 5-arg mul! to be efficient.
"""
function block_diag(As::LinearMap...)
    rows = collect(map(A -> size(A,1), As))
    cols = collect(map(A -> size(A,2), As))
    dims = (sum(rows), sum(cols))
    nmap = length(As)
    col_offsets = [0; cumsum(cols)[1:nmap-1]]
    row_offsets = [0; cumsum(rows)[1:nmap-1]]
    T = promote_type(map(eltype, As)...)
    return +([IndexMap{T}(As[ii], dims ; offset=(row_offsets[ii], col_offsets[ii]))
        for ii in 1:nmap]...)
end


#=
multiplication with vectors

When IndexMaps are put in a LinearCombination, hopefully A_mul_B!
is called only on the "first" block, so that the overhead of `fill`
happens just once.
=#

#if VERSION < v"1.3.0-alpha.115"

"""
Multiply a single IndexMap with a vector.
Basic *(A,x) in LinearMaps.jl#L22 uses similar() so must zero `y` here.
In principle we could zero out just the complement of `A.rows`.
"""
function A_mul_B!(y::AbstractVector, A::IndexMap, x::AbstractVector)
    m, n = size(A)
#    @show "todoIM-AB:", size(A)
    @boundscheck (m == length(y) && n == length(x)) || throw(DimensionMismatch("A_mul_B!"))
    fill!(y, zero(eltype(y)))
    @views @inbounds mul!(y[A.rows], A.map, x[A.cols], true, true)
    return y
end


"""
This 5-arg mul! is based on LinearAlgebra.mul! in LinearMaps.jl
with modifications to support the indexing used by an IndexMap.
y = α B x + β y
"""
function LinearAlgebra.mul!(y_in::AbstractVector, B::IndexMap,
        x_in::AbstractVector, α::Number=true, β::Number=false)
    length(y_in) == size(B, 1) || throw(DimensionMismatch("mul!"))
    A = B.map
#    @show "todoIM:", size(B), α, β
    xv = @views @inbounds x_in[B.cols]
    yv = @views @inbounds y_in[B.rows]
    if isone(α) # α = 1, so B x + β y
        if iszero(β) # β = 0: basic y = B*x case
            fill!(y_in, zero(eltype(y_in)))
            A_mul_B!(yv, A, xv)
        elseif isone(β) # β = 1
            yv .+= A * xv
        else # β != 0, 1
            rmul!(y_in, β)
            yv .+= A * xv
        end # β-cases
    elseif iszero(α) # α = 0, so β y
        if iszero(β) # β = 0
             fill!(y_in, zero(eltype(y_in)))
        elseif !isone(β) # β != 1
            rmul!(y_in, β) # β != 0, 1
        end # β-cases
    else # α != 0, 1, so α B x + β y
        if iszero(β) # β = 0, so α B x
            fill!(y_in, zero(eltype(y_in)))
            A_mul_B!(yv, A, xv)
            rmul!(yv, α)
        else
            if !isone(β) # β != 0, 1
                rmul!(y_in, β)
            end
            yv .+= rmul!(A * xv, α)
        end # β-cases
    end # α-cases

    return y_in
end

#else # 5-arg mul! is available for matrices

    # throw("todo: not done")

#end # VERSION
