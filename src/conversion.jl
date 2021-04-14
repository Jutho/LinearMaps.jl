# Matrix: create matrix representation of LinearMap
function Base.Matrix{T}(A::LinearMap) where {T}
    M, N = size(A)
    mat = Matrix{T}(undef, (M, N))
    v = fill(zero(T), N)
    @inbounds for i in 1:N
        v[i] = one(T)
        _unsafe_mul!(view(mat, :, i), A, v)
        v[i] = zero(T)
    end
    return mat
end
Base.Matrix(A::LinearMap{T}) where {T} = Matrix{T}(A)
Base.Array(A::LinearMap) = Matrix(A)
Base.convert(::Type{T}, A::LinearMap) where {T<:Matrix} = T(A)
Base.convert(::Type{Array}, A::LinearMap) = convert(Matrix, A)
Base.convert(::Type{AbstractMatrix}, A::LinearMap) = convert(Matrix, A)
Base.convert(::Type{AbstractArray}, A::LinearMap) = convert(AbstractMatrix, A)

# sparse: create sparse matrix representation of LinearMap
function SparseArrays.sparse(A::LinearMap{T}) where {T}
    M, N = size(A)
    rowind = Int[]
    nzval = T[]
    colptr = Vector{Int}(undef, N+1)
    v = fill(zero(T), N)
    Av = Vector{T}(undef, M)

    @inbounds for i in 1:N
        v[i] = one(T)
        _unsafe_mul!(Av, A, v)
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
Base.convert(::Type{SparseMatrixCSC}, A::LinearMap) = sparse(A)
SparseArrays.SparseMatrixCSC(A::LinearMap) = sparse(A)

# special cases

# ScaledMap
Base.Matrix{T}(A::ScaledMap{<:Any, <:Any, <:VecOrMatMap}) where {T} =
    convert(Matrix{T}, A.λ * convert(AbstractMatrix, A.lmap))
Base.convert(::Type{AbstractMatrix}, A::ScaledMap) = A.λ * convert(AbstractMatrix, A.lmap)
SparseArrays.sparse(A::ScaledMap{<:Any, <:Any, <:VecOrMatMap}) =
    A.λ * sparse(A.lmap.lmap)

# UniformScalingMap
Base.Matrix{T}(J::UniformScalingMap) where {T} = Matrix{T}(J.λ*I, size(J))
Base.convert(::Type{AbstractMatrix}, J::UniformScalingMap) = Diagonal(fill(J.λ, J.M))

# WrappedMap
Base.Matrix{T}(A::WrappedMap) where {T} = Matrix{T}(A.lmap)
Base.convert(::Type{T}, A::WrappedMap) where {T<:Matrix} = convert(T, A.lmap)
Base.Matrix{T}(A::VectorMap) where {T} = copyto!(Matrix{eltype(T)}(undef, size(A)), A.lmap)
Base.convert(::Type{T}, A::VectorMap) where {T<:Matrix} = T(A)
Base.convert(::Type{AbstractMatrix}, A::WrappedMap) = convert(AbstractMatrix, A.lmap)
SparseArrays.sparse(A::WrappedMap) = sparse(A.lmap)
Base.convert(::Type{SparseMatrixCSC}, A::WrappedMap) = convert(SparseMatrixCSC, A.lmap)

# TransposeMap & AdjointMap
for (T, t) in ((AdjointMap, adjoint), (TransposeMap, transpose))
    @eval Base.convert(::Type{AbstractMatrix}, A::$T) = $t(convert(AbstractMatrix, A.lmap))
    @eval SparseArrays.sparse(A::$T) = $t(convert(SparseMatrixCSC, A.lmap))
end

# LinearCombination
function Base.Matrix{T}(ΣA::LinearCombination{<:Any, <:Tuple{Vararg{VecOrMatMap}}}) where {T}
    maps = ΣA.maps
    mats = map(A->getfield(A, :lmap), maps)
    return Matrix{T}(sum(mats))
end
function SparseArrays.sparse(ΣA::LinearCombination{<:Any, <:Tuple{Vararg{VecOrMatMap}}})
    maps = ΣA.maps
    mats = map(A->getfield(A, :lmap), maps)
    return convert(SparseMatrixCSC, sum(mats))
end

# CompositeMap
function Base.Matrix{T}(AB::CompositeMap{<:Any, <:Tuple{VecOrMatMap, LinearMap}}) where {T}
    B, A = AB.maps
    require_one_based_indexing(B)
    Y = Matrix{eltype(AB)}(undef, size(AB))
    @views for i in 1:size(Y, 2)
        _unsafe_mul!(Y[:, i], A, B.lmap[:, i])
    end
    return Y
end
for ((TA, fieldA), (TB, fieldB)) in (((VecOrMatMap, :lmap), (VecOrMatMap, :lmap)),
                                     ((VecOrMatMap, :lmap), (UniformScalingMap, :λ)),
                                     ((UniformScalingMap, :λ), (VecOrMatMap, :lmap)))
    @eval function Base.convert(::Type{AbstractMatrix},
                                AB::CompositeMap{<:Any,<:Tuple{$TB,$TA}})
        B, A = AB.maps
        return A.$fieldA*B.$fieldB
    end
end
function Base.Matrix{T}(AB::CompositeMap{<:Any, <:Tuple{VecOrMatMap, VecOrMatMap}}) where {T}
    B, A = AB.maps
    return convert(Matrix{T}, A.lmap*B.lmap)
end
function SparseArrays.sparse(AB::CompositeMap{<:Any, <:Tuple{VecOrMatMap, VecOrMatMap}})
    B, A = AB.maps
    return convert(SparseMatrixCSC, A.lmap*B.lmap)
end
function Base.Matrix{T}(λA::CompositeMap{<:Any, <:Tuple{VecOrMatMap, UniformScalingMap}}) where {T}
    A, J = λA.maps
    return convert(Matrix{T}, J.λ*A.lmap)
end
function SparseArrays.sparse(λA::CompositeMap{<:Any, <:Tuple{VecOrMatMap, UniformScalingMap}})
    A, J = λA.maps
    return convert(SparseMatrixCSC, J.λ*A.lmap)
end
function Base.Matrix{T}(Aλ::CompositeMap{<:Any, <:Tuple{UniformScalingMap, VecOrMatMap}}) where {T}
    J, A = Aλ.maps
    return convert(Matrix{T}, A.lmap*J.λ)
end
function SparseArrays.sparse(Aλ::CompositeMap{<:Any, <:Tuple{UniformScalingMap, VecOrMatMap}})
    J, A = Aλ.maps
    return convert(SparseMatrixCSC, A.lmap*J.λ)
end

# BlockMap & BlockDiagonalMap
Base.Matrix{T}(A::BlockMap) where {T} = hvcat(A.rows, convert.(Matrix{T}, A.maps)...)
Base.convert(::Type{AbstractMatrix}, A::BlockMap) =
    hvcat(A.rows, convert.(AbstractMatrix, A.maps)...)
function SparseArrays.sparse(A::BlockMap)
    return hvcat(
        A.rows,
        convert(SparseMatrixCSC, first(A.maps)),
        convert.(AbstractMatrix, Base.tail(A.maps))...
    )
end
Base.Matrix{T}(A::BlockDiagonalMap) where {T} = Base._cat((1,2), convert.(Matrix{T}, A.maps)...)
Base.convert(::Type{AbstractMatrix}, A::BlockDiagonalMap) = sparse(A)
function SparseArrays.sparse(A::BlockDiagonalMap)
    return blockdiag(convert.(SparseMatrixCSC, A.maps)...)
end

# KroneckerMap & KroneckerSumMap
Base.Matrix{T}(A::KroneckerMap) where {T} = kron(convert.(Matrix{T}, A.maps)...)
Base.convert(::Type{AbstractMatrix}, A::KroneckerMap) =
    kron(convert.(AbstractMatrix, A.maps)...)
function SparseArrays.sparse(A::KroneckerMap)
    return kron(
        convert(SparseMatrixCSC, first(A.maps)),
        convert.(AbstractMatrix, Base.tail(A.maps))...
    )
end

function Base.Matrix{T}(L::KroneckerSumMap) where {T}
    A, B = L.maps
    IA = Diagonal(ones(Bool, size(A, 1)))
    IB = Diagonal(ones(Bool, size(B, 1)))
    return kron(Matrix{T}(A), IB) + kron(IA, Matrix{T}(B))
end
function Base.convert(::Type{AbstractMatrix}, L::KroneckerSumMap)
    A, B = L.maps
    IA = Diagonal(ones(Bool, size(A, 1)))
    IB = Diagonal(ones(Bool, size(B, 1)))
    return kron(convert(AbstractMatrix, A), IB) + kron(IA, convert(AbstractMatrix, B))
end
function SparseArrays.sparse(L::KroneckerSumMap)
    A, B = L.maps
    IA = sparse(Diagonal(ones(Bool, size(A, 1))))
    IB = sparse(Diagonal(ones(Bool, size(B, 1))))
    return kron(convert(AbstractMatrix, A), IB) + kron(IA, convert(AbstractMatrix, B))
end

# FillMap
Base.Matrix{T}(A::FillMap) where {T} = fill(T(A.λ), size(A))
