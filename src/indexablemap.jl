struct IndexableMap{T,A<:LinearMap{T},F} <: LinearMap{T}
    lmap::A
    getind::F
end

MulStyle(A::IndexableMap) = MulStyle(A.lmap)

Base.size(A::IndexableMap) = size(A.lmap)
LinearAlgebra.issymmetric(A::IndexableMap) = issymmetric(A.lmap)
LinearAlgebra.ishermitian(A::IndexableMap) = ishermitian(A.lmap)
LinearAlgebra.isposdef(A::IndexableMap) = isposdef(A.lmap)

Base.:(==)(A::IndexableMap, B::IndexableMap) = A.lmap == B.lmap

Base.adjoint(A::IndexableMap) = IndexableMap(adjoint(A.lmap), (i,j) -> adjoint(A.getind(j,i)))
Base.transpose(A::IndexableMap) = IndexableMap(transpose(A.lmap), (i,j) -> transpose(A.getind(j,i)))
# rewrapping preserves indexability but redefines, e.g., symmetry properties
LinearMap(A::IndexableMap; getind=nothing, kwargs...) = IndexableMap(LinearMap(A.lmap; kwargs...), getind)
# addition/subtraction/scalar multiplication preserve indexability
Base.:(+)(A::IndexableMap, B::IndexableMap) =
    IndexableMap(A.lmap + B.lmap, (i,j) -> A.getind(i,j) + B.getind(i,j))
Base.:(-)(A::IndexableMap, B::IndexableMap) =
    IndexableMap(A.lmap - B.lmap, (i,j) -> A.getind(i,j) - B.getind(i,j))
for typ in (RealOrComplex, Number)
    @eval begin
        Base.:(*)(α::$typ, A::IndexableMap) = IndexableMap(α * A.lmap, (i,j) -> α*A.getind(i,j))
        Base.:(*)(A::IndexableMap, α::$typ) = IndexableMap(A.lmap * α, (i,j) -> A.getind(i,j)*α)
    end
end
Base.:(*)(A::IndexableMap, J::UniformScalingMap) =
    size(A, 2) == J.M ? A*J.λ : throw(DimensionMismatch("*"))
Base.:(*)(J::UniformScalingMap, A::IndexableMap) =
    size(A, 1) == J.M ? J.λ*A : throw(DimensionMismatch("*"))

Base.@propagate_inbounds Base.getindex(A::IndexableMap, ::Colon, ::Colon) = A.getind(1:size(A, 1), 1:size(A, 2))
Base.@propagate_inbounds Base.getindex(A::IndexableMap, rows, ::Colon) = A.getind(rows, 1:size(A, 2))
Base.@propagate_inbounds Base.getindex(A::IndexableMap, ::Colon, cols) = A.getind(1:size(A, 1), cols)
Base.@propagate_inbounds Base.getindex(A::IndexableMap, rows, cols) = A.getind(rows, cols)

for (In, Out) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix))
    @eval begin
        function _unsafe_mul!(y::$Out, A::IndexableMap, x::$In)
            return _unsafe_mul!(y, A.lmap, x)
        end
        function _unsafe_mul!(y::$Out, A::IndexableMap, x::$In, α::Number, β::Number)
            return _unsafe_mul!(y, A.lmap, x, α, β)
        end
    end
end
