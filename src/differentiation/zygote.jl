struct ZygoteDiffBackend <: AbstractDiffBackend end

function Manifolds._gradient(f, p, ::ZygoteDiffBackend)
    return Zygote.gradient(f, p)[1]
end

function Manifolds._gradient!(f, X, p, ::ZygoteDiffBackend)
    return copyto!(X, Zygote.gradient(f, p)[1])
end

push!(Manifolds._diff_backends, ZygoteDiffBackend())
