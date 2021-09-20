struct ZygoteDiffBackend <: AbstractDiffBackend end

function Manifolds._gradient(f, p, ::ZygoteDiffBackend)
    return Zygote.gradient(f, p)
end

function Manifolds._gradient!(f, X, p, ::ZygoteDiffBackend)
    return Zygote.gradient!(X, f, p)
end

push!(Manifolds._diff_backends, ZygoteDiffBackend())
