struct ReverseDiffBackend <: AbstractDiffBackend end

function Manifolds._gradient(f, p, ::ReverseDiffBackend)
    return ReverseDiff.gradient(f, p)
end

function Manifolds._gradient!(f, X, p, ::ReverseDiffBackend)
    return ReverseDiff.gradient!(X, f, p)
end

push!(Manifolds._diff_backends, ReverseDiffBackend())
