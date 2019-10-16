function local_metric_jacobian(M::MetricManifold, x)
    n = size(x, 1)
    ∂g = reshape(ForwardDiff.jacobian(x -> local_metric(M, x), x), n, n, n)
    return ∂g
end

function christoffel_symbols_second_jacobian(M::MetricManifold, x)
    n = size(x, 1)
    ∂Γ = reshape(
        ForwardDiff.jacobian(x -> christoffel_symbols_second(M, x), x),
        n, n, n, n
    )
    return ∂Γ
end
