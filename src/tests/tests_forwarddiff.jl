

function ManifoldTests.test_forwarddiff(M::Manifold, pts, tv)
    return for (p, X) in zip(pts, tv)
        exp_f(t) = distance(M, p, exp(M, p, t[1] * X))
        d12 = norm(M, p, X)
        for t = 0.1:0.1:0.9
            Test.@test d12 ≈ ForwardDiff.derivative(exp_f, t)
        end

        retract_f(t) = distance(M, p, retract(M, p, t[1] * X))
        for t = 0.1:0.1:0.9
            Test.@test ForwardDiff.derivative(retract_f, t) ≥ 0
        end
    end
end
