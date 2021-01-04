
function ManifoldTests.test_reversediff(M::Manifold, pts, tv)
    return for (p, X) in zip(pts, tv)
        exp_f(t) = distance(M, p, exp(M, p, t[1] * X))
        d12 = norm(M, p, X)
        for t in 0.1:0.1:0.9
            Test.@test d12 ≈ ReverseDiff.gradient(exp_f, [t])[1]
        end

        retract_f(t) = distance(M, p, retract(M, p, t[1] * X))
        for t in 0.1:0.1:0.9
            Test.@test ReverseDiff.gradient(retract_f, [t])[1] ≥ 0
        end
    end
end
