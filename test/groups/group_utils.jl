struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: AbstractManifold{ℝ} end

struct NotImplementedGroupDecorator{𝔽,M<:AbstractManifold{𝔽}} <:
       AbstractDecoratorManifold{𝔽}
    manifold::M
end
function active_traits(f, M::NotImplementedGroupDecorator, args...)
    return merge_traits(IsEmbeddedSubmanifold(), active_traits(f, M.manifold, args...))
end

function Manifolds.decorated_manifold(M::NotImplementedGroupDecorator)
    return M.manifold
end

struct DefaultTransparencyGroup{𝔽,M<:AbstractManifold{𝔽},A<:AbstractGroupOperation} <:
       AbstractDecoratorManifold{𝔽}
    manifold::M
    op::A
end
function active_traits(f, M::DefaultTransparencyGroup, args...)
    return merge_traits(
        Manifolds.IsGroupManifold(M.op),
        active_traits(f, M.manifold, args...),
    )
end

function Manifolds.decorated_manifold(M::DefaultTransparencyGroup)
    return M.manifold
end
