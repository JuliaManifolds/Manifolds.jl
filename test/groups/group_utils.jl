struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: Manifold{ℝ} end

struct NotImplementedGroupDecorator{M} <: AbstractDecoratorManifold{ℝ}
    manifold::M
end
