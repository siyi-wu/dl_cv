"""Check split/permutation consistency, model gradients and DDPM equations."""
import torch
from src.models import classifier, Denoiser, Diffusion


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    x = torch.randn(3,1,28,28)
    p = torch.randperm(784)
    assert torch.equal(x.flatten(1)[:,p][:,p.argsort()],x.flatten(1))
    # A fixed pixel permutation can be exactly absorbed by the MLP first layer.
    a,b=classifier('mlp'),classifier('mlp')
    b.load_state_dict(a.state_dict())
    with torch.no_grad(): b[1].weight.copy_(a[1].weight[:,p])
    assert torch.allclose(a(x),b(x.flatten(1)[:,p].reshape_as(x)),atol=1e-6)
    assert sum(v.numel() for v in a.parameters())==101770
    assert classifier('cnn')(x).shape==(3,10)
    model=Denoiser(32)
    process=Diffusion(200,torch.device('cpu'))
    t=torch.tensor([0,50,199]); noise=torch.randn_like(x)
    noisy=process.add_noise(x,t,noise)
    alpha=process.abar[t][:,None,None,None]
    assert torch.allclose((noisy-alpha.sqrt()*x)/(1-alpha).sqrt(),noise,atol=2e-5)
    loss=(model(noisy,t)-noise).square().mean(); loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    class Oracle(torch.nn.Module):
        def forward(self,z,t): return noise
    clean=x.clamp(-1,1)
    z=process.add_noise(clean,torch.zeros(3,dtype=torch.long),noise)
    out=process.reverse_step(Oracle(),z,0,torch.randn_like(z))
    assert torch.allclose(out,clean,atol=3e-4)
    # Validate posterior mean against the equivalent epsilon form away from clipping.
    step=75; z=process.add_noise(clean,torch.full((3,),step),noise)
    out=process.reverse_step(Oracle(),z,step,torch.zeros_like(z))
    expected=(z-process.beta[step]/(1-process.abar[step]).sqrt()*noise)/process.alpha[step].sqrt()
    assert torch.allclose(out,expected,atol=2e-5)
    model.eval()
    small=Diffusion(4,torch.device('cpu'))
    one,_=small.sample(model,x,123); two,_=small.sample(model,x,123)
    assert torch.equal(one,two) and torch.isfinite(one).all()
    print('PASS: permutation invariance, shapes, gradients, forward noise, DDPM posterior, deterministic sampling')


if __name__=='__main__': main()
