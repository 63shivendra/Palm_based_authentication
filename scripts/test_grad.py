import torch
import torch.nn.functional as F

emb = torch.randn(16, 512, requires_grad=True, device='cuda')
w = torch.randn(200, 512, requires_grad=True, device='cuda')
emb_n = F.normalize(emb, p=2, dim=1)
w_n = F.normalize(w, p=2, dim=1)

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    cos = F.linear(emb_n, w_n)
    theta = torch.acos(torch.clamp(cos, -0.999, 0.999))
    loss = theta.sum()

loss.backward()
print('BF16 ArcFace acos grad norm:', emb.grad.norm().item(), 'Has NaN:', torch.isnan(emb.grad).any().item())

