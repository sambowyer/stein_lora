import torch
from torch import Tensor
from torch.optim import AdamW
from .multi_lora import MultiLoraLayer, nn_ParallelLinear


class SVGD():
    def __init__(self, model, lr, sigma=1e-3, gamma=0.1, damping_lambda=1, base_optimizer=AdamW, base_optimizer_kwargs={}):
        self.sigma = sigma
        self.gamma = gamma
        self.damping_lambda = damping_lambda

        self.lr = lr

        self.base_optimizer = base_optimizer(model.parameters(),
                                             lr=lr if lr > 0 else -lr, # hack to allow negative learning rates
                                             **base_optimizer_kwargs)

        self.lora_weights = [(module.lora_A[model.active_adapter].weight, module.lora_B[model.active_adapter].weight) for module in model.modules() if isinstance(module, MultiLoraLayer)]

        self.step_count = 0

    def step(self):
        for A, B in self.lora_weights:
            update_A, update_B = svgd_step(A, B, self.sigma, self.gamma, self.damping_lambda, e=self.step_count)

            if self.lr < 0:
                update_A *= -1
                update_B *= -1

            A.grad = update_A
            B.grad = update_B

        self.base_optimizer.step()

        self.step_count += 1        

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def accelerate_prepare(self, accelerator, lr_scheduler=None):
        if lr_scheduler is not None:
            self.base_optimizer, lr_scheduler = accelerator.prepare(self.base_optimizer, lr_scheduler)
            return lr_scheduler
        else:
            self.base_optimizer = accelerator.prepare(self.base_optimizer)

@torch.enable_grad()
def svgd_step(A : Tensor, B : Tensor, sigma, gamma, damping_lambda=1, e=-1):
    '''
    A: torch.Tensor of shape (K, r, d_in)
    B: torch.Tensor of shape (K, d_out, r)
    sigma: float or 'auto'
    gamma: float

    Returns:
    update_A: torch.Tensor of shape (K, r, d_in)
    update_B: torch.Tensor of shape (K, d_out, r)
    '''

    # check shapes
    K = A.shape[0]
    assert K == B.shape[0]

    r = A.shape[1]
    assert r == B.shape[2]

    d_in = A.shape[2]
    d_out = B.shape[1]

    device = A.device

    # get negative log likelihood (loss) gradients
    neg_log_lik_grad_A = A.grad.clone()
    neg_log_lik_grad_B = B.grad.clone()

    # breakpoint()
    # reset grads
    A.grad.zero_()
    B.grad.zero_()

    # (improper) uniform prior
    A_log_prior_grad_ = 0 #torch.zeros((K,), device=device)
    B_log_prior_grad_ = 0 #torch.zeros((K,), device=device)

    # precompute AAt and BtB as (K, K, r, r) tensors
    AAt = torch.matmul(A.unsqueeze(1), A.unsqueeze(0).transpose(2, 3))
    BtB = torch.matmul(B.unsqueeze(1).transpose(2, 3), B.unsqueeze(0))

    # precompute MSDs using the trace trick i.e.
    #   MSDs[i,j] =     torch.dot(AAt[i,i].flatten(), BtB[i,i].flatten()) 
    #               +   torch.dot(AAt[j,j].flatten(), BtB[j,j].flatten())
    #               - 2*torch.dot(AAt[i,j].flatten(), BtB[i,j].flatten())

    traces = (AAt.reshape((K, K, r*r)) * BtB.reshape((K, K, r*r))).sum(-1)
    traces_diag = traces.diagonal()
    MSDs = (traces_diag[:, None] + traces_diag[None, :] - 2 * traces) / (d_in * d_out)

    if sigma == 'auto':
        # use .quantile(0.5) instead of .median to get mean of two middle values (pytorch.median takes lower of the two)
        sigma = (MSDs[*torch.triu_indices(K,K,1)].quantile(0.5) / (2*torch.log(torch.tensor(K, device=device))).sqrt()).item()

        if sigma == 0:
            sigma = 1e-18

    # kernel_matrix = torch.exp(-MSDs / (2*sigma**2))
    kernel_matrix = torch.exp(-MSDs / sigma)

    # construct the updates
    # first the driving force term
    update_A = torch.einsum('ij,jkl->ikl', kernel_matrix, neg_log_lik_grad_A + A_log_prior_grad_) / K
    update_B = torch.einsum('ij,jkl->ikl', kernel_matrix, neg_log_lik_grad_B + B_log_prior_grad_) / K

    # add damping term
    update_A -= ((1 - damping_lambda) / K) * neg_log_lik_grad_A
    update_B -= ((1 - damping_lambda) / K) * neg_log_lik_grad_B

    # then the repulsive force term 
    kernel_grad_A, kernel_grad_B = torch.autograd.grad(kernel_matrix.sum(), (A, B))

    rep_A = kernel_grad_A * gamma / K
    rep_B = kernel_grad_B * gamma / K

    update_A += rep_A
    update_B += rep_B

    # print(f"ΔA ~ {update_A.abs().mean():.3g}, ΔB ~ {update_B.abs().mean():.3g}, sigma ~ {sigma:.3g}")
    # print(f" A ~ {A.abs().mean():.3g},  B ~ {B.abs().mean():.3g}")

    # if e % 10 == 0 and e>1:
    #     breakpoint()

    return update_A, update_B
