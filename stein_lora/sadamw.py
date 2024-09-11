import torch
from torch import Tensor
# from .svgd import svgd_step
from .multi_lora import MultiLoraLayer
from torch.optim.optimizer import _get_scalar_dtype

# def _get_scalar_dtype(is_fused=None):
#     if is_fused:
#         return torch.float32
#     return (
#         torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
#     )



class SAdamW(torch.optim.AdamW):
    '''This class implements AdamW but using SVGD updates in place of gradients'''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, sigma=1e-3, gamma=0.1, low_mem=False):
        super(SAdamW, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.sigma = sigma
        self.gamma = gamma

        self.low_mem = low_mem

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,  # N.B. we're going to replace these grads with the SVGD updates
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False

        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            
            ##################### SVGD update #####################
            # grads.append(p.grad)
            # breakpoint()

            ######## (rest of code left the same as AdamW) ########
            

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError(
                    "`requires_grad` is not supported for `step` in differentiable mode"
                )

            # Foreach without capturable does not support a tensor lr
            if (
                group["foreach"]
                and isinstance(group["lr"], Tensor)
                and not group["capturable"]
            ):
                raise RuntimeError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )

            state_steps.append(state["step"])

        ######################## NEW CODE ###########################
        assert len(params_with_grad) % 2 == 0
        for i in range(len(params_with_grad) // 2):
            A = params_with_grad[2*i]
            B = params_with_grad[2*i+1]

            assert A.shape[:-2] == B.shape[:-2]
            assert A.shape[-2:] == B.shape[-2:][::-1]

            if self.low_mem:
                update_A, update_B = svgd_step_low_mem(A, B, self.sigma, self.gamma)
            else:
                update_A, update_B, sigma = svgd_step(A, B, self.sigma, self.gamma, e=state["step"])

                # up_A, up_B = svgd_step_low_mem(A, B, sigma, self.gamma)

            assert update_A.shape == A.shape
            assert update_B.shape == B.shape 

            # breakpoint()

            grads.extend([update_A, update_B])

        #############################################################


        return has_complex

@torch.enable_grad()
def svgd_step(A : Tensor, B : Tensor, sigma, gamma, e=-1):
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

    # get log likelihood (loss) gradients
    log_lik_grad_A = A.grad.clone()
    log_lik_grad_B = B.grad.clone()

    # reset grads
    A.grad.zero_()
    B.grad.zero_()

    # with torch.no_grad():
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
        sigma = (MSDs[*torch.triu_indices(K,K,1)].median() / (2*torch.log(torch.tensor(K, device=device))).sqrt()).item()

        if sigma == 0:
            sigma = 1e-18

    # kernel_matrix = torch.exp(-MSDs / (2*sigma**2))
    kernel_matrix = torch.exp(-MSDs / sigma)
    # breakpoint()
    # kernel_matrix.sum().backward()
    Ag, Bg = torch.autograd.grad(kernel_matrix.sum(), (A, B))
    autograd_rep_A = Ag * gamma / K
    autograd_rep_B = Bg * gamma / K

    # construct the updates
    # first the driving force term
    update_A = torch.einsum('ij,jkl->ikl', kernel_matrix, log_lik_grad_A + A_log_prior_grad_) / K
    update_B = torch.einsum('ij,jkl->ikl', kernel_matrix, log_lik_grad_B + B_log_prior_grad_) / K

    # then the repulsive force term 
    rep_A = gamma * ((2*kernel_matrix/(sigma))[...,None,None] * ((BtB.diagonal().permute(2,0,1) @ A) - (BtB @ A.unsqueeze(0)).transpose(0,1))).sum(0) / K
    rep_B = gamma * ((2*kernel_matrix/(sigma))[...,None,None] * ((B @ AAt.diagonal().permute(2,0,1)) - (B.unsqueeze(1) @ AAt))).sum(0) / K

    # TODO: Figure out why (as it currently stands)
    #   auto_grad_rep_X = 2* rep_X / (768**2)
    
    # breakpoint()
    # update_A += rep_A
    # update_B += rep_B

    update_A += autograd_rep_A
    update_B += autograd_rep_B

    # print(f"ΔA ~ {update_A.abs().mean():.3g}, ΔB ~ {update_B.abs().mean():.3g}, sigma ~ {sigma:.3g}")
    # print(f" A ~ {A.abs().mean():.3g},  B ~ {B.abs().mean():.3g}")

    if e % 10 == 0 and e>1:
        breakpoint()

    return update_A, update_B, sigma
    
def svgd_step_low_mem(A : Tensor, B : Tensor, sigma, gamma):
    '''
    A: torch.Tensor of shape (K, r, d_in)
    B: torch.Tensor of shape (K, d_out, r)
    sigma: float
    gamma: float

    Returns:
    update_A: torch.Tensor of shape (K, r, d_in)
    update_B: torch.Tensor of shape (K, d_out, r)
    '''

    K = A.shape[0]
    assert K == B.shape[0]

    # get log likelihood (loss) gradients
    log_lik_grad_A = A.grad.clone()
    log_lik_grad_B = B.grad.clone()

    # reset grads
    A.grad.zero_()
    B.grad.zero_()

    # (improper) uniform prior
    A_log_prior_grad = torch.zeros((K,))
    B_log_prior_grad = torch.zeros((K,))

    # construct weight-update tensor
    update_A = torch.zeros_like(A)
    update_B = torch.zeros_like(B)

    # kernel_matrix = torch.zeros((K,K))
 
    # get evaluations and gradients of kernel
    for i in range(K):
        for j in range(K):
            kernel_val = rbf_kernel(A, B, i, j, sigma)
            # kernel_val.backward()

            # kernel_matrix[i,j] = kernel_val
            # print(f"{e} kernel_val: {kernel_val}, MSD: {MSD(adapter, i, j, -1)}")

            # if e > 1: breakpoint()
            breakpoint()
        
            update_A[i].add_(kernel_val * (log_lik_grad_A[j] + A_log_prior_grad[j]) - (gamma/K)*A.grad[j])
            update_B[i].add_(kernel_val * (log_lik_grad_B[j] + B_log_prior_grad[j]) - (gamma/K)*B.grad[j])

            # reset grads
            A.grad.zero_()
            B.grad.zero_()

    return update_A, update_B

def MSD(A, B, i,j):
    d_in = A.shape[0]
    d_out = B.shape[1]
    
    def trace_computation(i_, j_):
        AAt = torch.matmul(A[i_]  , A[j_].T)
        BtB = torch.matmul(B[i_].T, B[j_])

        return torch.dot(AAt.flatten(), BtB.flatten())
    
    return (trace_computation(i, i) + trace_computation(j, j) - 2*trace_computation(i, j)) / (d_in * d_out)


def rbf_kernel(A, B, i, j, sigma):
    return torch.exp(-MSD(A, B, i, j) / (2*sigma**2))