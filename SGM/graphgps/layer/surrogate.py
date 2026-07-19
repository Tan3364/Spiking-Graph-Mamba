import torch


#class piecewise_quadratic(torch.autograd.Function):
#    @staticmethod
#    def forward(ctx, x, threshold):
#        if x.requires_grad:
#            ctx.save_for_backward(x)
#        return (x-threshold >= 0).to(x)
#
#    @staticmethod
#    def backward(ctx, grad_output):
#        x = ctx.saved_tensors[0]
#        x_abs = x.abs()
#        mask = x_abs > 1
#        grad_x = (grad_output * (-x_abs + 1.0)).masked_fill_(mask, 0)
#        return grad_x, None
#
#
#def piecewise_quadratic_surrogate():
#    return piecewise_quadratic.apply

#    
#class ternary_piecewise_quadratic(torch.autograd.Function):
#    @staticmethod
#    def forward(ctx, x, threshold):
#        if x.requires_grad:
#            ctx.save_for_backward(x)
#        out_s = torch.sign(x)
#        out_s[torch.abs(x) < threshold] = 0.0
#        # ---- 平滑输出：用于梯度近似 ----
#        out_bp = torch.clamp(x, -1, 1)
#        # ---- “前向修正” = 硬输出 + 平滑梯度代理 ----
#        out = (out_s - out_bp).detach() + out_bp
#        
#        return out
#
#    @staticmethod
##    def backward(ctx, grad_output):
##        x = ctx.saved_tensors[0]
##        out_bp = torch.clamp(x, -1, 1)
##        grad_x = grad_output * (1 - torch.abs(out_bp))
##        return grad_x, None
#
##    @staticmethod
#    def backward(ctx, grad_output):
#        x = ctx.saved_tensors[0]
#        grad_x = (torch.abs(x) <= 1).float()  # 仅在[-1,1]区间内传递梯度
#        # 构建梯度近似：使用 straight-through estimator（STE）
#        #grad_x = (torch.sign(x).float() - out_bp).detach() + out_bp
#        grad_x = grad_output * grad_x  # 链式法则
#        return grad_x, None
#
#def ternary_piecewise_quadratic_surrogate():
#    return ternary_piecewise_quadratic.apply

class quant4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,threshold):
        ctx.save_for_backward(input)
        return torch.round(torch.clamp(input, min=0, max=4))
    

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
      
        grad_input[input < 0] = 0
        grad_input[input > 4.0] = 0
        return grad_input, None

        
def quant4_surrogate():
    return quant4.apply
    
class ternary_piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        if x.requires_grad:
            #ctx.save_for_backward(x)
            threshold = torch.tensor(threshold, device=x.device, dtype=x.dtype)
            ctx.save_for_backward(x, threshold)
        out_s = torch.sign(x)
        out_s[torch.abs(x) < threshold] = 0.0
        # ---- 平滑输出：用于梯度近似 ----
        #out_bp = torch.clamp(x, -threshold, threshold)
        # ---- “前向修正” = 硬输出 + 平滑梯度代理 ----
        #out = (out_s - out_bp).detach() + out_bp
        
        return out_s
        #return (x-threshold >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        delta = 0.125  # 可以改为参数

        grad_x = torch.zeros_like(x)

        # 在 +threshold 附近的梯度（倒抛物线）
        mask_pos = (x > threshold - delta) & (x < threshold + delta)
        #grad_x[mask_pos] = 1 - ((x[mask_pos] - threshold) / delta) ** 2

        # 在 -threshold 附近的梯度（对称倒抛物线）
        mask_neg = (x > -threshold - delta) & (x < -threshold + delta)
        #grad_x[mask_neg] = 1 - ((x[mask_neg] + threshold) / delta) ** 2
        
        #查看ln_vth
        grad_x[mask_pos] = 1
        grad_x[mask_neg] = 1

        # 梯度输出
        grad_x = grad_output * grad_x
        return grad_x, None

def ternary_piecewise_quadratic_surrogate():
    return ternary_piecewise_quadratic.apply