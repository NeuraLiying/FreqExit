

class LossTracker:

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.ce_avg = None
        self.ee_avg = None
        self.distill_avg = None
        self.hf_avg = None
    
    def update(self, ce_loss, ee_loss, distill_loss, hf_loss):

        def _update_avg(curr_avg, new_val):
            if curr_avg is None:
                return new_val.detach()
            return self.momentum * curr_avg + (1 - self.momentum) * new_val.detach()
        
        self.ce_avg = _update_avg(self.ce_avg, ce_loss)
        self.ee_avg = _update_avg(self.ee_avg, ee_loss)
        self.distill_avg = _update_avg(self.distill_avg, distill_loss)
        self.hf_avg = _update_avg(self.hf_avg, hf_loss)
    
    def get_hf_scale(self, alpha=1.0, eps=1e-6):
      
        if any(avg is None for avg in [self.ce_avg, self.ee_avg, self.distill_avg, self.hf_avg]):
            return 1.0
        
   
        other_losses_avg = (self.ce_avg + self.ee_avg + self.distill_avg) / 3
      
        scale = alpha * other_losses_avg / (self.hf_avg + eps)
        return scale