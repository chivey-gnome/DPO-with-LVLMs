import torch
import torch.nn.functional as F
from trl import DPOTrainer


class CalDPOTrainer(DPOTrainer):
    """
    Calibrated DPO Trainer implementation.
    
    This minimal implementation only overrides the dpo_loss() method.
    All other functionality (forward passes, reference model handling, 
    vision model support, etc.) is inherited from DPOTrainer.
    """
    
    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        loss_type: str = "sigmoid",
        model_output: dict[str, torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute Cal-DPO loss.
        
        Args:
            chosen_logps: Log probabilities of chosen responses from policy model
            rejected_logps: Log probabilities of rejected responses from policy model
            ref_chosen_logps: Log probabilities of chosen responses from reference model
            ref_rejected_logps: Log probabilities of rejected responses from reference model
            loss_type: Type of loss (ignored, Cal-DPO always uses calibrated sigmoid)
            model_output: Model output dict (for compatibility, not used)
        
        Returns:
            losses: Cal-DPO loss
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses
        """
        # Compute implicit rewards
        chosen_rewards = chosen_logps - ref_chosen_logps
        rejected_rewards = rejected_logps - ref_rejected_logps
        
        # Standard DPO loss (without beta multiplication )
        dpo_loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
        
        # Cal-DPO calibration losses using F.mse_loss 
        calibration_target = 0.5 / self.beta
        
        # F.mse_loss(chosen_reward, 0.5*1/beta) + F.mse_loss(reject_reward, -0.5*1/beta)
        cal_loss = (
            F.mse_loss(chosen_rewards, torch.full_like(chosen_rewards, calibration_target), reduction='none') +
            F.mse_loss(rejected_rewards, torch.full_like(rejected_rewards, -calibration_target), reduction='none')
        )
        
        # Total Cal-DPO loss
        losses = dpo_loss + cal_loss
        
        return losses, chosen_rewards, rejected_rewards