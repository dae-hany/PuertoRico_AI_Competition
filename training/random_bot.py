import torch
import torch.nn as nn


class RandomBot(nn.Module):
    """
    Random action selection bot for baseline comparison.
    
    Selects uniformly at random from valid actions.
    Used as a baseline opponent in evaluation scenarios.
    """
    
    def __init__(self, action_dim: int = 200):
        super().__init__()
        self.action_dim = action_dim
        
    def get_action_and_value(self, obs_t, mask_t, phase_ids=None, action=None):
        """
        Select a random valid action.
        
        Args:
            obs_t: Observation tensor (not used)
            mask_t: Action mask tensor (1 = valid, 0 = invalid)
            phase_ids: Phase IDs (not used)
            action: Pre-specified action (not used)
            
        Returns:
            chosen_act: Randomly selected valid action
            logp: Log probability (dummy, always 0)
            entropy: Entropy (dummy, always 0)
            value: Value estimate (dummy, always 0)
        """
        # Generate random priorities
        priority = torch.rand((1, self.action_dim), dtype=torch.float32)
        
        # Mask invalid actions
        priority[mask_t == 0] = -1e9
        
        # Select argmax (effectively random among valid due to uniform random)
        chosen_act = torch.argmax(priority, dim=1)
        
        # Return dummy logp, entropy, value
        return (
            chosen_act,
            torch.zeros_like(chosen_act, dtype=torch.float),
            torch.zeros_like(chosen_act, dtype=torch.float),
            torch.zeros_like(chosen_act, dtype=torch.float),
        )
