import torch
import torch.nn as nn
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

import torch.nn.functional as F

class CornLossMulti(torch.nn.Module):
    """
    Compute the CORN loss for multi-class classification.
    """
    def __init__(self, class_labels_num):
        super().__init__()
        self.class_labels_num = class_labels_num # [Classes, Labels]
    
    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*(num_labels-1)]
            targets: torch.Tensor, shape [batch_size]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        loss = 0
        for c, chunk in enumerate(chunks):
            loss += corn_loss(chunk, targets[:, c], chunk.shape[1]+1)
        return loss/len(chunks)

    def logits2labels(self, logits):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*(num_labels-1)]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        labels = []
        for c, chunk in enumerate(chunks):
            label = corn_label_from_logits(chunk)
            labels.append(label)
        return torch.stack(labels, dim=1)
    
    def logits2probabilities(self, logits):
        # Argmax can leed to different output: https://github.com/Raschka-research-group/coral-pytorch/discussions/27
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*(num_labels-1)]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        classes_probs = []
        for c, chunk in enumerate(chunks):
            cumulative_probs = torch.sigmoid(chunk)
            # cumulative_probs = torch.cumprod(probas, dim=1)
            
            # Add boundary conditions P(y >= 1) = 1 and P(y >= num_classes) = 0
            cumulative_probs = torch.cat([torch.ones_like(cumulative_probs[:, :1]), cumulative_probs, torch.zeros_like(cumulative_probs[:, :1])], dim=1)
            
            # Compute class probabilities
            # cumulative_probs = torch.cat([torch.ones_like(cumulative_probs[:, :1]), cumulative_probs], dim=1)
            probs = cumulative_probs[:, :-1] - cumulative_probs[:, 1:]
            # probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # probs = cumulative_probs
          
            classes_probs.append(probs)
        return torch.stack(classes_probs, dim=1)
    




class MulitCELoss(nn.Module):
    """
    CrossEntropyLoss per class-label group.
    """
    def __init__(self, class_labels_num):
        """
        Args:
            class_labels_num: List[int], number of labels for each class group
        """
        super().__init__()
        self.class_labels_num = class_labels_num
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, sum(class_labels_num)]
            targets: torch.Tensor, shape [batch_size, len(class_labels_num)]
        """
        chunks_logits = torch.split(logits, self.class_labels_num, dim=1)
        loss = 0
        for i, logit_chunk in enumerate(chunks_logits):
            target_chunk = targets[:, i]
            loss += self.criterion(logit_chunk, target_chunk)

        return loss / len(chunks_logits)

    def logits2labels(self, logits):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, sum(class_labels_num)]
        Returns:
            torch.Tensor, shape [batch_size, len(class_labels_num)]
        """
        chunks_logits = torch.split(logits, self.class_labels_num, dim=1)
        labels = [torch.argmax(chunk, dim=1) for chunk in chunks_logits]
        return torch.stack(labels, dim=1)

    def logits2probabilities(self, logits):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, sum(class_labels_num)]
        Returns:
            torch.Tensor, shape [batch_size, len(class_labels_num), max_labels]
        """
        chunks_logits = torch.split(logits, self.class_labels_num, dim=1)
        probs = [F.softmax(chunk, dim=1) for chunk in chunks_logits]
        return torch.stack(probs, dim=1)


class MultiBCELoss(nn.Module):
    """
        BCEWithLogitsLoss per class-label group.
    """
    def __init__(self, class_labels_num):
        """
        Args:
            class_labels_num: List[int], number of labels for each class group
        """
        super().__init__()
        self.class_labels_num = class_labels_num
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, sum(class_labels_num)]
            targets: torch.Tensor, shape [batch_size, sum(class_labels_num)]
        """
        chunks_logits = torch.split(logits, self.class_labels_num, dim=1)
        chunks_targets = torch.split(targets, self.class_labels_num, dim=1)

        loss = 0
        for logit_chunk, target_chunk in zip(chunks_logits, chunks_targets):
            loss += self.criterion(logit_chunk, target_chunk.float())

        return loss / len(chunks_logits)

    def logits2labels(self, logits, threshold=0.5):
        probs = torch.sigmoid(logits)
        return (probs > threshold).int()

    def logits2probabilities(self, logits):
        return torch.sigmoid(logits)


class WeightedCELoss(nn.Module):
    """
    Weighted Cross Entropy Loss for ordinal classification.
    Uses a weight matrix to penalize predictions based on class distances.
    """
    def __init__(self, class_labels_num, weight_matrix=None):
        super().__init__()
        if weight_matrix is None:
            # 기본 가중치 매트릭스: 클래스 간 거리에 따른 페널티
            weight_matrix = torch.tensor([
                [1.0, 2.0, 4.0],  # No Lesion -> [Benign(2), Malignant(4)]
                [2.0, 1.0, 2.0],  # Benign -> [No Lesion(2), Malignant(2)]
                [4.0, 2.0, 1.0]   # Malignant -> [No Lesion(4), Benign(2)]
            ])
        self.register_buffer('weight_matrix', weight_matrix)
        self.class_labels_num = class_labels_num
        self.base_criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes] or [batch_size]
        """
        # 2D 타겟을 1D로 변환 (필요한 경우)
        if len(targets.shape) > 1:
            targets = targets[:, 0]  # 첫 번째 클래스의 타겟만 사용
            
        base_loss = self.base_criterion(logits, targets)
        
        # 예측 클래스
        pred_classes = torch.argmax(logits, dim=1)
        
        # 각 샘플에 대한 가중치 적용
        batch_weights = self.weight_matrix[targets, pred_classes]
        weighted_loss = base_loss * batch_weights
        
        return weighted_loss.mean()

    def logits2labels(self, logits):
        return torch.argmax(logits, dim=1)
    
    def logits2probabilities(self, logits):
        return F.softmax(logits, dim=1)


class OrdinalWeightedCELoss(nn.Module):
    """
    Ordinal Weighted Cross Entropy Loss.
    Applies exponential weights based on the ordinal distance between predicted and true classes.
    """
    def __init__(self, class_labels_num, ordinal_weights=None):
        super().__init__()
        if ordinal_weights is None:
            # 기본 가중치: 거리가 멀수록 exponential하게 증가
            ordinal_weights = torch.tensor([1.0, 2.0, 4.0])
        self.register_buffer('ordinal_weights', ordinal_weights)
        self.class_labels_num = class_labels_num
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes] or [batch_size]
        """
        # 2D 타겟을 1D로 변환 (필요한 경우)
        if len(targets.shape) > 1:
            targets = targets[:, 0]  # 첫 번째 클래스의 타겟만 사용
            
        base_loss = self.criterion(logits, targets)
        
        # 예측과 실제 클래스 간의 거리에 따른 가중치 적용
        pred_classes = torch.argmax(logits, dim=1)
        ordinal_distances = torch.abs(pred_classes - targets).float()
        weights = torch.pow(self.ordinal_weights[0], ordinal_distances)
        
        weighted_loss = base_loss * weights
        return weighted_loss.mean()

    def logits2labels(self, logits):
        return torch.argmax(logits, dim=1)
    
    def logits2probabilities(self, logits):
        return F.softmax(logits, dim=1)