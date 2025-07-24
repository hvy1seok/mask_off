
import torch 
import torch.nn as nn 
import torchvision.models as models
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import timm

from .base_model import BasicClassifier, BasicRegression

def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model) 

class TransformerFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8 if dim % 8 == 0 else 4,
                dim_feedforward=dim * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True
            ),
            num_layers=1
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([x, cls_tokens], dim=1)
        x = self.transformer(x)
        return x[:, -1]  # CLS 토큰 출력 사용

class LinearFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, x):
        return self.linear(x.mean(dim=1))  # Global average pooling + linear

class AverageFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.mean(dim=1)  # Simple global average pooling

class _MST(nn.Module):
    def __init__(
        self, 
        out_ch=2,  # CORN loss: num_classes-1 outputs (3-1 = 2)
        backbone_type="dinov2",
        model_size = "s", # 34, 50, ... or 's', 'b', 'l'
        slice_fusion_type = "transformer", # transformer, linear, average, none 
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type

        # DINOv2 모델 크기 약어를 전체 이름으로 매핑
        dinov2_model_size_map = {
            's': 'small',
            'b': 'base',
            'l': 'large',
            'g': 'giant'
        }

        if backbone_type == "resnet":
            Model = _get_resnet_torch(model_size)
            self.backbone = Model(weights="DEFAULT")
            emb_ch = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_type == "dinov2":
            full_model_size = dinov2_model_size_map.get(model_size, model_size)
            model_name = f'vit_{full_model_size}_patch14_dinov2'
            # torch.hub.load 대신 timm.create_model 사용 (오프라인 호환)
            self.backbone = timm.create_model(model_name, pretrained=False)
            self.backbone.mask_token = None
            emb_ch = self.backbone.embed_dim
        
        if slice_fusion_type == "transformer":
            self.slice_fusion = TransformerFusion(emb_ch)
        elif slice_fusion_type == "linear":
            self.slice_fusion = LinearFusion(emb_ch)
        elif slice_fusion_type == "average":
            self.slice_fusion = AverageFusion()
        else:
            self.slice_fusion = None

        # 단일 linear layer로 변경
        self.linear = nn.Linear(emb_ch, out_ch)

    def forward(self, x):
        B, P, C, D, H, W = x.shape
        
        # [B, P, C, D, H, W] -> [B*P*D, C, H, W]
        x = rearrange(x, 'b p c d h w -> (b p d) c h w')
        
        # [B*P*D, C, H, W] -> [B*P*D, 3, H, W]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale to RGB

        # [B*P*D, 3, H, W] -> [B*P*D, E]
        x = self.backbone(x)
        
        # [B*P*D, E] -> [B, P*D, E]
        x = rearrange(x, '(b p d) e -> b (p d) e', b=B, p=P)
        
        # [B, P*D, E] -> [B, E]
        if self.slice_fusion is not None:
            x = self.slice_fusion(x)
        
        # [B, E] -> [B, 1]
        x = self.linear(x)
        
        return x
    

class MST(BasicRegression):
    # MST - https://arxiv.org/abs/2411.15802 
    def __init__(
            self,
            in_ch=1, 
            out_ch=2,  # CORN loss: num_classes-1 outputs (3-1 = 2)
            spatial_dims=3,
            backbone_type="dinov2",
            model_size = "s", # 34, 50, ... or 's', 'b', 'l'
            slice_fusion_type = "transformer", # transformer, linear, average, none  
            optimizer_kwargs={'lr':1e-6}, 
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)  
        self.mst = _MST(out_ch=out_ch, backbone_type=backbone_type, model_size=model_size, slice_fusion_type=slice_fusion_type)
    
    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch['source']
        else:
            x = batch
        return self.mst(x)

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        """MST를 위한 커스텀 _step 메서드"""
        target = batch['target']
        batch_size = target.shape[0]
        self.batch_size = batch_size 

        # MST 모델에 전체 배치 전달
        pred = self.model(batch)

        # ------------------------- Compute Loss ---------------------------
        logging_dict = {}
        
        # 차원 맞추기 - one-hot target을 class index로 변환
        if len(target.shape) > 1 and target.shape[1] > 1:  # [B, 3] 형태인 경우
            target = torch.argmax(target, dim=1)  # one-hot -> class index
        elif len(target.shape) > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)  # [B, 1] -> [B]
            
        logging_dict['loss'] = self.loss_func(pred, target)

        # --------------------- Compute Metrics  -------------------------------
        pred_labels = self.loss_func.logits2labels(pred)
        
        with torch.no_grad():
            # Aggregate here to compute for entire set later 
            self.mae[state+"_"].update(pred_labels, target)
            
            # ----------------- Log Scalars ----------------------
            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val, batch_size=batch_size, on_step=True, on_epoch=True, 
                         sync_dist=False) 

        return logging_dict['loss']


class MSTRegression(BasicRegression):
    def __init__(
            self,
            in_ch=1, 
            out_ch=1, 
            spatial_dims=3,
            backbone_type="dinov2",
            model_size = "s", # 34, 50, ... or 's', 'b', 'l'
            slice_fusion_type = "transformer", # transformer, linear, average, none  
            optimizer_kwargs={'lr':1e-6}, 
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)  
        self.mst = _MST(out_ch=out_ch, backbone_type=backbone_type, model_size=model_size, slice_fusion_type=slice_fusion_type)
    
    def forward(self, x):
        return self.mst(x)