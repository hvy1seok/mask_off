"""
ODELIA Challenge MST Model Inference
"""

from pathlib import Path
import json
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import torchio as tio
import re

# 경로 설정
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
TEMP_PATH = Path("/tmp")  # 임시 파일 저장 경로

def crop_breast_height(image, margin_top=10):
    """유방 높이에 맞게 이미지 크롭"""
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data > threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512-int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256-top
    return tio.Crop((0,0, bottom, top, 0, 0))

def preprocess_to_unilateral(image_path, ref_img=None, side='both'):
    """이미지를 unilateral 형식으로 변환"""
    # 이미지 로드 및 전처리
    img = tio.ScalarImage(image_path)
    img = tio.ToCanonical()(img)
    
    if ref_img is None:
        # Spacing 조정
        target_spacing = (0.7, 0.7, 3)
        img = tio.Resample(target_spacing)(img)
        ref_img = img  # 첫 번째 이미지를 참조 이미지로 사용
    else:
        # 참조 이미지에 맞춰 리샘플링
        img = tio.Resample(ref_img)(img)
    
    # 크기 조정
    target_shape = (224, 224, 32)
    padding_constant = img.data.min().item()
    transform = tio.CropOrPad(target_shape, padding_mode=padding_constant)
    img = transform(img)
    
    # 유방 높이에 맞게 크롭
    img = crop_breast_height(img)
    
    # 좌우 분리
    results = {}
    if side in ['left', 'both']:
        left_crop = tio.Crop((0, 256, 0, 0, 0, 0))
        results['left'] = left_crop(img)
    if side in ['right', 'both']:
        right_crop = tio.Crop((256, 0, 0, 0, 0, 0))
        results['right'] = right_crop(img)
    
    return results, ref_img

def compute_subtraction(prev_img_sitk, curr_img_sitk):
    """Subtraction 이미지 계산 (sequential 방식: 현재 - 이전)"""
    # numpy 배열로 변환
    prev = sitk.GetArrayFromImage(prev_img_sitk)
    curr = sitk.GetArrayFromImage(curr_img_sitk)
    
    # subtraction 계산 (int16 유지)
    sub = curr - prev
    sub = sub.astype(np.int16)
    
    # SITK 이미지로 변환
    sub_nii = sitk.GetImageFromArray(sub)
    sub_nii.CopyInformation(prev_img_sitk)  # 원본 이미지의 메타데이터 복사
    
    return sub_nii

def preprocess_image(image_array):
    """이미지 전처리"""
    # numpy array -> torch tensor
    image_tensor = torch.from_numpy(image_array).float()
    
    # 정규화
    image_tensor = (image_tensor - image_tensor.mean()) / (image_tensor.std() + 1e-6)
    
    # 차원 추가 [D, H, W] -> [1, D, H, W]
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

# 우리의 MST 모델 코드 import
from resources.odelia.models.mst import MST

def load_model():
    """모델과 가중치 로드"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MST(in_ch=1, out_ch=2, loss_kwargs={'class_labels_num': 2}).to(device)
    
    # 가중치 로드
    weights_path = MODEL_PATH / "epoch=23-step=624.ckpt"
    if not weights_path.exists():
        raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {weights_path}")
        
    checkpoint = torch.load(weights_path, map_location=device)
    
    # state_dict 키 정리 (Lightning 형식 대응)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_state_dict = {k.replace('model.', '').replace('mst.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(model_state_dict)
    
    model.eval()
    return model, device

def run():
    """메인 추론 함수"""
    # 모델 로드
    model, device = load_model()
    
    # 임시 디렉토리 생성
    TEMP_PATH.mkdir(exist_ok=True)
    
    # 입력 데이터 로드
    inputs = load_json_file(INPUT_PATH / "inputs.json")
    
    # 좌우 유방별 예측 결과 저장
    bilateral_results = {"left": {}, "right": {}}
    
    for side in ['left', 'right']:
        try:
            # --- 1. Find and sort all image paths ---
            all_dce_paths = {}
            dce_dirs = list((INPUT_PATH / "images").glob("dce-breast-mri-*"))
            for path in dce_dirs:
                match = re.search(r'dce-breast-mri-(\d+)', path.name)
                if match:
                    timepoint = int(match.group(1))
                    try:
                        mha_file = next(path.glob("*.mha"))
                        all_dce_paths[timepoint] = mha_file
                    except StopIteration:
                        print(f"Warning: No .mha file found in {path}")
            
            t2_mha_file = next((INPUT_PATH / "images/t2-breast-mri").glob("*.mha"))

            # --- 2. Process images and build input tensor ---
            final_input_tensors = []
            ref_img = None
            
            # Process T2 image
            t2_images, ref_img = preprocess_to_unilateral(t2_mha_file, ref_img=None, side=side)
            t2_tensor = preprocess_image(sitk.GetArrayFromImage(t2_images[side].as_sitk()))
            final_input_tensors.append(t2_tensor)

            # Process Pre-contrast (timepoint 0)
            pre_contrast_path = all_dce_paths.get(0)
            if pre_contrast_path is None:
                raise FileNotFoundError("DCE timepoint 0 (Pre-contrast) not found.")
            
            pre_images, _ = preprocess_to_unilateral(pre_contrast_path, ref_img=ref_img, side=side)
            pre_tensor = preprocess_image(sitk.GetArrayFromImage(pre_images[side].as_sitk()))
            final_input_tensors.append(pre_tensor)

            # Initialize previous image for sequential subtraction
            prev_images_sitk = pre_images[side].as_sitk()
            
            # Process Post-contrast images and create sequential subtractions
            for timepoint in sorted(all_dce_paths.keys()):
                if timepoint == 0:
                    continue # Skip pre-contrast as it's already processed

                post_contrast_path = all_dce_paths[timepoint]
                
                curr_images, _ = preprocess_to_unilateral(post_contrast_path, ref_img=ref_img, side=side)
                curr_images_sitk = curr_images[side].as_sitk()
                
                sub_sitk = compute_subtraction(prev_images_sitk, curr_images_sitk)
                
                sub_tensor = preprocess_image(sitk.GetArrayFromImage(sub_sitk))
                final_input_tensors.append(sub_tensor)
                
                prev_images_sitk = curr_images_sitk

            # --- 3. Final batch creation and inference ---
            dce_tensor = torch.stack(final_input_tensors, dim=0)
            input_tensor = dce_tensor.unsqueeze(0)
            
            # 추론
            with torch.no_grad():
                input_tensor = input_tensor.to(device)
                logits = model(input_tensor)  # [1, 2]
                
                # CORN loss 출력을 확률로 변환
                cumulative_probs = torch.sigmoid(logits)
                cumulative_probs = torch.cummax(cumulative_probs.flip(-1), dim=-1)[0].flip(-1)
                cumulative_probs = torch.cat([
                    torch.ones_like(cumulative_probs[:, :1]), 
                    cumulative_probs, 
                    torch.zeros_like(cumulative_probs[:, :1])
                ], dim=1)
                probs = cumulative_probs[:, :-1] - cumulative_probs[:, 1:]
                probs = F.softmax(probs, dim=1)  # 확률 정규화
                
                # 각 클래스의 확률
                bilateral_results[side] = {
                    "normal": float(probs[0, 0].item()),
                    "benign": float(probs[0, 1].item()),
                    "malignant": float(probs[0, 2].item())
                }
                
        except Exception as e:
            print(f"Warning: Error processing {side} breast: {str(e)}")
            # 에러 발생 시 기본값 설정
            bilateral_results[side] = {
                "normal": 0.999,
                "benign": 0.001,
                "malignant": 0.000
            }
    
    # 결과 저장
    write_json_file(
        location=OUTPUT_PATH / "bilateral-breast-classification-likelihoods.json",
        content=bilateral_results
    )
    
    return 0

def load_json_file(location):
    """JSON 파일 로드"""
    with open(location, "r") as f:
        return json.loads(f.read())

def write_json_file(location, content):
    """JSON 파일 저장"""
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

if __name__ == "__main__":
    raise SystemExit(run()) 