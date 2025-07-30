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
import sys
from resources.odelia.models.mst import MST
import resources.odelia.models.mst as mst_module
import resources.odelia.models.base_model as base_model_module
# 표준 라이브러리
import tarfile
import gzip
import io
import pickle
import types

# 경로 설정
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
TEMP_PATH = Path("/tmp")  # 임시 파일 저장 경로

def load_model():
    """모델과 가중치 로드 (모든 파일 형식 자동 감지)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 가중치 파일 경로 찾기 (확장자 무관)
    try:
        weights_path = next(MODEL_PATH.glob("*"))
    except StopIteration:
        raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {MODEL_PATH}")

    # 2. sys.modules 패치 (체크포인트 unpickle 중 모듈 경로 문제 대응)
    #   1) 최상위 "odelia" 패키지, 2) "odelia.models" 서브패키지를 가짜 모듈로 등록하고
    #   3) 실제 구현이 있는 mst/base_model 모듈은 기존 resources 경로로 매핑합니다.

    # (1) 최상위 패키지
    if 'odelia' not in sys.modules:
        odelia_pkg = types.ModuleType('odelia')
        sys.modules['odelia'] = odelia_pkg
    else:
        odelia_pkg = sys.modules['odelia']

    # (2) 서브패키지 "odelia.models"
    if 'odelia.models' not in sys.modules:
        odelia_models_pkg = types.ModuleType('odelia.models')
        sys.modules['odelia.models'] = odelia_models_pkg
    else:
        odelia_models_pkg = sys.modules['odelia.models']

    # (3) 실제 모듈 매핑
    sys.modules['odelia.models.mst'] = mst_module
    sys.modules['odelia.models.base_model'] = base_model_module

    # (4) 서브패키지 속성으로도 연결(언피클 시 getattr 사용 가능하도록)
    odelia_models_pkg.mst = mst_module
    odelia_models_pkg.base_model = base_model_module
    odelia_pkg.models = odelia_models_pkg
    
    checkpoint = None
    
    # 3. 파일 타입에 따라 체크포인트 로드 (순차적 시도)
    try:
        # 시도 1: TAR 아카이브 (.tar, .tar.gz 등)
        print("Info: Attempting to load as a TAR archive...")
        with tarfile.open(weights_path, 'r:*') as tar:
            member_name = tar.getnames()[0]
            print(f"Info: Found member '{member_name}', extracting to memory.")
            extracted_file = tar.extractfile(member_name)
            if extracted_file:
                # torch.load는 seek 가능한 파일 객체가 필요하므로 메모리 버퍼 사용
                with io.BytesIO(extracted_file.read()) as f:
                    checkpoint = torch.load(f, map_location=device, weights_only=False)
        print("Info: Successfully loaded from TAR archive.")
    except tarfile.TarError:
        print("Info: Not a TAR archive. Trying other formats.")
        # TAR가 아니면 다른 형식 시도
        try:
            # 시도 2: Gzip 압축 파일 (순수 gzip)
            print("Info: Attempting to load as a Gzip file...")
            with gzip.open(weights_path, 'rb') as f:
                checkpoint = torch.load(f, map_location=device, weights_only=False)
            print("Info: Successfully loaded from Gzip file.")
        except (gzip.BadGzipFile, EOFError, pickle.UnpicklingError):
            print("Info: Not a Gzip file. Trying as a plain file.")
            # 시도 3: 일반 파일
            try:
                print("Info: Attempting to load as a plain file...")
                with open(weights_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location=device, weights_only=False)
                print("Info: Successfully loaded as a plain file.")
            except Exception as e:
                print(f"Error: All attempts to load the model failed.")
                raise e

    if checkpoint is None:
        raise IOError(f"Could not load checkpoint file: {weights_path}")

    # 4. 모델 재구성 및 가중치 로드
    hyper_parameters = checkpoint['hyper_parameters']
    if 'loss_kwargs' not in hyper_parameters:
        hyper_parameters['loss_kwargs'] = {'class_labels_num': 2}
        
    model = MST(**hyper_parameters)
    
    # 패치 정리
    if 'odelia.models.mst' in sys.modules:
        del sys.modules['odelia.models.mst']
    if 'odelia.models.base_model' in sys.modules:
        del sys.modules['odelia.models.base_model']
    
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()
    return model, device

def crop_breast_height(image, margin_top=10):
    """유방 높이에 맞게 이미지 크롭"""
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data > threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    if torch.any(fg_rows):
        top = min(max(512-int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    else:
        top = 0  # 안전한 기본값
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

def compute_subtraction(pre_img_sitk, post_img_sitk):
    """Subtraction 이미지 계산 (baseline 방식: Post - Pre)"""
    # numpy 배열로 변환
    pre = sitk.GetArrayFromImage(pre_img_sitk)
    post = sitk.GetArrayFromImage(post_img_sitk)
    
    # subtraction 계산 (int16 유지)
    sub = post - pre
    sub = sub.astype(np.int16)
    
    # SITK 이미지로 변환
    sub_nii = sitk.GetImageFromArray(sub)
    sub_nii.CopyInformation(pre_img_sitk)  # 원본 이미지의 메타데이터 복사
    
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
# from resources.odelia.models.mst import MST # This line is now redundant as MST is imported directly

def run():
    """메인 추론 함수"""
    # 모델 로드
    model, device = load_model()
    
    # 임시 디렉토리 생성
    TEMP_PATH.mkdir(exist_ok=True)
    
    # 입력 데이터 로드
    # inputs = load_json_file(INPUT_PATH / "inputs.json")
    
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
            
            try:
                t2_mha_file = next((INPUT_PATH / "images/t2-breast-mri").glob("*.mha"))
            except StopIteration:
                raise FileNotFoundError("t2-breast-mri not found")

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

            # Pre-contrast 이미지를 baseline으로 고정
            pre_images_sitk = pre_images[side].as_sitk()
            
            # Process Post-contrast images and create baseline subtractions (Post_i - Pre)
            for timepoint in sorted(all_dce_paths.keys()):
                if timepoint == 0:
                    continue # Skip pre-contrast as it's already processed

                post_contrast_path = all_dce_paths[timepoint]
                
                # 현재 post-contrast 이미지를 불러와서 baseline(pre-contrast)으로 subtraction
                curr_images, _ = preprocess_to_unilateral(post_contrast_path, ref_img=ref_img, side=side)
                curr_images_sitk = curr_images[side].as_sitk()
                
                sub_sitk = compute_subtraction(pre_images_sitk, curr_images_sitk)
                
                sub_tensor = preprocess_image(sitk.GetArrayFromImage(sub_sitk))
                final_input_tensors.append(sub_tensor)

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
                probs = torch.clamp(probs, min=0.0, max=1.0)  # 안전장치: 0-1 범위로 제한
                
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