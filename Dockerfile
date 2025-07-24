FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/app

# 비루트 user 생성
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# 유저 변경
USER user

# 작업 디렉토리 설정
WORKDIR /opt/app

# requirements.txt 및 기타 리소스 복사
COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# 패키지 설치 (user의 pip로)
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# 추론 코드 복사
COPY --chown=user:user inference.py /opt/app/

# 실행
ENTRYPOINT ["python", "inference.py"] 