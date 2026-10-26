@echo off
echo Installing Torch 2.3.0...
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

echo Installing dependencies...
pip install -r requirements.txt

echo Uninstalling basicsr to avoid conflicts...
pip uninstall -y basicsr
pip install git+https://github.com/XPixelGroup/BasicSR

echo Installing pytorch3d...
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

echo Installing sam2...
pip install git+https://github.com/hitsz-zuoqi/sam2/

echo Installing diff-gaussian-rasterization...
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/

echo Installing simple-knn...
pip install git+https://github.com/camenduru/simple-knn/

echo Installation completed!
pause
