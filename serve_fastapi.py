# serve_fastapi.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
from pathlib import Path
from train_siamese_with_metrics import SiameseWithMetrics, inference_single
import tempfile
import shutil

app = FastAPI()
MODEL_PATH = "./ckpts/best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
def load_model():
    global model
    model = SiameseWithMetrics(emb_dim=256, pretrained=False).to(DEVICE)
    ck = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ck['model_state'])
    model.eval()
load_model()

@app.post("/predict_files")
async def predict_files(ref: UploadFile = File(...), test: UploadFile = File(...)):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        ref_path = tmpdir / ref.filename
        test_path = tmpdir / test.filename
        with open(ref_path, 'wb') as f:
            f.write(await ref.read())
        with open(test_path, 'wb') as f:
            f.write(await test.read())
        out = inference_single(model, str(ref_path), str(test_path), DEVICE)
        return JSONResponse(content=out)
    finally:
        shutil.rmtree(tmpdir)

@app.post("/predict_paths")
async def predict_paths(payload: dict):
    ref = payload.get('ref_path'); test = payload.get('test_path')
    if not ref or not test:
        raise HTTPException(status_code=400, detail="ref_path and test_path required")
    if not Path(ref).exists() or not Path(test).exists():
        raise HTTPException(status_code=400, detail="file not found")
    out = inference_single(model, ref, test, DEVICE)
    return JSONResponse(content=out)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
