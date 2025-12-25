from fastapi import FastAPI
from lid_cavity_slant_predictor import LidCavity
from fastapi.middleware.cors import CORSMiddleware

def get_velocity(layers,model_path,dim=2):
    tsonn = LidCavity(path=model_path,layers=layers)
    tsonn.restore()
    res = tsonn.prediction() if dim==2 else tsonn.prediction_3D()
    return res

res600 = get_velocity(layers=[2,16,32,32,64,32,32,16,3],model_path="./model/model.pth")
res2000 = get_velocity(layers=[3,32,64,64,128,64,64,64,32,4],model_path="./model/slant3d-re2000.pth",dim=3)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/vel")
async def get_velocity():
    return {"message": "Hello World","Data": res600}

@app.get("/vel3d")
async def get_velocity_3d():
    return {"message": "Hello World","Data": res2000}
