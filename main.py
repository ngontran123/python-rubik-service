import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from typing import Union
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime

from service import rubikSolver, rubikSolveTo

app = FastAPI(title="Rubik Solver", description="Solve Rubik Cube Algorithm.")




class RubikItem(BaseModel):
    name: str
    facelets: str
    original_cube: Union[str, None]
    des_cube: Union[str, None]


@app.post("/solve_rubik/", status_code=200)
async def solve_rubik(rb_item: RubikItem):
 try:
    rubik_facelets = rb_item.facelets
    rubik_name = rb_item.name
    if rubik_name == "":
        raise HTTPException(status_code=404, detail="Rubik name cannot be empty")
    if rubik_name == "Rubik's 3x3":
        start_time = datetime.now()
        res = await rubikSolver(rubik_facelets)
        end_time = datetime.now()
        handle_time = (end_time - start_time).total_seconds()
        headers = {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        response = {"status": "true", "data": res}
        return JSONResponse(content=response, headers=headers)
 except Exception as e:
     raise HTTPException(status_code=404, detail=str(e))


@app.post("/solve_to_rubik/", status_code=200)
async def solve_to_rubik(rb_item: RubikItem):
 try:
    rubik_name = rb_item.name
    original_cube = rb_item.original_cube
    des_cube = rb_item.des_cube
    if rubik_name == "":
        raise HTTPException(status_code=404, detail="Rubik name cannot be empty")
    elif original_cube is None or des_cube is None:
        raise HTTPException(status_code=404, detail="Original cube or des cube cannot be empty")
    if rubik_name == "Rubik's 3x3":
        start_time = datetime.now()
        res = await rubikSolveTo(original_cube, des_cube)
        end_time = datetime.now()
        handle_time = (end_time - start_time).total_seconds()
        headers = {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        response = {"status": "true", "data": res}
        return JSONResponse(content=response, headers=headers)
 except Exception as e:
      raise HTTPException(status_code=404, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8001, reload=True)
