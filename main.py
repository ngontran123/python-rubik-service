import uvicorn
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, Body
from typing import Union, List
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import traceback
import os

import cv2
import numpy as np

import service

from service import rubikSolver, rubikSolveTo, maskImage, handleRectangle

app = FastAPI(title="Rubik Solver", description="Solve Rubik Cube Algorithm.")


class RubikItem(BaseModel):
    name: str
    facelets: str
    original_cube: Union[str, None]
    des_cube: Union[str, None]


class BaseInput(BaseModel):
    original_cube: str


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
        print(traceback.format_exc())
        raise HTTPException(status_code=404, detail=str(e))


@app.post('/color_detection_image/', status_code=200)
async def color_detection_image(img: List[UploadFile] = File(...), original_cube: str = Form(...)):
    try:
        # color_ranges = \
        # {
        #         'red': ((0, 100, 100), (1, 255, 255)),
        #         'green': ((40, 40, 40), (80, 255, 255)),
        #         'blue': ((100, 150, 0), (140, 255, 255)),
        #         'yellow': ((20, 100, 100), (30, 255, 255)),
        #         'orange': ((10, 100, 20), (25, 255, 255)),
        #         'white': ((0, 0, 200), (180, 20, 255))
        # }
        colors = ''
        print('Original_cube here is:' + original_cube)

        for index, image in enumerate(img):
            print("Image name is:" + image.filename)

            contents = await image.read()

            file_path = os.path.join(os.getcwd(), image.filename)

            print('File path here is:' + file_path)

            with open(image.filename, "wb") as f:
                f.write(contents)
            # np_img = np.frombuffer(contents, dtype=np.uint8)
            # print("np image:"+str(np_img))
            # if np_img is None:
            #     print('Np image is none')
            #
            #decode_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            # print('decode image:'+str(decode_img))
            #
            # cv2.imwrite('decode.jpg', decode_img)
            color_list = handleRectangle(file_path)

            print('color here is:' + str(color_list))

            if len(color_list) < 9:
                if os.path.exists(image.filename):
                    os.remove(image.filename)
                raise HTTPException(status_code=404, detail=f"Color detected failed at index {index}.")
            color_converted = ''
            for color in color_list:
                converted = await service.convertColor(color)
                color_converted += converted
            colors += color_converted
            if os.path.exists(image.filename):
                os.remove(image.filename)
            # return JSONResponse(content=response, headers={'Content-Type': 'application/json'})
            # with open(f'uploaded_{image.filename}', 'wb') as f:
            #     f.write(image)
        move_steps = await service.rubikSolveTo(original_cube, colors)
        response = {"status": True, "data": move_steps, "message": "Image Color has been retrieved"}
        return JSONResponse(content=response, headers={'Content-Type': 'application/json'})
    except Exception as e:
        print(traceback.format_exc())
        for image in img:
            if os.path.exists(image.filename):
                os.remove(image.filename)
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8004, reload=False)
