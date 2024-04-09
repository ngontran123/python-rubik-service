from fastapi import FastAPI
from typing import Union
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from service import rubikSolver

app = FastAPI()


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    facelets = 'FURBUFRURLUBDRRFBFFRRUFLFLFDDDLDDLBDRDBDLUURLLUULBFBBB'
    val = rubikSolver(facelets)
