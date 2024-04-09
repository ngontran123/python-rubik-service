import twophase.solver as sv
import twophase.performance as pf
from typing import Union, List


def rubikSolver(facelets: Union[str, None] = None) -> str:
    print('facelets is:' + facelets)
    print(len(facelets))
    cubestring = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'

    res = sv.solve(facelets, 20, 3)
    res_temp = sv.solveto(cubestring, facelets, 50, 2)
    print(f'solve to is:{res_temp}')
    print(f'solution is:{res}')
    return res
