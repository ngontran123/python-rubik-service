import twophase.solver as sv
import twophase.performance as pf
from typing import Union, List


async def rubikSolver(facelets: Union[str, None] = None) -> str:
    print('facelets is:' + facelets)
    print(len(facelets))
    res = sv.solve(facelets, 20, 3)
    print(f'solution is:{res}')
    return res


async def rubikSolveTo(original_cube: str, facelets: str) -> str:
    res = sv.solveto(original_cube, facelets, 50, 2)
    return res
