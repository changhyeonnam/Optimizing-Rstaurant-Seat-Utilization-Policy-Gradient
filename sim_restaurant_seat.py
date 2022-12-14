import copy

import numpy as np
import matplotlib.pyplot as plt
import time

def update_dict(dt, key, value=None):
    if key not in dt:
        dt[key] = value
    else:
        print(f'key{key} is duplicated.')

def print_matrix(current_seat_grid,color_map):

    grid = np.ndarray(shape=(current_seat_grid.shape[0], current_seat_grid.shape[1], 3), dtype=int)
    for i in range(0, grid.shape[0]):
        for j in range(0, grid.shape[1]):
            grid[i][j] = color_map[int(current_seat_grid[i][j])]

    plt.show()
    plt.imshow(grid)


def update_matrix(total_customer_dict, current_seat_grid, customer_info, unique_id, eating_job):
    complete_ppl = set()

    # check if there are people who finished eating.
    for pid in eating_job:
        cur_time = total_customer_dict[pid]['time']
        print(pid,cur_time)
        total_customer_dict[pid]['time']= cur_time -1
        if total_customer_dict[pid]['time'] <= 0:
            complete_ppl.add(pid)
    eating_job = eating_job-complete_ppl

    # remove people who finished eating.
    for ppl in complete_ppl:
        ys, xs = np.where(current_seat_grid == ppl)
        for y,x in zip(ys,xs):
            current_seat_grid[y][x]=0


    height,width = customer_info['height'],customer_info['width']
    grid_height,grid_width = current_seat_grid.shape

    check = False
    for y in range(grid_height):
        for x in range(grid_width):
            if(current_seat_grid[y][x]!=0):
                continue
            if x+width>= grid_width or y+height>=grid_height:
                continue

            if not np.any(current_seat_grid[y:y+height][x:x+width]):
                for i in range(y,y+height):
                    for j in range(x,x+width):
                        current_seat_grid[i][j]=unique_id
                check=True
                break

        if check:
            eating_job.add(unique_id)
            break

    return current_seat_grid, eating_job

def simualation():
    customer_dict = {}
    customer_dict[1]={'width':1,'height':1, 'count':1, 'time':1, 'money':1}
    customer_dict[2]={'width':1,'height':2, 'count':2, 'time':2, 'money':2}
    customer_dict[3]={'width':2,'height':1, 'count':2, 'time':2,'money':2}
    customer_dict[4]={'width':2,'height':2, 'count':3, 'time':4, 'money':3}
    customer_dict[5]={'width':2,'height':2, 'count':4, 'time':5, 'money':2}

    width = 8
    height =8
    current_seat_grid = np.zeros((width,height))
    sequence_ppl =[4,4,4,4,4,4,4,4,2,3,3,4,5,4,4,4,1,3,]
    eating_job = set()
    total_ppl_dict = {}
    color_map = {}
    update_dict(color_map, 0, np.array([255, 255, 255]))
    for idx in range(1,len(sequence_ppl)+1):
        color = np.random.uniform(0, 255, (3))
        update_dict(color_map, idx, color)

    for idx, type in enumerate(sequence_ppl,1):
        timestep = idx
        unique_id = idx
        customer_info = copy.deepcopy(customer_dict[type])

        # update total customer info dictionary
        update_dict(total_ppl_dict,unique_id,customer_info)

        current_seat_grid, eating_job = update_matrix(total_ppl_dict, current_seat_grid, customer_info, unique_id, eating_job)
        print(current_seat_grid)
        print_matrix(current_seat_grid,color_map)



if __name__ == '__main__':
    simualation()