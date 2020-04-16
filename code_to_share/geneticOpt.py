#!/usr/bin/env python3

'''Basic genetic algorithm.
Nicholas Cooper
nick.cooper13@gmail.com
'''

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class GA:
    '''A basic genetic algorithm designed to be able to handle constraints.
    All constraint functions that are passed must return scalar values that 
    will be used in a manner similar to penalties.  If there is no violation 
    the constraint function must return 0.
    A simple way to handle multi-dimensional constraints is to add all violation 
    and return the sum.
    '''
    def __init__(self, obj, bounds, tol=1e-4, maxiter=1000, args=(), 
        constraints=None, callback=None, mscale=0.1, pmutate = 0.01):
        self.obj = obj
        self.bounds = bounds
        self.cons_tup = constraints
        self.cons_equal = []
        self.cons_inequal = []
        self.callback = callback
        self.args = args
        self.numvars = len(bounds)
        self.x = []
        self.pop_size = (30*self.numvars, self.numvars)
        self.fitness = []
        self.scale = mscale
        self.pmutate = pmutate

        self.tol = tol
        self.xhist = []
        self.maxiter = maxiter

        self.fevals = 0
        self.iter = 0
        self.infeasible = False

        if constraints is not None:
            self.setConstraints()
            self.lam_eq = np.zeros(len(self.cons_equal))
            self.lam_ineq = np.zeros(len(self.cons_inequal))

        self.gen_ini_pop()

    def objWrap(self, x):
        self.fevals += 1
        args = self.args
        return self.obj(x, *args)

    def gen_ini_pop(self):
        ipop = np.zeros(self.pop_size)

        if self.numvars < 2:
            r = np.random.random()
            ipop = self.bounds[0][0] + r*(self.bounds[0][1] - self.bounds[0][0])
        else:
            for i in range(self.numvars):
                for j in range(self.pop_size[0]):
                    r = np.random.random()
                    tup = self.bounds[i]
                    ipop[j][i] = tup[0] + r*(tup[1] - tup[0])

        self.ini_pop = ipop

    def eval_fitness(self, pop):
        fvals = np.zeros(self.pop_size[0])
        fitnessvals = np.zeros(self.pop_size[0])

        for i in range(self.pop_size[0]):
            fvals[i] = self.objWrap(pop[i])

        f_low = np.min(fvals)
        DF = 1.1*np.max(fvals) - 0.1*f_low

        for i in range(self.pop_size[0]):
            fitnessvals[i] = (-fvals[i] + DF) / np.max((1, -f_low + DF))

        self.fitness = fitnessvals
        return fvals, fitnessvals

    def tournament(self, pop):
        # Pair up all of the populations
        
        best = np.zeros(self.pop_size)
        count = 0
        for iterations in range(2):
            idxs = np.arange(len(pop))
            num_idxs = len(idxs)

            for i in range(count, num_idxs//2 + count):
                # print(i, idxs)
                np.random.shuffle(idxs)
                x1 = pop[idxs[0]]
                x2 = pop[idxs[1]]
                fit1 = self.fitness[idxs[0]]
                fit2 = self.fitness[idxs[1]]

                if self.cons_tup is not None:
                    v1, v2 = self.test_feasibility(x1, x2)
                    if v1 != 0 and v1 !=0:
                        self.infeasible = True
                    else:
                        self.infeasible = False
                    # if v1 < 1e-3 and v2 < 1e-3:
                    #     if fit1 > fit2:
                    #         best[i] = x1
                    #     else:
                    #         best[i] = x2
                    if v1 < v2:
                        best[i] = x1
                    elif v1 > v2:
                        best[i] = x2
                    else:
                        if fit1 > fit2:
                            best[i] = x1
                        else:
                            best[i] = x2
                else:
                    if fit1 > fit2:
                        best[i] = x1
                    else:
                        best[i] = x2

                idxs = np.delete(idxs, [0, 1])

            count += i + 1
        # print(v1, v2)
        return best

    def crossover(self, pop):
        # print("TODO: not finished with this part yet?")
        children = np.zeros(self.pop_size)
        idxs = np.arange(len(pop))
        num_idxs = len(idxs)

        for i in range(num_idxs-1):
            np.random.shuffle(idxs)
            if self.fitness[idxs[0]] > self.fitness[idxs[1]]:
                p1 = pop[idxs[1]]
                p2 = pop[idxs[0]]
            else:
                p1 = pop[idxs[0]]
                p2 = pop[idxs[1]]

            c1 = 0.5*p1 + 0.5*p2
            c2 = 2*p2 - p1

            children[i] = c1
            children[i+1] = c2

        return children

    def mutation(self, pop, p=None, d=None):

        if d is None:
            d = self.scale
        if p is None:
            p = self.pmutate

        pop_new = np.zeros(np.shape(pop))
        for i in range(len(pop)):
            r = np.random.random()
            if r < p:
                # print(i, r)
                # print("mutated")
                for j in range(len(pop[i])):
                    r = np.random.random()
                    pop_new[i][j] = pop[i][j] + (r - 0.5)*d
                    
            else:
                pop_new[i] = pop[i]

            # Check bounds - return to feasible space with a random selection?
            for j in range(len(pop[i])):
                if pop_new[i][j] > self.bounds[j][1] or pop_new[i][j] < self.bounds[j][0]:
                    r = np.random.random()
                    tup = self.bounds[j]
                    pop_new[i][j] = tup[0] + r*(tup[1] - tup[0])
                    # pop_new[i] = pop[i]

        return pop_new

    def test_feasibility(self, x1, x2):
        v1 = 0
        v2 = 0

        for eq_con in self.cons_equal:
            con_fun = eq_con[0]
            scale = eq_con[1]
            if scale is None:
                scale = 1
            v1 += abs(con_fun(x1)/scale)
            v2 += abs(con_fun(x2)/scale)

        for ineq_con in self.cons_inequal:
            con_fun = ineq_con[0]
            scale = ineq_con[1]
            if scale is None:
                scale = 1
            v1 += abs(con_fun(x1)/scale)
            v2 += abs(con_fun(x2)/scale)

        return v1, v2


    def optimize(self, verbose=False):
        # print("TODO: write optimization method")
        ipop = self.ini_pop
        iterations = 0
        old_best = 1e16
        besty = 0
        num_good = 0

        while iterations < self.maxiter:

            old_best = besty

            fvals, fitness = self.eval_fitness(ipop)
            pool = self.tournament(ipop)
            pop = self.crossover(pool)
            new_pop = self.mutation(pop)
            # print(new_pop)
            ipop = new_pop
            iterations += 1

            bestx = np.zeros(self.numvars)
            fit = 0
            for i in range(len(fitness)):
                # print(fitness[i])
                if fitness[i] > fit:
                    fit = fitness[i]
                    bestx = new_pop[i]

            besty = self.objWrap(bestx)
            old_best

            if abs(old_best - besty) < self.tol:
                num_good += 1
            else:
                num_good = 0

            if self.callback is not None:
                self.callback(bestx)#new_pop)

            if num_good == 3:
                print("Converged")
                break

            if verbose:
                # print(iterations)
                # print(bestx)
                if self.infeasible:
                    v1, v2 = self.test_feasibility(bestx, bestx)
                    print(f"Generation {iterations}: {besty}, infeasible {v1}")
                else:
                    print(f"Generation {iterations}: {besty}")

        final_pop = new_pop
        fvals, fitness = self.eval_fitness(final_pop)
        # print(final_pop)

        bestx = np.zeros(self.numvars)
        fit = 0
        for i in range(len(fitness)):
            # print(fitness[i])
            if fitness[i] > fit:
                fit = fitness[i]
                bestx = final_pop[i]

        finalx = bestx
        finaly = self.objWrap(finalx)

        if iterations == self.maxiter:
            print("Max number of iterations exceeded.")

        return finalx, finaly
    

    def gen_ext_pen(self, func, x, cons_eq_list, cons_ineq_list, mu1, mu2, args=()):
        Sc1 = 0
        for i in range(len(cons_eq_list)):
            try:
                for con in cons_eq_list[i][0](x):
                    Sc1 += con**2
            except:
                Sc1 += cons_eq_list[i][0](x)**2#, *args)
        Sc2=0
        for i in range(len(cons_ineq_list)):
            try:
                for con in cons_ineq_list[i][0](x):
                    Sc2 += min(0, con)**2
            except:
                Sc2 += min(0, cons_ineq_list[i][0](x))**2

        f = func(x)#, *args
        F = f + mu1/2*Sc1 + mu2/2*Sc2

        return F

    def setConstraints(self):

        try:
            len(self.cons_tup)
        except:
            self.cons_tup = (self.cons_tup,)

        self.num_cons = len(self.cons_tup)

        for con_dict in self.cons_tup:
            store_info = []
            optional_keys = ('jac', 'scale')

            for key in optional_keys:
                if key not in con_dict.keys():
                    con_dict[key] = None

            store_info = (con_dict['fun'], con_dict['scale'])

            if con_dict['type'] == 'eq':
                self.cons_equal.append(store_info)

            elif con_dict['type'] == 'ineq':
                self.cons_inequal.append(store_info)

            else:
                print("Not a valid constraint type (options: 'eq', 'ineq'")
                quit()


def main():
    def f(var):
        x1 = var[0]
        x2 = var[1]
        return x1**2 - 2*x2
    def con1(var):
        x1 = var[0]
        x2 = var[1]
        return x2 - 2*x1
    def con2(var):
        x1 = var[0]
        x2 = var[1]
        return x1 + x2

    def rosen(X):
        """The Rosenbrock function"""
        x = X[0]
        y = X[1]
        return 100.0*(x-y**2.0)**2.0 + (1-y)**2.0

    pop_hist = []
    def callback(pop):
        pop_hist.append(pop)

    ga = GA(rosen, bounds = ((-3, 3), (-3, 3)), maxiter=1000, 
        callback=callback, tol=1e-8, scale=0.1, pmutate = 0.1)
    # constraints = ({'fun':con1, 'type':'ineq'}, {'fun':con2, 'type':'ineq'})
    # ga = GA(f, bounds = ((-10, 10), (-10, 10)), maxiter=1000, callback=callback, 
    #     tol=1e-8, scale=0.1, pmutate=0.1, constraints=constraints)

    # print("Initial:")
    # print(ga.ini_pop)
    # print("Fitness:")
    # print(ga.eval_fitness(ga.ini_pop))
    # winners = ga.tournament(ga.ini_pop)
    # print("Tournament winners:")
    # print(winners)
    # cross = ga.crossover(winners)
    # print("Crossover")
    # print(cross)
    # print("Mutation:")
    # print(ga.mutation(cross, p=0.1))

    sol = ga.optimize()
    print("Solution")
    print(sol)
    # print(pop_hist)
    pop_hist = np.array(pop_hist)

    # sci = minimize(rosen, [1, 1], method='BFGS')
    # print(sci)

    # 2D Plot=============================================
    rmin = -3
    rmax = 3
    npts = 100
    xx = np.linspace(rmin, rmax, npts)
    yy = np.linspace(rmin, rmax, npts)

    X, Y = np.meshgrid(xx, yy)

    # fig, (ax1, ax2) = plt.subplots(ncols = 2)
    # ax1.imshow(f1([X, Y]), extent = [rmin, rmax, rmin, rmax])
    # ax1.scatter(f1x[0], f1x[1], color='red')
    # ax1.set_title("$f1 \leq 1$")
    # ax2.imshow(f2([X, Y]), extent = [rmin, rmax, rmin, rmax])
    # ax2.scatter(f2x[0], f2x[1], color='red')
    # ax2.set_title("$f2 \leq 5$")

    def handle_close(event):
        quit()

    fig, ax = plt.subplots(figsize=(9,7))
    fig.canvas.mpl_connect('close_event', handle_close)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    vals = rosen([X, Y])
    im = plt.imshow(vals, extent = [rmin, rmax, rmin, rmax], origin='lower')
    sc = plt.scatter(pop_hist[0, :, 0], pop_hist[0, :, 1], marker='.')

    for i in range(len(pop_hist)):
        im.set_data(vals)
        # plt.colorbar()
        # plt.scatter(f1x[0], f1x[1], color='red', label='constrained')
        # plt.scatter(xopt, yopt, color='green', label='unconstrained')


        sc.set_offsets(pop_hist[i, :])
        # plt.scatter(pop_hist[i, :, 0], pop_hist[i, :, 1], marker='.')
        plt.draw()

        plt.pause(0.2)


if __name__ == "__main__":
    main()



