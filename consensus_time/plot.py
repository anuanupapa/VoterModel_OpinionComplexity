import numpy as np
import matplotlib.pyplot as plt

data = np.load("consensusData_ER0.3.npz")

N=data["N"]
A=data["iniA"]
trials=data["trials"]
cons_all_arr=data["consensus"]
ctime_all_arr=data["ctime"]
aAB_arr=data["alphaAB"]
aBA_arr=data["alphaBA"]

consensusmean = np.mean(cons_all_arr, axis=-1)
ctimemean = np.mean(ctime_all_arr, axis=-1)
consensusstd = np.std(cons_all_arr, axis=-1)
ctimestd = np.std(ctime_all_arr, axis=-1)

plt.clf()
for ABind in range(len(aAB_arr)):
    plt.plot(aBA_arr, consensusmean[ABind,:], '.-',
             label=r'$\alpha_{AB}=$'+str(aAB_arr[ABind]))
    plt.fill_between(aBA_arr,
                     consensusmean[ABind,:]-consensusstd[ABind,:],
                     consensusmean[ABind,:]+consensusstd[ABind,:],
                     alpha=0.3)
plt.xlabel(r'$\alpha_{BA}$')
plt.ylabel("fracion of population for A at time 200")
plt.legend(loc=(0.02,0.02))
plt.savefig("consensus.png")
plt.show()


plt.clf()
for ABind in range(len(aAB_arr)):
    plt.plot(aBA_arr, ctimemean[ABind,:], '.-',
             label=r'$\alpha_{AB}=$'+str(aAB_arr[ABind]))
    plt.fill_between(aBA_arr,
                     ctimemean[ABind,:]-ctimestd[ABind,:],
                     ctimemean[ABind,:]+ctimestd[ABind,:],
                     alpha=0.3)
plt.xlabel(r'$\alpha_{BA}$')
plt.ylabel("Amount of time required to reach consensus")
plt.legend()
plt.savefig("time_consensus.png")
plt.show()

