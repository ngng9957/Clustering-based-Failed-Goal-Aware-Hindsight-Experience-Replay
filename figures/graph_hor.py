import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def data_process(data, size):
	means = []
	x = []
	stds = []
	for i in range(len(data)):
		means.append(np.mean(data[max(i - size, 0):i + 1]))
		x.append(i)
		stds.append(np.std(data[max(i - size, 0):i + 1]))
	stds = np.array(stds)

	return x, means, stds

def data_fill(data):
	temp = np.array(data[-10:])
	while len(data) < 200:
		data += temp[np.random.randint(10, size=10)].tolist()
	return data[:200]

tasks = ['FetchPush-v1','FetchPickAndPlace-v1','FetchSlide-v1']
titles = ['Push','PickAndPlace',"Slide"]
names = ['HER','FAHER','HER_EBP','FAHER_EBP']
names_l = ['HER','FAHER','HER+EBP','FAHER+EBP']
seeds = ['121','123','125','127','133']

data = np.zeros((len(tasks),len(names),5,200))
data2 = np.zeros((len(tasks),len(names),4,200))
for k in range(len(tasks)):

	for i in range(len(names)):
		for j in range(len(seeds)):
			with open(tasks[k]+"/"+names[i]+"/success_rates_"+seeds[j]+".pickle","rb") as fr:
				data_t = pickle.load(fr)[:200]
				data0 = data_fill(data_t)
				data[k,i,j] = np.array(data0)

		data0 = data[k,i].mean(0)
		x0, means0, stds0 = data_process(data0.tolist(), 20)
		data_M = data[k,i].max(0)
		x_M, means_M, stds_M = data_process(data_M.tolist(), 20)
		data_m = data[k,i].min(0)
		x_m, means_m, stds_m = data_process(data_m.tolist(), 20)

		data2[k,i,0] = means0
		data2[k,i,1] = x0
		data2[k,i,2] = means_m
		data2[k,i,3] = means_M

colors = ['blue','orange','green','red']
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728']
abc = ['(a)','(b)','(c)','(d)','(e)','(f)']
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
fig.subplots_adjust(left=0.035, right=0.99, bottom=0.19,top=0.95, wspace=0.15, hspace=0.3)
for i in range(2):
	for j in range(3):
		for spine in ax[i][j].spines.values():
		    spine.set_visible(False)
		ax[i][j].tick_params(top=False, bottom=False, left=False, right=False)
		ax[i][j].set_xlim(0,200)
		if i == 1 and j == 0:
			ax[i][j].set_xlim(0,100)
		ax[i][j].set_ylim(-0.05,1.05)
		ax[i][j].set_facecolor('#E7EAEF')
		ax[i][j].grid(True,color='white',alpha=0.7)
		for k in range(2):
			if i == 0:
				k2 = k
			else:
				k2 = k+2
			
			means0, x0, means_m, means_M = data2[j,k2]
			if j == 0:
				ax[i][j].plot(means0, label=names_l[k2], c=colors[k2],linewidth=3.0)
			else:
				ax[i][j].plot(means0, c=colors[k2],linewidth=3.0)
			ax[i][j].fill_between(x0, means_m, means_M, alpha=0.3, color=colors[k2], edgecolor="none")
		ax[i][j].set_xlabel('Epoch\n'+abc[i*3+j], size=20)
		if i == 0:
			ax[i][j].set_title(titles[j], size=30)
		# if j == 0:
		ax[i][j].set_ylabel('Success Rate',fontsize = 20)

fig.legend(loc="lower center", ncol = 4, bbox_to_anchor=(0.5, 0), fontsize = 30)
# plt.show()
plt.savefig('Result.png',dpi=600)
