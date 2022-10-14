import matplotlib.pyplot as plt
from numpy import size
fig = plt.figure(figsize=(8,10))
fig.tight_layout()
plt.rc('font', family='Times New Roman')
plt.subplots_adjust(wspace=0, hspace=0.5)
plt.subplot(3,1,1)
fz = 18
rt = 12
num_emotions = [1168, 1149, 739, 711, 620, 392]
emotions = ['neutral', 'frustrated', 'sad', 'anger', 'excited', 'happy']
plt.bar(list(range(6)),num_emotions, width=0.5)
plt.xticks(list(range(6)), emotions, fontsize=fz, rotation=rt)
plt.title('IEMOCAP trainset.', fontsize=fz)

plt.subplot(3,1,2)

num_emotions = [3035, 2184, 1285, 1076, 900, 784, 671]
emotions = ['neutral', 'joyful', 'scared', 'mad', 'peaceful','powerful',  'sad']
plt.bar(list(range(7)),num_emotions, width=0.5)
plt.xticks(list(range(7)), emotions, fontsize=fz, rotation=rt)
plt.title('EmoryNLP trainset.', fontsize=fz)

plt.subplot(3,1,3)

num_emotions = [4711, 1743, 1205, 1109, 683, 271, 268]
emotions = ['neutral', 'joy', 'suprise', 'anger', 'sadness', 'disgust','fear']
plt.bar(list(range(7)),num_emotions, width=0.5)
plt.xticks(list(range(7)), emotions, fontsize=fz, rotation=rt)
plt.title('MELD trainset.', fontsize=fz)
plt.savefig('./dist.pdf',bbox_inches='tight', dpi=600)
plt.show()
