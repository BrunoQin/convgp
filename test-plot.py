import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib2tikz import save as tikz_save

dfs = [
    (pd.read_pickle('./results/ocean-wconv20_hist.pkl'), "Conv - weighted"),
]

for df, descr in dfs:
    p = df[~df.acc.isnull()]
    plt.figure(10)
    plt.plot(p.t / 3600, p.acc * 100, label=descr)

    plt.figure(11)
    plt.plot(p.t / 3600, p.nlpp, label=descr)

    plt.figure(12)
    pss = p[~df.lml.isnull() & ~(df.lml == 0.0)]
    plt.plot(pss.t / 3600, pss.lml)

    print("%s\t acc: %f\tnlpp: %f" % (descr, p.iloc[-1].acc, p.iloc[-1].nlpp))
plt.figure(10)
plt.xlabel("Time (hrs)")
plt.ylabel("Test error (\%)")
plt.xlim([0, 14])
plt.ylim(1, 3)
plt.savefig('OCEAN_data/mnist-acc.png')
tikz_save('OCEAN_data/mnist-acc.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)

plt.figure(11)
plt.xlabel("Time (hrs)")
plt.ylabel("Test nlpp")
plt.xlim([0, 14])
plt.ylim([0.035, 0.12])
plt.savefig('OCEAN_data/mnist-nlpp.png')
tikz_save('OCEAN_data/mnist-nlpp.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)

plt.figure(12)
plt.xlabel("Time (hrs)")
plt.ylabel("Evidence lower bound")
plt.xlim([0, 14])
plt.ylim([-30000, -6000])
plt.savefig('OCEAN_data/mnist-bounds.png')
tikz_save('OCEAN_data/mnist-bounds.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)

plt.show()
