import numpy as np
from matplotlib import pyplot as plt
import cv2, csv


def show_hist_of_mask():
    mask = cv2.imread('./data/Masks/001_label.tif')
    plt.imshow(mask)
    plt.show()
    # mask = cv2.resize(mask, (100, 100), cv2.INTER_BITS)

    rng = np.random.RandomState(10)  # deterministic random data
    a = np.hstack((rng.normal(size=1000),
                   rng.normal(loc=5, scale=2, size=1000)))
    _ = plt.hist(mask.reshape(-1), bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    # Text(0.5, 1.0, "Histogram with 'auto' bins")
    plt.show()


def csv_to_np(pth):
    import pandas as pd
    WS = pd.read_excel(pth)
    WS_np = np.array(WS)
    return WS_np
    # res = []
    # with open(pth, newline='') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #     for row in spamreader:
    #         res.append(np.array(row))
    # return np.array(res)


def create_graphs(metrics_arr, save_path='', average=0):
    from matplotlib import pyplot as plt
    plt.xlabel('time (2h)')
    for k in range(1, metrics_arr.shape[1]):
        metric_val = list(metrics_arr[:, k].astype(float))
        if average != 0:
            metric_val = np.sum(np.array(metric_val +[0]).reshape((-1, average)), axis=0)

        plt.plot(np.arange(len(metric_val)), metric_val, 'o-', markersize=2, label=f'class_{k}')

    plt.ylabel('Area Percentage')
    plt.title(f'Area Percentage Analysis')
    plt.legend()
    if save_path:
        from pathlib import Path
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path) + 'graphs' + '.png')
        plt.close()
    else:
        plt.show()
    pass


def csv_to_graph(csv_path):
    np_arr = csv_to_np(csv_path)
    create_graphs(np_arr, 'October_grsph.png')

if __name__ == '__main__':
    csv_to_graph('Example4_2016_10.xlsx')