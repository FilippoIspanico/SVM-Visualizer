import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

import warnings
warnings.filterwarnings("ignore")

fig, axes = plt.subplots()
n = 10

df = pd.DataFrame({'x': 3 * np.random.rand(n),
                   'y': 3 * np.random.rand(n),
                   'label': np.random.randint(0, 2, size=n)
                   })


axes.plot(df['x'][df['label'] == 0], df['y'][df['label'] == 0], 'o', )
axes.plot(df['x'][df['label'] == 1], df['y'][df['label'] == 1], 'o', c="red")
axes.set_xlim([0, 3])
axes.set_ylim([0, 3])
axes.set_title("SVM decision boundaries")
idx = None
label = None


def plot_svc():
    clf = svm.SVC(kernel="linear", gamma=2)

    X = df[['x', 'y']]
    y = df['label']

    clf.fit(X, y)

    common_params = {"estimator": clf, "X": X, "ax": axes}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )


plot_svc()


def get_closes_point(x: float, y: float, dataframe: pd.DataFrame, toll: float):
    """

    :param x:
    :param y:
    :param dataframe:
    :param toll:
    :return:
    """
    global idx, label

    point = (x, y)
    points = np.asarray(df[['x', 'y']])  #Extracting all dataframe features
    dist_2 = np.sum((points - point) ** 2, axis=1)  # Extract closes point in the features according to Euclidean distance
    idx = np.argmin(dist_2)
    distance = np.min(dist_2)
    label = df.at[idx, 'label']

    if distance < toll:
        return idx, label
    return None


def button_press_callback(event):
    'whenever a mouse button is pressed'
    global idx
    global label
    if event.inaxes is None:
        return
    if event.button != 1:
        return

    inv = axes.transData.inverted()
    x, y = inv.transform((event.x, event.y))
    result = get_closes_point(x, y, df, 0.05)
    if result is None:
        return
    idx, label = result

def motion_notify_callback(event):
    'on mouse movement'
    global idx
    if idx is None:
        return
    if event.inaxes is None:
        return
    if event.button != 1:
        return

    inv = axes.transData.inverted()
    x, y = inv.transform((event.x, event.y))
    df.at[idx, 'x'] = x
    df.at[idx, 'y'] = y
    # print(f"idx: {idx}, df[idx]: {df.at[idx, 'x'], df.at[idx, 'y']}")

    axes.clear()
    axes.plot(df['x'][df['label'] == 0], df['y'][df['label'] == 0], 'o')
    axes.plot(df['x'][df['label'] == 1], df['y'][df['label'] == 1], 'o', c="red")
    axes.set_xlim([0, 3])
    axes.set_ylim([0, 3])
    axes.set_title("SVM decision boundaries")
    plot_svc()
    fig.canvas.draw()

def button_release_callback(event):
    'whenever a mouse button is released'
    global idx
    if event.button != 1:
        return
    idx = None


fig.canvas.mpl_connect('button_press_event', button_press_callback)
fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)
fig.canvas.mpl_connect('button_release_event', button_release_callback)
plt.show()
