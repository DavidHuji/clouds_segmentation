import numpy as np
import matplotlib.pyplot as plt
def draw_result(lst_iter, lst_loss, lst_acc, title):
    plt.plot(lst_iter, lst_loss, '-b', label='loss')
    plt.plot(lst_iter, lst_acc, '-r', label='accuracy')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(title+".png")  # should before show method

    # show
    plt.show()


def test_draw():
    # iteration num
    lst_iter = range(20)

    # loss of iteration
    # lst_loss =
    lst_loss = [
 0.747342706,
 0.801952124,
 0.680964947,
 0.729423523,
 0.714049637,
 0.717229247,
 0.763305485,
 0.645006955,
 0.725889862,
 0.690941691,
 0.770473838,
 0.707231104,
 0.690755308,
 0.690408349,
 0.681188643,
 0.696964145,
 0.688203633,
 0.744002938,
 0.700892508,
 0.644649565]
    # lst_loss = np.random.randn(1, 100).reshape((100, ))

    # accuracy of iteration
    lst_acc = [
 0.762362957,
 0.773151994,
 0.77882719,
 0.760058999,
 0.716290951,
 0.744630158,
 0.799878061,
 0.753071785,
 0.693307877,
 0.663317263,
 0.68537575,
 0.71147728,
 0.775495648,
 0.710079789,
 0.733927488,
 0.736750007,
 0.639829516,
 0.778404593,
 0.722961843,
 0.676621199]
    # lst_acc = np.random.randn(1, 100).reshape((100, ))
    draw_result(lst_iter, lst_loss, lst_acc, "sgd_method")


if __name__ == '__main__':
    test_draw()
