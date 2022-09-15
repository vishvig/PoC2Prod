import threading


def background(f):
    """
     a threading decorator
    use @background above the function you want to run in the background
    :param f:
    :return:
    """

    def bg_f(*a, **kw):
        thread = threading.Thread(target=f, args=a, kwargs=kw)
        thread.start()

    return bg_f
