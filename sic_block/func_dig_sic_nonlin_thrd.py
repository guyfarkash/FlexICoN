"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import time
import threading as thrd


global in_mutex, out_mutex
in_mutex = thrd.Lock()
out_mutex = thrd.Lock()


class sic(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block

    def __init__(self, frame_size=5000, preamble_size=600, k_size=30, order=3, frames=500, error_report=1, warning_report=0, information_report=0):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='SIC',   # will show up in GRC
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).

        if frame_size <= preamble_size:
            print('Error: Frame must be > Preamble !')
            exit(1)

        if preamble_size <= 2 * k_size:
            print('Error: Preamble must be > 2k !')
            exit(1)

        global fs, ps, k, o, f, er, wr, ir
        fs = int(frame_size)
        ps = int(preamble_size)
        k = int(k_size)
        o = int(order)
        f = int(frames)
        er = bool(error_report)
        wr = bool(warning_report)
        ir = bool(information_report)

        global in0_buff, in1_buff, out_buff, h
        in0_buff = []
        in1_buff = []
        out_buff = []
        h = np.zeros(2 * k)
        self.h = h

        global first, fc
        first = 1
        fc = 0

        self.thread = sic_thread()
        self.thread.setName('SIC')
        self.thread.start()
        # self.thread.join()

    def work(self, input_items, output_items):

        # global in_mutex, out_mutex, in0_buff, in1_buff, out_buff
        global in0_buff, in1_buff, out_buff

        t = time.time()
        in_mutex.acquire()
        in0_buff = np.append(in0_buff, input_items[0])
        in1_buff = np.append(in1_buff, input_items[1])
        in_mutex.release()
        in_buff_time = time.time() - t

        # generate output...
        out_len = len(output_items[0])
        out_buff_len = len(out_buff)

        out_buff_time = 0
        if out_buff_len >= out_len:
            # print('full'
            output_items[0][:] = out_buff[:out_len]

            out = out_buff[out_len:]
            t = time.time()
            out_mutex.acquire()
            # out_buff = out_buff[out_len:]
            out_buff = out
            out_mutex.release()
            out_buff_time = time.time() - t

            self.h = h
        # elif out_buff_len > 0:
        #     # print('some empty'
        #     output_items[0][:out_buff_len] = out_buff[:out_buff_len]
        #     out_mutex.acquire()
        #     out_buff = out_buff[out_buff_len:]
        #     out_mutex.release()
        #     out_len = out_buff_len
        else:
            # print('empty'
            output_items[0] = np.zeros(out_len)
            # return 0

        # self.h = h
        print('Input , Output (buffers mutex time) -> {}, {}'.format(in_buff_time, out_buff_time))

        return out_len


class sic_thread(thrd.Thread):
    def __init__(self):
        ''' Constructor. '''
        thrd.Thread.__init__(self)
        # self.vec = vec

    def run(self):

        global in_mutex, out_mutex
        global fs, ps, k, o, f, er, wr, ir
        global in0_buff, in1_buff, out_buff, h
        global first, fc

        done = 0
        count = 0
        while not(done):
            in_size = min(len(in0_buff), len(in1_buff))

            if in_size >= fs + 2 * k and not(first):  # not the first frame
                if fc % f == 0:
                    # find h
                    h = find_h_np(k, o, ps, in0_buff[: k + fs], in1_buff[: k + fs], wr)

                # calc the output
                A = build_A(in1_buff[:fs + 2 * k], k, o)
                res = in0_buff[k: k + fs] - np.dot(A, h)

                out_mutex.acquire()
                # out_buff = np.append(out_buff, in0_buff[k: k + fs] - np.dot(A, h))
                out_buff = np.append(out_buff, res)
                out_mutex.release()

                in_0 = in0_buff[fs:]
                in_1 = in1_buff[fs:]
                in_mutex.acquire()
                # in0_buff = in0_buff[fs:]  # keep last 2k samples for next time
                in0_buff = in_0
                # in1_buff = in1_buff[fs:]  # keep last 2k samples for next time
                in1_buff = in_1
                in_mutex.release()

                fc += 1

            elif in_size >= fs + k and first:  # the first frame
                if fc % f == 0:
                    # find h
                    h = find_h_np(k, o, ps, in0_buff[: fs], in1_buff[: fs], wr)

                # calc the output
                A = build_A(in1_buff[: fs + k], k, o)
                res = in0_buff[k: fs] - np.dot(A, h)

                out_mutex.acquire()
                out_buff = np.append(out_buff, in0_buff[:k])
                # out_buff = np.append(out_buff, in0_buff[k: fs] - np.dot(A, h))
                out_buff = np.append(out_buff, res)
                out_mutex.release()

                in_0 = in0_buff[fs - k:]
                in_1 = in1_buff[fs - k:]
                in_mutex.acquire()
                # in0_buff = in0_buff[fs - k:]  # keep last 2k samples for next time
                in0_buff = in_0
                # in1_buff = in1_buff[fs - k:]  # keep last 2k samples for next time
                in1_buff = in_1
                in_mutex.release()

                fc += 1
                first = 0

            elif not(first):
                count += 1
                time.sleep(0.5)
                in_size = min(len(in0_buff), len(in1_buff))
                if in_size < k + fs and count >= 5:
                    out_mutex.acquire()
                    out_buff = np.append(out_buff, in0_buff[:in_size])
                    out_mutex.release()
                    done = 1
                elif in_size >= k + fs:
                    count = 0

            elif first:
                count += 1
                time.sleep(0.5)
                in_size = min(len(in0_buff), len(in1_buff))
                if in_size < fs and count >= 5:
                    out_mutex.acquire()
                    out_buff = np.append(out_buff, in0_buff[:in_size])
                    out_mutex.release()
                    done = 1
                elif in_size >= fs:
                    count = 0

        print('SIC is done!')


def find_h_np(k, order, p, y, x, wr=0):
    # build matrix A from x (the transmitted signal)
    A = build_A(x[: p], k, order)
    # build weights vector h from matrix A and y (the received signal)
    # the beginning of the first input stream is (k + 1) the end is (k + 1 + 2k)
    h = np.linalg.lstsq(A, y[k: p - k])[0]
    if np.linalg.LinAlgError and wr:
        print('Warn: Least Squares Alg did not converge!')

    return h


def build_A(x, k, order):
    n = len(x) - 2 * k
    A = np.zeros((n, 2 * k * order), dtype=np.complex64)

    # A[0,0]...A[0,2k-1] = x[0].....x[2k-1]
    # A[1,0]...A[1,2k-1] = x[1].....x[1 + 2k-1]
    # ......................................
    # A[n-1,0]...A[1,2k] = x[n-1]...x[n-1 + 2k-1]
    # repeat for 2nd order
    # repeat for 3rd order
    # ...
    for i in range(0, n):
        for j in range(0, order):
            A[i, 2 * k * j: 2 * k * (j + 1)] = np.power(x[i: i + 2 * k], j + 1)
    return A
